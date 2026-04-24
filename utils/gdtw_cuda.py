## adpated from https://github.com/samcohen16/Aligning-Time-Series

import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import torch

import torch.nn as nn
import torch.nn.functional as F

# 假设你的附件3和附件4分别保存为了 dtw_cuda.py 和 soft_dtw_cuda.py
# 如果合并在一个文件里，请直接引用类名
from utils.dtw_cuda import DTW, _DTWCUDA, _DTW
from utils.soft_dtw_cuda import SoftDTW, _SoftDTWCUDA, _SoftDTW


class GromovDTW(nn.Module):
    def __init__(self, max_iter=5, gamma=1.0, solver='soft', 
                 bandwidth=0.1,  # <--- 关键修改：默认开启带宽限制 (0.1 表示 10% 的窗口)
                 device='cuda', tol=1e-3, verbose=False):
        super(GromovDTW, self).__init__()
        self.max_iter = max_iter
        self.gamma = gamma
        self.solver = solver
        self.bandwidth = bandwidth
        self.verbose = verbose
        self.tol = tol
        self.device = device
        
        use_cuda = (device != 'cpu' and device != torch.device('cpu'))
        
        # 初始化求解器
        # 注意：这里我们将在 forward 中动态处理 float64，所以这里的初始化主要是为了保留配置
        self.sdtw_solver = SoftDTW(use_cuda=use_cuda, gamma=gamma, bandwidth=bandwidth)
        self.dtw_solver = DTW(use_cuda=use_cuda, bandwidth=bandwidth)

    def _pairwise_euclidean_dist(self, x):
        """
        计算欧氏距离矩阵，增加数值稳定性保护
        """
        # x: [B, T, D]
        x_norm = (x**2).sum(2).view(x.shape[0], x.shape[1], 1)
        y_norm = x_norm.view(x.shape[0], 1, x.shape[1])
        dist = x_norm + y_norm - 2.0 * torch.bmm(x, x.transpose(1, 2))
        
        # 关键：Clamp防止出现负数（计算误差导致）产生NaN
        return torch.clamp(dist, min=1e-6)

    def init_alignment_batch(self, B, N, M, device, dtype):
        """
        初始化对齐矩阵，仅初始化带宽内的区域以节省显存 (逻辑上)
        实际实现为了简单仍初始化全图，但在带宽限制下CUDA只会访问带宽内
        """
        A = torch.zeros(B, N, M, device=device, dtype=dtype)
        
        # 简单对角线初始化
        indices_n = torch.arange(N, device=device)
        indices_m = torch.floor(indices_n * M / N).long()
        indices_m = torch.clamp(indices_m, 0, M - 1)
        A[:, indices_n, indices_m] = 1.0
        A[:, 0, 0] = 1.0
        A[:, -1, -1] = 1.0
        return A

    def tensor_product_batch(self, C1, C2, A):
        # 转换为 float32 进行矩阵乘法以节省显存（如果精度允许），或者保持 float64
        # 这里为了防NaN，建议尽量保持与输入一致的精度
        C1_sq = C1 ** 2
        C2_sq = C2 ** 2
        
        p = A.sum(dim=2, keepdim=True)
        term1 = torch.bmm(C1_sq, p)
        
        q = A.sum(dim=1, keepdim=True)
        term2 = torch.bmm(C2_sq, q.transpose(1, 2)).transpose(1, 2)
        
        term3 = -2.0 * torch.bmm(torch.bmm(C1, A), C2)
        
        return term1 + term2 + term3

    def solve_linear_assignment(self, D):
        # 必须开启梯度以获取对齐矩阵 A
        D_var = D.detach().clone().requires_grad_(True)
        
        if self.solver == 'soft':
            if self.sdtw_solver.use_cuda:
                func = _SoftDTWCUDA.apply
            else:
                func = _SoftDTW.apply
            
            # 使用 float64 运行 SoftDTW 以防止 NaN
            loss_val = func(D_var, self.gamma, self.bandwidth)
        else:
            if self.dtw_solver.use_cuda:
                func = _DTWCUDA.apply
            else:
                func = _DTW.apply
            loss_val = func(D_var, self.bandwidth)

        # 获取对齐矩阵 A
        alignment_A = torch.autograd.grad(loss_val.sum(), D_var, create_graph=True)[0]
        return alignment_A

    def forward(self, x, y):
        """
        Input: [B, T, D]
        """
        # --- 修复 NaN 策略 1: 输入归一化 ---
        # 如果时序数据数值很大，距离矩阵会巨大，导致 exp 溢出
        # 对每个样本在时间维度进行 Instance Norm
        x = F.instance_norm(x.transpose(1, 2)).transpose(1, 2)
        y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        # --- 修复 NaN 策略 2: 强制 Double 精度 (Float64) ---
        # 仅在计算 GDTW 内部逻辑时使用 Double，显存增加但稳定性极大提升
        orig_dtype = x.dtype
        x = x.double()
        y = y.double()

        B, Tx, _ = x.shape
        _, Ty, _ = y.shape
        
        # 1. 计算结构内距离矩阵
        C1 = self._pairwise_euclidean_dist(x)
        C2 = self._pairwise_euclidean_dist(y)
        
        # 2. 初始化
        A = self.init_alignment_batch(B, Tx, Ty, x.device, dtype=torch.float64)
        
        # 3. 迭代
        for i in range(self.max_iter):
            # 显式释放显存：每次迭代前清理不需要的图
            # 注意：A 需要保留用于下一次迭代，但中间变量可以覆盖
            
            Tens = self.tensor_product_batch(C1, C2, A)
            
            # 解决 OOM: Detach 掉之前的计算图，只保留当前的 A
            # Frank-Wolfe 算法通常只需要当前的 Tensor Product
            A = A.detach() 
            
            A_new = self.solve_linear_assignment(Tens)
            A = A_new # Update
        
        # 4. Final Loss
        final_Tens = self.tensor_product_batch(C1, C2, A)
        
        if self.solver == 'soft':
             if self.sdtw_solver.use_cuda:
                func = _SoftDTWCUDA.apply
             else:
                func = _SoftDTW.apply
             # 确保使用 double
             loss = func(final_Tens, self.gamma, self.bandwidth)
             loss = loss.mean()
        else:
            loss = (final_Tens * A).sum(dim=(1, 2)).mean()
            
        # 归一化 Loss，防止梯度过大 (T*T)
        scale = Tx
        loss = loss / scale
        
        # 转回原始精度 (例如 float32 或 float16) 以兼容后续网络
        return loss.to(dtype=orig_dtype)


# 使用示例
if __name__ == "__main__":
    # 模拟数据 [Batch, Time, Dim]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, D = 32, 96, 7
    pred = torch.randn(B, T, D, device=device, requires_grad=True)
    target = torch.randn(B, T, D, device=device)
    
    # 实例化 GDTW 模块
    # 注意：你需要确保 dtw_cuda.py 和 soft_dtw_cuda.py 在路径中
    criterion = GromovDTW(max_iter=10, gamma=0.1, solver='soft', device=device, verbose=True)
    
    # 前向传播
    loss = criterion(pred, target)
    print(f"GDTW Loss: {loss.item()}")
    
    # 反向传播
    loss.backward()
    print("Backward pass successful. Pred grad norm:", pred.grad.norm().item())
