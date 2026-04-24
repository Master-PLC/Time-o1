import torch

from numba import cuda
from torch.autograd import Function
import torch.nn as nn


# ==============================================================================
# 1. Forward Kernel (保持不变)
# ==============================================================================
@cuda.jit
def ldtw_step_kernel(prev_cost, D, curr_cost, choices, N, M, current_k, bandwidth):
    b = cuda.blockIdx.z
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + 1
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1

    if i <= N and j <= M:
        if abs(i - j) > bandwidth * max(N, M) and bandwidth > 0:
             curr_cost[b, i, j] = 1e9 
             return

        c_diag = prev_cost[b, i-1, j-1]
        c_vert = prev_cost[b, i-1, j]
        c_hori = prev_cost[b, i, j-1]

        min_val = c_diag
        choice = 0 

        if c_vert < min_val:
            min_val = c_vert
            choice = 1 
        
        if c_hori < min_val:
            min_val = c_hori
            choice = 2 
        
        if min_val >= 1e8:
            curr_cost[b, i, j] = 1e9
            choices[b, current_k, i, j] = -1 
        else:
            # D 在传入前已经被 detach，这里只是读取数值
            curr_cost[b, i, j] = min_val + D[b, i-1, j-1]
            choices[b, current_k, i, j] = choice

# ==============================================================================
# 2. Backward Kernel (保持不变)
# ==============================================================================
@cuda.jit
def ldtw_backtrack_kernel(choices, grad_map, best_k_indices, N, M):
    b = cuda.blockIdx.x
    curr_i = N
    curr_j = M
    curr_k = best_k_indices[b]
    
    grad_map[b, curr_i, curr_j] = 1.0
    
    for k in range(curr_k, 0, -1):
        decision = choices[b, k, curr_i, curr_j]
        if decision == 0:   # Diagonal
            curr_i -= 1
            curr_j -= 1
        elif decision == 1: # Vertical
            curr_i -= 1
        elif decision == 2: # Horizontal
            curr_j -= 1
        else:
            break 
            
        if curr_i > 0 and curr_j > 0:
            grad_map[b, curr_i, curr_j] = 1.0

# ==============================================================================
# 3. PyTorch Function (修正重点在这里)
# ==============================================================================
class _LDTWHardLowMem(Function):
    @staticmethod
    def forward(ctx, D, max_len, bandwidth):
        B, N, M = D.shape
        dev = D.device
        
        # 初始化 tensor，默认 requires_grad=False，但为了安全起见，后面也会转 cuda array
        prev_cost = torch.ones((B, N+2, M+2), device=dev, dtype=torch.float32) * 1e9
        curr_cost = torch.ones((B, N+2, M+2), device=dev, dtype=torch.float32) * 1e9
        
        # 初始化起点
        prev_cost.fill_(1e9)
        prev_cost[:, 0, 0] = 0.0
        
        # choices 使用 int8
        choices = torch.full((B, max_len + 1, N + 1, M + 1), -1, device=dev, dtype=torch.int8)
        final_costs = torch.full((B, max_len + 1), 1e9, device=dev)

        threads_per_block = (16, 16)
        blocks_per_grid_x = (N + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (M + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, B)

        # 【修正 1】将不需要求导的 Tensor 转换为 CUDA Array Interface
        # D 是必须 detach() 的，因为它在计算图中
        D_cuda = cuda.as_cuda_array(D.detach()) 
        prev_cost_cuda = cuda.as_cuda_array(prev_cost)
        curr_cost_cuda = cuda.as_cuda_array(curr_cost)
        choices_cuda = cuda.as_cuda_array(choices)

        for k in range(1, max_len + 1):
            ldtw_step_kernel[blocks_per_grid, threads_per_block](
                prev_cost_cuda, 
                D_cuda, 
                curr_cost_cuda, 
                choices_cuda, 
                N, M, k, bandwidth
            )
            
            final_costs[:, k] = curr_cost[:, N, M]
            prev_cost.copy_(curr_cost)
        
        min_valid_len = max(N, M)
        if min_valid_len <= max_len:
            valid_final_costs = final_costs[:, min_valid_len:]
            min_vals, min_idx_rel = torch.min(valid_final_costs, dim=1)
            best_k_indices = min_idx_rel + min_valid_len
        else:
            return torch.zeros(B, device=dev), None

        ctx.save_for_backward(choices, best_k_indices)
        ctx.dims = (N, M)
        
        return min_vals

    @staticmethod
    def backward(ctx, grad_output, l=None):
        choices, best_k_indices = ctx.saved_tensors
        N, M = ctx.dims
        B = choices.shape[0]
        
        grad_map = torch.zeros((B, N + 1, M + 1), device=grad_output.device, dtype=torch.float32)
        
        # 【修正 2】Backward 也要 detach 或确保 tensor 无梯度需求
        # choices, best_k_indices 本身是 saved_tensors，不带梯度
        # grad_map 是新创建的，也不带梯度
        ldtw_backtrack_kernel[B, 1](
            cuda.as_cuda_array(choices), 
            cuda.as_cuda_array(grad_map), 
            cuda.as_cuda_array(best_k_indices), 
            N, M
        )
        
        grad_map = grad_map[:, 1:, 1:]
        
        # 这里进行链式法则，PyTorch 会自动处理这里的梯度连接
        grad_input = grad_output.view(-1, 1, 1) * grad_map
        
        return grad_input, None, None

# ==============================================================================
# 4. LDTWHard Module (保持不变)
# ==============================================================================
class LDTW(nn.Module):
    def __init__(self, max_length, bandwidth=1.0, use_cuda=True):
        super(LDTW, self).__init__()
        self.max_length = max_length
        self.bandwidth = bandwidth
        self.use_cuda = use_cuda 

    def _calc_distance_matrix(self, x, y):
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        N, M = X.shape[1], Y.shape[1]
        curr_max = self.max_length
        if curr_max is None: curr_max = N + M
        
        D = self._calc_distance_matrix(X, Y)
        
        # 调用 apply，注意 D 是带有梯度的
        return _LDTWHardLowMem.apply(D, curr_max, self.bandwidth), None


# ==============================================================================
# 5. 测试代码
# ==============================================================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("--- Starting Low-Memory Hard-LDTW Test ---")
        
        # 模拟显存敏感场景：Batch=16, Length=200
        # 如果用 float32 4D tensor: 16*200*200*300*4 bytes ≈ 768 MB (不算太大，但如果是 L=500 就会很大)
        # 如果用 int8 4D tensor: 192 MB -> 显著节省
        B, N, M, Dim = 16, 100, 100, 128
        max_len = 120 # 比较紧的约束 (对角线是100)
        
        x = torch.randn(B, N, Dim, device='cuda', requires_grad=True)
        y = torch.randn(B, M, Dim, device='cuda')
        
        model = LDTW(max_length=max_len, bandwidth=0.5)
        
        # Forward
        loss = model(x, y).mean()
        print(f"Loss (Hard constraint max_len={max_len}): {loss.item():.4f}")
        
        # Backward
        loss.backward()
        print("Backward pass successful.")
        print(f"Gradient shape: {x.grad.shape}")
        print(f"Gradient non-zero elements: {x.grad.nonzero().size(0)} (sparsity check)")
        
        # 验证约束是否生效
        # 假如 max_len 设置得极小 (例如 < 100)，应该返回 0 或报错(代码里处理为返回0或inf)
        # 这里只是演示显存优化逻辑
