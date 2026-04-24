import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import torch

from utils.soft_dtw_cuda import _SoftDTWCUDA


def pairwise_distances_cuda(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y if y is not None else x, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1) if y is not None else x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))


class DilateLossCUDA(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=1.0, bandwidth=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)

    def forward(self, outputs, targets):
        device = outputs.device
        batch_size, N, n_vars = outputs.shape

        D = torch.zeros((batch_size, N, N), device=device, dtype=outputs.dtype)
        for k in range(batch_size):
            D[k] = pairwise_distances_cuda(targets[k], outputs[k])

        # 一次调用，同时获得loss和路径概率矩阵E
        D_requires_grad = D.detach().requires_grad_(True)
        loss_shape = _SoftDTWCUDA.apply(D_requires_grad, self.gamma, self.bandwidth)

        # E (路径概率矩阵) 通过 backward 计算得出
        E = torch.autograd.grad(loss_shape, D_requires_grad, retain_graph=True)[0].detach()

        Omega = pairwise_distances_cuda(
            torch.arange(1, N+1, device=device, dtype=outputs.dtype).view(N, 1)
        )

        loss_temporal = torch.sum(E * Omega) / (N * N * batch_size)

        loss = self.alpha * loss_shape + (1 - self.alpha) * loss_temporal

        return loss