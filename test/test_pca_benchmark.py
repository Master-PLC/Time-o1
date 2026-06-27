"""
PCA 方法基准测试
=================
针对 get_pca_base 的 5 种 pca_dim 模式 ("all", "T", "D", "Tucker", "KronPCA")
评测以下指标:
  1. 拟合时间 (fit time)
  2. 投影 + 逆投影时间 (forward + inverse time)
  3. 重建误差: MSE, MAE, RMSE
  4. 相对误差 (relative error)
  5. 解释方差占比 (explained variance ratio)
  6. 压缩率 (compression ratio = 原始参数量 / 低秩参数量)

运行方式:
  python -m test.test_pca_benchmark                          # 使用默认参数
  python -m test.test_pca_benchmark --N 200 --T 96 --D 7    # 自定义数据形状
  python -m test.test_pca_benchmark --rank_ratio 0.5         # 自定义 rank ratio
  python -m test.test_pca_benchmark --seed 42 --repeats 5    # 多次重复取均值
"""

import sys
import os
import time
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.polynomial import get_pca_base, pca_torch, pca_torch_inverse, Basis_Cache


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def generate_synthetic_data(N, T, D, seed=0):
    """生成带有低秩结构的合成时间序列数据 [N, T, D]。

    数据由一个秩-5 的信号 + 噪声构成, 以便 PCA 方法能有效压缩。
    """
    rng = np.random.RandomState(seed)
    rank = min(5, T, D)
    # 低秩信号
    U = rng.randn(N, rank)
    V_T = rng.randn(rank, T)
    V_D = rng.randn(rank, D)
    signal = np.einsum('nr,rt,rd->ntd', U, V_T, V_D)
    # 噪声
    noise = 0.1 * rng.randn(N, T, D)
    data = signal + noise
    return data.astype(np.float32)


def compute_reconstruction_metrics(original, reconstructed):
    """计算重建误差指标。

    Args:
        original: np.ndarray [N, T, D]
        reconstructed: np.ndarray [N, T, D]

    Returns:
        dict: 包含 MSE, MAE, RMSE, relative_error
    """
    diff = original - reconstructed
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    # 相对误差: ||X - X_hat||_F / ||X||_F
    rel_error = float(np.linalg.norm(diff) / (np.linalg.norm(original) + 1e-12))
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Relative_Error': rel_error,
    }


def compute_explained_variance(original, reconstructed):
    """计算重建解释方差占比。

    explained_variance = 1 - Var(X - X_hat) / Var(X)
    """
    residual_var = np.var(original - reconstructed)
    total_var = np.var(original)
    if total_var < 1e-12:
        return 1.0
    return float(1.0 - residual_var / total_var)


def compute_compression_ratio(N, T, D, pca_dim, rank_ratio):
    """计算理论压缩率。

    压缩率 = 原始参数量 / 低秩参数量
    """
    original_params = T * D  # 每个样本的参数量

    if pca_dim == "all":
        rank = max(1, int(T * D * rank_ratio))
        rank = min(rank, N, T * D)  # sklearn PCA 限制
        low_rank_params = rank
    elif pca_dim == "T":
        rank = max(1, int(T * rank_ratio))
        low_rank_params = rank * D
    elif pca_dim == "D":
        rank = max(1, int(D * rank_ratio))
        low_rank_params = T * rank
    elif pca_dim in ("Tucker", "KronPCA"):
        if isinstance(rank_ratio, (int, float)):
            rT, rD = rank_ratio, rank_ratio
        else:
            rT, rD = rank_ratio
        r_T = max(1, int(T * rT))
        r_D = max(1, int(D * rD))
        low_rank_params = r_T * r_D
    else:
        return 1.0

    ratio = original_params / max(low_rank_params, 1)
    return ratio


# ---------------------------------------------------------------------------
# Tucker / KronPCA 的逆变换 (pca_torch_inverse 不支持这两种模式)
# ---------------------------------------------------------------------------

def _kronecker_style_inverse(low_rank, pca_cache, pca_dim, reinit):
    """Tucker 和 KronPCA 的逆变换: X_recon = Z ×_T V_T ×_D V_D

    low_rank: [B, r_T, r_D]
    components: [r_T, T], [r_D, D]
    重建: X_recon[b, t, d] = Σ_{r,s} Z[b,r,s] * V_T[t,r] * V_D[d,s]
           = Z ×_1 V_T^T ×_2 V_D^T   (其中 V_T^T = component_T.T => [T, r_T])
    等价 einsum: 'brs, tr, ds -> btd'  (component_T 存储为 [r_T, T])
    """
    component_T, component_D = pca_cache.components  # [r_T, T], [r_D, D]
    # 逆投影: brs, rt -> 实际上需要 component_T.T 即 [T, r_T]
    V_T = component_T.T  # [T, r_T]
    V_D = component_D.T  # [D, r_D]
    data = torch.einsum('brs,tr,ds->btd', low_rank, V_T, V_D)
    # 暂时 float64 不需要

    if reinit:
        init = pca_cache.initializer
        if init and len(init) == 2 and len(init[0]) > 0 and len(init[1]) > 0:
            (mean_T, std_T), (mean_D, std_D) = init
            # 逆顺序: 先还原 D 维标准化, 再还原 T 维
            data = data * std_D[None, None, :] + mean_D[None, None, :]
            data = data * std_T[None, :, None] + mean_T[None, :, None]

    return data


# ---------------------------------------------------------------------------
# 重建函数: 从 get_pca_base 的输出进行 project -> inverse_project
# ---------------------------------------------------------------------------

def full_cycle_reconstruction(data_np, pca_dim, rank_ratio, reinit=1, device='cpu'):
    """执行完整的 fit -> project -> inverse_project 流程, 返回重建结果和耗时。

    Returns:
        reconstructed: np.ndarray [N, T, D]
        fit_time: float  (秒)
        cycle_time: float  (秒)  — project + inverse 的时间
    """
    N, T, D = data_np.shape

    # 对 "all" 模式, 限制 n_components 不超过 min(N, T*D)
    effective_rank_ratio = rank_ratio
    if pca_dim == "all":
        max_components = min(N, T * D)
        desired = int(T * D * rank_ratio)
        if desired > max_components:
            effective_rank_ratio = max_components / (T * D)

    # ---- 1. Fit: 计算基 ----
    t0 = time.perf_counter()
    base, initializer, weights = get_pca_base(
        data_np, rank_ratio=effective_rank_ratio, pca_dim=pca_dim, reinit=reinit,
        speedup_sklearn=0, pca_iter_max=200, pca_tol=1e-6,
    )
    fit_time = time.perf_counter() - t0

    # ---- 2. 构建 cache ----
    cache = Basis_Cache(base, initializer, weights, device=device)

    # ---- 3. Project + Inverse ----
    data_tensor = torch.from_numpy(data_np).float().to(device)

    t1 = time.perf_counter()
    low_rank = pca_torch(data_tensor, pca_dim, cache, use_weights=0, reinit=bool(reinit),
                         chan_indep=0, device=device)

    if pca_dim in ("Tucker", "KronPCA"):
        reconstructed = _kronecker_style_inverse(low_rank, cache, pca_dim, reinit=bool(reinit))
    elif pca_dim == "all" and reinit:
        # pca_torch_inverse 对 all+reinit 的顺序: 先 reshape [B,T,D] 再 * std + mean
        # 但 initializer 的 shape 是 [T*D], 需要先逆标准化再 reshape
        # 手动实现以规避维度不匹配
        pca_components = cache.components  # [rank, T*D]
        data_flat = torch.einsum('br,rt->bt', low_rank, pca_components)  # [B, T*D]
        mean, std = cache.initializer
        data_flat = data_flat * std + mean  # [B, T*D] * [T*D]
        reconstructed = data_flat.reshape(N, T, -1)
    else:
        reconstructed = pca_torch_inverse(low_rank, pca_dim, cache, use_weights=0, reinit=bool(reinit),
                                          pred_len=T, chan_indep=0, device=device)
    cycle_time = time.perf_counter() - t1

    reconstructed_np = reconstructed.detach().cpu().numpy()
    return reconstructed_np, fit_time, cycle_time


# ---------------------------------------------------------------------------
# 主测试逻辑
# ---------------------------------------------------------------------------

def run_benchmark(N=100, T=96, D=7, rank_ratio=0.5, repeats=3, seed=0, device='cpu'):
    """对所有 PCA 方法运行基准测试。

    返回一个 dict, key 是 pca_dim, value 是指标字典。
    """
    pca_methods = ["T", "D", "Tucker", "KronPCA"]
    results = OrderedDict()

    print("=" * 80)
    print(f"PCA 方法基准测试")
    print(f"数据形状: N={N}, T={T}, D={D}")
    print(f"Rank ratio: {rank_ratio},  重复次数: {repeats},  随机种子: {seed}")
    print(f"设备: {device}")
    print("=" * 80)

    for method in pca_methods:
        print(f"\n--- 测试 pca_dim='{method}' ---")
        fit_times = []
        cycle_times = []
        all_metrics = []

        for r in range(repeats):
            data_np = generate_synthetic_data(N, T, D, seed=seed + r)

            try:
                recon, ft, ct = full_cycle_reconstruction(
                    data_np, pca_dim=method, rank_ratio=rank_ratio,
                    reinit=1, device=device,
                )
                fit_times.append(ft)
                cycle_times.append(ct)

                metrics = compute_reconstruction_metrics(data_np, recon)
                metrics['Explained_Variance'] = compute_explained_variance(data_np, recon)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"  [repeat {r}] 失败: {e}")
                import traceback
                traceback.print_exc()

        if not all_metrics:
            print(f"  方法 '{method}' 全部失败, 跳过。")
            continue

        # 聚合结果
        avg_fit_time = np.mean(fit_times)
        avg_cycle_time = np.mean(cycle_times)
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        compression = compute_compression_ratio(N, T, D, method, rank_ratio)

        result = {
            'fit_time_mean': avg_fit_time,
            'fit_time_std': np.std(fit_times),
            'cycle_time_mean': avg_cycle_time,
            'cycle_time_std': np.std(cycle_times),
            'compression_ratio': compression,
            **avg_metrics,
        }
        results[method] = result

        print(f"  Fit 时间:    {avg_fit_time:.4f} ± {np.std(fit_times):.4f} s")
        print(f"  Cycle 时间:  {avg_cycle_time:.6f} ± {np.std(cycle_times):.6f} s")
        print(f"  MSE:         {avg_metrics['MSE']:.6e}")
        print(f"  MAE:         {avg_metrics['MAE']:.6e}")
        print(f"  RMSE:        {avg_metrics['RMSE']:.6e}")
        print(f"  Rel Error:   {avg_metrics['Relative_Error']:.6f}")
        print(f"  Explained V: {avg_metrics['Explained_Variance']:.6f}")
        print(f"  压缩率:      {compression:.2f}x")

    return results


def print_comparison_table(results):
    """以表格形式打印各方法对比。"""
    if not results:
        return

    print("\n" + "=" * 120)
    print(f"{'方法':<12} | {'Fit Time(s)':<14} | {'Cycle Time(s)':<14} | "
          f"{'MSE':<12} | {'MAE':<12} | {'Rel Error':<12} | "
          f"{'Expl. Var':<12} | {'压缩率':<8}")
    print("-" * 120)

    for method, r in results.items():
        print(f"{method:<12} | "
              f"{r['fit_time_mean']:<14.4f} | "
              f"{r['cycle_time_mean']:<14.6f} | "
              f"{r['MSE']:<12.4e} | "
              f"{r['MAE']:<12.4e} | "
              f"{r['Relative_Error']:<12.6f} | "
              f"{r['Explained_Variance']:<12.6f} | "
              f"{r['compression_ratio']:<8.2f}x")

    print("=" * 120)


def run_rank_ratio_sweep(N=100, T=96, D=7, seed=0, device='cpu'):
    """在不同 rank_ratio 下扫描, 观察精度-压缩率的权衡。"""
    ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
    pca_methods = ["all", "T", "D", "Tucker", "KronPCA"]

    print("\n" + "=" * 130)
    print("Rank Ratio 扫描 (单次运行)")
    print("=" * 130)

    data_np = generate_synthetic_data(N, T, D, seed=seed)

    header = f"{'Ratio':<8}"
    for m in pca_methods:
        header += f" | {m + ' MSE':<14} {m + ' RelErr':<14} {m + ' ExpVar':<14}"
    print(header)
    print("-" * len(header))

    for ratio in ratios:
        row = f"{ratio:<8.2f}"
        for method in pca_methods:
            try:
                recon, _, _ = full_cycle_reconstruction(
                    data_np, pca_dim=method, rank_ratio=ratio,
                    reinit=1, device=device,
                )
                metrics = compute_reconstruction_metrics(data_np, recon)
                exp_var = compute_explained_variance(data_np, recon)
                row += (f" | {metrics['MSE']:<14.4e} "
                        f"{metrics['Relative_Error']:<14.6f} "
                        f"{exp_var:<14.6f}")
            except Exception:
                row += f" | {'FAIL':<14} {'FAIL':<14} {'FAIL':<14}"
        print(row)

    print("=" * 130)


def run_scalability_test(device='cpu'):
    """测试不同数据规模下各方法的耗时, 验证可扩展性。"""
    configs = [
        (50,  24,  4),
        (100, 48,  7),
        (200, 96,  7),
        (500, 96,  14),
        (200, 192, 7),
    ]
    pca_methods = ["all", "T", "D", "Tucker", "KronPCA"]
    rank_ratio = 0.5

    print("\n" + "=" * 100)
    print("可扩展性测试 (rank_ratio=0.5)")
    print("=" * 100)

    header = f"{'(N,T,D)':<18}"
    for m in pca_methods:
        header += f" | {m + ' fit(s)':<14}"
    print(header)
    print("-" * len(header))

    for N, T, D in configs:
        data_np = generate_synthetic_data(N, T, D, seed=0)
        row = f"{str((N,T,D)):<18}"
        for method in pca_methods:
            try:
                _, ft, _ = full_cycle_reconstruction(
                    data_np, pca_dim=method, rank_ratio=rank_ratio,
                    reinit=1, device=device,
                )
                row += f" | {ft:<14.4f}"
            except Exception:
                row += f" | {'FAIL':<14}"
        print(row)

    print("=" * 100)


# ---------------------------------------------------------------------------
# pytest 兼容的测试用例
# ---------------------------------------------------------------------------

class TestPCAMethods:
    """可以通过 pytest 运行的测试类。

    运行方式:
      pytest test/test_pca_benchmark.py -v
    """

    N, T, D = 200, 24, 7  # N 需要 >= T*D*ratio 以避免 sklearn PCA 限制
    RANK_RATIO = 0.5
    DEVICE = 'cpu'

    def _get_data(self, seed=42):
        return generate_synthetic_data(self.N, self.T, self.D, seed=seed)

    # ---- 全秩重建精度测试 ----

    def test_pca_all_full_rank_reconstruction(self):
        """pca_dim='all' 全秩重建误差应接近 0。"""
        data = self._get_data()
        recon, _, _ = full_cycle_reconstruction(data, 'all', rank_ratio=1.0, device=self.DEVICE)
        metrics = compute_reconstruction_metrics(data, recon)
        assert metrics['Relative_Error'] < 0.05, f"全秩 'all' 相对误差过大: {metrics['Relative_Error']:.6f}"

    def test_pca_T_full_rank_reconstruction(self):
        """pca_dim='T' 全秩重建: chan_indep=0 时会平均各通道的 PCA 基,
        因此即使全秩, 重建也会有损, 但解释方差应较高。"""
        data = self._get_data()
        recon, _, _ = full_cycle_reconstruction(data, 'T', rank_ratio=1.0, device=self.DEVICE)
        exp_var = compute_explained_variance(data, recon)
        assert exp_var > 0.1, f"全秩 'T' 解释方差过低: {exp_var:.6f}"

    def test_pca_D_full_rank_reconstruction(self):
        """pca_dim='D' 全秩重建: chan_indep=0 时会平均各时间步的 PCA 基,
        因此即使全秩, 重建也会有损, 但解释方差应 > 0。"""
        data = self._get_data()
        recon, _, _ = full_cycle_reconstruction(data, 'D', rank_ratio=1.0, device=self.DEVICE)
        exp_var = compute_explained_variance(data, recon)
        assert exp_var > 0.0, f"全秩 'D' 解释方差过低: {exp_var:.6f}"

    def test_pca_Tucker_full_rank_reconstruction(self):
        """pca_dim='Tucker' 全秩重建误差应接近 0。"""
        data = self._get_data()
        recon, _, _ = full_cycle_reconstruction(data, 'Tucker', rank_ratio=1.0, device=self.DEVICE)
        metrics = compute_reconstruction_metrics(data, recon)
        assert metrics['Relative_Error'] < 0.05, f"全秩 'Tucker' 相对误差过大: {metrics['Relative_Error']:.6f}"

    def test_pca_KronPCA_full_rank_reconstruction(self):
        """pca_dim='KronPCA' 全秩重建误差应接近 0。"""
        data = self._get_data()
        recon, _, _ = full_cycle_reconstruction(data, 'KronPCA', rank_ratio=1.0, device=self.DEVICE)
        metrics = compute_reconstruction_metrics(data, recon)
        assert metrics['Relative_Error'] < 0.05, f"全秩 'KronPCA' 相对误差过大: {metrics['Relative_Error']:.6f}"

    # ---- 低秩重建: 有低秩信号时应比随机数据误差低 ----

    def test_low_rank_data_exploits_structure(self):
        """对有低秩结构的数据, 低秩 PCA 的 MSE 应远小于数据方差。"""
        data = self._get_data()
        data_var = np.var(data)
        for method in ["all", "T", "D"]:
            recon, _, _ = full_cycle_reconstruction(data, method, rank_ratio=0.5, device=self.DEVICE)
            metrics = compute_reconstruction_metrics(data, recon)
            # MSE 应显著小于数据总方差
            assert metrics['MSE'] < data_var, (
                f"方法={method}, MSE={metrics['MSE']:.4e} >= 数据方差={data_var:.4e}"
            )

    # ---- 单调性测试 ----

    def test_low_rank_reduces_error_monotonically(self):
        """随着 rank_ratio 增加, 'all' 模式重建误差应单调下降。"""
        data = self._get_data()
        prev_err = float('inf')
        for ratio in [0.1, 0.25, 0.5, 0.75, 1.0]:
            recon, _, _ = full_cycle_reconstruction(data, 'all', rank_ratio=ratio, device=self.DEVICE)
            err = compute_reconstruction_metrics(data, recon)['Relative_Error']
            assert err <= prev_err + 1e-3, (
                f"'all' 模式 rank_ratio={ratio} 的误差 {err:.6f} 大于 "
                f"更低 rank_ratio 的误差 {prev_err:.6f}"
            )
            prev_err = err

    # ---- 输出形状测试 ----

    def _safe_rank_ratio(self, pca_dim, rank_ratio):
        """限制 rank_ratio 以避免 sklearn PCA 的 n_components > min(N, features) 错误。"""
        if pca_dim == "all":
            max_comp = min(self.N, self.T * self.D)
            desired = int(self.T * self.D * rank_ratio)
            if desired > max_comp:
                return max_comp / (self.T * self.D)
        return rank_ratio

    def test_output_shapes_all(self):
        """pca_dim='all' 的投影和重建形状正确。"""
        data = self._get_data()
        ratio = self._safe_rank_ratio('all', 0.5)
        base, init, weights = get_pca_base(data, rank_ratio=ratio, pca_dim='all', reinit=1)
        cache = Basis_Cache(base, init, weights, device=self.DEVICE)
        tensor = torch.from_numpy(data).float()
        low = pca_torch(tensor, 'all', cache, reinit=True, device=self.DEVICE)
        expected_rank = int(self.T * self.D * ratio)
        assert low.shape == (self.N, expected_rank), f"投影形状错误: {low.shape}, 期望 ({self.N}, {expected_rank})"

    def test_output_shapes_T(self):
        """pca_dim='T' 的投影和重建形状正确。"""
        data = self._get_data()
        base, init, weights = get_pca_base(data, rank_ratio=0.5, pca_dim='T', reinit=1)
        cache = Basis_Cache(base, init, weights, device=self.DEVICE)
        tensor = torch.from_numpy(data).float()
        low = pca_torch(tensor, 'T', cache, reinit=True, device=self.DEVICE)
        rank = int(self.T * 0.5)
        assert low.shape == (self.N, rank, self.D), f"投影形状错误: {low.shape}"
        recon = pca_torch_inverse(low, 'T', cache, reinit=True, pred_len=self.T, device=self.DEVICE)
        assert recon.shape == (self.N, self.T, self.D), f"重建形状错误: {recon.shape}"

    def test_output_shapes_D(self):
        """pca_dim='D' 的投影和重建形状正确。"""
        data = self._get_data()
        base, init, weights = get_pca_base(data, rank_ratio=0.5, pca_dim='D', reinit=1)
        cache = Basis_Cache(base, init, weights, device=self.DEVICE)
        tensor = torch.from_numpy(data).float()
        low = pca_torch(tensor, 'D', cache, reinit=True, device=self.DEVICE)
        rank = int(self.D * 0.5)
        assert low.shape == (self.N, self.T, rank), f"投影形状错误: {low.shape}"
        recon = pca_torch_inverse(low, 'D', cache, reinit=True, pred_len=self.T, device=self.DEVICE)
        assert recon.shape == (self.N, self.T, self.D), f"重建形状错误: {recon.shape}"

    def test_output_shapes_Tucker(self):
        """pca_dim='Tucker' 的投影和逆变换形状正确。"""
        data = self._get_data()
        base, init, weights = get_pca_base(data, rank_ratio=0.5, pca_dim='Tucker', reinit=1)
        cache = Basis_Cache(base, init, weights, device=self.DEVICE)
        tensor = torch.from_numpy(data).float()
        low = pca_torch(tensor, 'Tucker', cache, reinit=True, device=self.DEVICE)
        r_T = max(1, int(self.T * 0.5))
        r_D = max(1, int(self.D * 0.5))
        assert low.shape == (self.N, r_T, r_D), f"投影形状错误: {low.shape}"
        recon = _kronecker_style_inverse(low, cache, 'Tucker', reinit=True)
        assert recon.shape == (self.N, self.T, self.D), f"重建形状错误: {recon.shape}"

    def test_output_shapes_KronPCA(self):
        """pca_dim='KronPCA' 的投影和逆变换形状正确。"""
        data = self._get_data()
        base, init, weights = get_pca_base(data, rank_ratio=0.5, pca_dim='KronPCA', reinit=1)
        cache = Basis_Cache(base, init, weights, device=self.DEVICE)
        tensor = torch.from_numpy(data).float()
        low = pca_torch(tensor, 'KronPCA', cache, reinit=True, device=self.DEVICE)
        r_T = max(1, int(self.T * 0.5))
        r_D = max(1, int(self.D * 0.5))
        assert low.shape == (self.N, r_T, r_D), f"投影形状错误: {low.shape}"
        recon = _kronecker_style_inverse(low, cache, 'KronPCA', reinit=True)
        assert recon.shape == (self.N, self.T, self.D), f"重建形状错误: {recon.shape}"

    # ---- 权重测试 ----

    def test_weights_normalization(self):
        """权重之和应接近 1 (各方法的 explained variance ratio)。

        对 pca_dim='T', weights shape=[D, rank], 每个 D 的权重独立 sum=1.
        对 pca_dim='D', weights shape=[T, rank], 每个 T 的权重独立 sum=1.
        对 pca_dim='all', weights shape=[rank], sum=1.
        """
        data = self._get_data()

        # all
        ratio = self._safe_rank_ratio('all', 1.0)
        _, _, weights = get_pca_base(data, rank_ratio=ratio, pca_dim='all', reinit=1)
        s = float(np.sum(weights))
        assert abs(s - 1.0) < 0.05, f"方法=all 权重之和 {s:.4f} 不接近 1"

        # T: weights shape [D, rank], 每行 sum=1
        _, _, weights = get_pca_base(data, rank_ratio=1.0, pca_dim='T', reinit=1)
        for d in range(weights.shape[0]):
            s = float(np.sum(weights[d]))
            assert abs(s - 1.0) < 0.05, f"方法=T, 第{d}通道权重之和 {s:.4f} 不接近 1"

        # D: weights shape [T, rank], 每行 sum=1
        _, _, weights = get_pca_base(data, rank_ratio=1.0, pca_dim='D', reinit=1)
        for t in range(weights.shape[0]):
            s = float(np.sum(weights[t]))
            assert abs(s - 1.0) < 0.05, f"方法=D, 第{t}时间步权重之和 {s:.4f} 不接近 1"

    def test_tucker_kronpca_weights_structure(self):
        """Tucker 和 KronPCA 的权重列表各有 2 个元素 (T 和 D 维度)。"""
        data = self._get_data()
        for method in ["Tucker", "KronPCA"]:
            _, _, weights = get_pca_base(data, rank_ratio=0.5, pca_dim=method, reinit=1)
            assert isinstance(weights, list), f"方法={method} 权重应为 list"
            assert len(weights) == 2, f"方法={method} 权重列表长度应为 2, 实际 {len(weights)}"

    def test_tucker_kronpca_base_structure(self):
        """Tucker 和 KronPCA 的 base 列表各有 2 个元素。"""
        data = self._get_data()
        for method in ["Tucker", "KronPCA"]:
            base, _, _ = get_pca_base(data, rank_ratio=0.5, pca_dim=method, reinit=1)
            assert isinstance(base, list), f"方法={method} base 应为 list"
            assert len(base) == 2, f"方法={method} base 列表长度应为 2, 实际 {len(base)}"
            r_T = max(1, int(self.T * 0.5))
            r_D = max(1, int(self.D * 0.5))
            assert base[0].shape == (r_T, self.T), f"base_T 形状错误: {base[0].shape}"
            assert base[1].shape == (r_D, self.D), f"base_D 形状错误: {base[1].shape}"

    # ---- reinit 参数测试 ----

    def test_reinit_no_reinit_consistency(self):
        """reinit=0 时不进行标准化, initializer 应为空列表。"""
        data = self._get_data()
        ratio = self._safe_rank_ratio('all', 0.5)
        _, init, _ = get_pca_base(data, rank_ratio=ratio, pca_dim='all', reinit=0)
        assert init == [] or init is None or (isinstance(init, list) and len(init) == 0), \
            f"reinit=0 时 initializer 应为空, 实际: {type(init)}, 值: {init}"

    def test_reinit_produces_normalizer(self):
        """reinit=1 时 initializer 不应为空。"""
        data = self._get_data()
        ratio = self._safe_rank_ratio('all', 0.5)
        _, init, _ = get_pca_base(data, rank_ratio=ratio, pca_dim='all', reinit=1)
        assert init is not None and len(init) > 0, \
            f"reinit=1 时 initializer 不应为空"

    # ---- 不同 rank_ratio 对比测试 ----

    def test_half_rank_beats_random(self):
        """rank_ratio=0.5 的 PCA 重建应优于随机投影。"""
        data = self._get_data()
        for method in ["all", "T", "D"]:
            recon, _, _ = full_cycle_reconstruction(data, method, rank_ratio=0.5, device=self.DEVICE)
            pca_err = compute_reconstruction_metrics(data, recon)['Relative_Error']
            # 随机重建 (用噪声数据作为对照)
            rng = np.random.RandomState(999)
            random_recon = rng.randn(*data.shape).astype(np.float32) * np.std(data) + np.mean(data)
            rand_err = compute_reconstruction_metrics(data, random_recon)['Relative_Error']
            assert pca_err < rand_err, (
                f"方法={method}, PCA 误差 {pca_err:.4f} >= 随机误差 {rand_err:.4f}"
            )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='PCA 方法基准测试')
    parser.add_argument('--N', type=int, default=10000, help='样本数')
    parser.add_argument('--T', type=int, default=96, help='时间步长')
    parser.add_argument('--D', type=int, default=7, help='通道数')
    parser.add_argument('--rank_ratio', type=float, default=1.0, help='保留比例')
    parser.add_argument('--repeats', type=int, default=3, help='重复次数')
    parser.add_argument('--seed', type=int, default=2023, help='随机种子')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cpu/cuda)')
    parser.add_argument('--sweep', action='store_true', help='执行 rank_ratio 扫描')
    parser.add_argument('--scalability', action='store_true', help='执行可扩展性测试')
    args = parser.parse_args()

    # 基础基准测试
    results = run_benchmark(
        N=args.N, T=args.T, D=args.D,
        rank_ratio=args.rank_ratio, repeats=args.repeats,
        seed=args.seed, device=args.device,
    )
    print_comparison_table(results)

    # 可选: rank_ratio 扫描
    if args.sweep:
        run_rank_ratio_sweep(N=args.N, T=args.T, D=args.D,
                             seed=args.seed, device=args.device)

    # 可选: 可扩展性测试
    if args.scalability:
        run_scalability_test(device=args.device)


if __name__ == '__main__':
    main()
