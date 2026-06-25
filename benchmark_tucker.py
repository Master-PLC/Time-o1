"""
benchmark_tucker.py
对比:
  - 自实现 (原始 SVD 版)        custom_svd
  - 自实现 (Gram + eigh 优化版) custom_gram
  - tensorly (pytorch backend)  tensorly
在 rank_ratio 接口下比较时间 / 重建误差 / 核能量比.
"""
import time
import torch
import tensorly as tl
from tensorly.decomposition import partial_tucker

# 兼容大矩阵: 换用 magma 也行, 但下面的 gram 版才是根治
# try:
#     torch.backends.cuda.preferred_linalg_library('magma')
# except Exception:
#     pass


# ---------------------------------------------------------------------------
# 基础算子
# ---------------------------------------------------------------------------
def _unfold(X, mode):
    return X.moveaxis(mode, 0).reshape(X.shape[mode], -1)

def _mode_product(X, M, mode):
    """X ×_mode M, M shape [R, size_mode]"""
    return torch.tensordot(M, X, dims=[[1], [mode]]).moveaxis(0, mode)


# ---------------------------------------------------------------------------
# 顶-R 左奇异向量: 两种实现
# ---------------------------------------------------------------------------
def _top_left_svd_svd(M, rank):
    """直接 SVD; 矩阵超宽时可能触发 cuSOLVER 错误"""
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    return U[:, :rank]

def _top_left_svd_gram(M, rank):
    """
    通过 Gram 矩阵 eigh 取顶-R 左奇异向量.
    M: [d, big]  ->  G = M M^T  [d, d], 仅依赖 d, 与 big 无关
    """
    d = M.shape[0]
    if M.shape[1] >= d:
        G = M @ M.T                                 # [d, d]
        # eigh 升序, 用 flip 转降序
        eigvals, eigvecs = torch.linalg.eigh(G)
        U = eigvecs.flip(-1)[:, :rank]
        return U.contiguous()
    else:
        # big < d 时, 走标准 SVD (一般不会发生在大数据展开里)
        U, _, _ = torch.linalg.svd(M, full_matrices=False)
        return U[:, :rank]


# ---------------------------------------------------------------------------
# Partial-HOOI Tucker (可切换 SVD 实现)
# ---------------------------------------------------------------------------
def _partial_tucker_torch(X, modes, ranks, n_iter_max=100, tol=1e-6, svd_impl='svd'):
    top_svd = _top_left_svd_gram if svd_impl == 'gram' else _top_left_svd_svd

    factors = []
    for mode, rank in zip(modes, ranks):
        factors.append(top_svd(_unfold(X, mode), rank))

    norm_X_sq = (X ** 2).sum().item()
    prev_core_norm_sq = None
    n_iter_done = 0
    for it in range(n_iter_max):
        for i, (mode, rank) in enumerate(zip(modes, ranks)):
            Y = X
            for j, (m, f) in enumerate(zip(modes, factors)):
                if j != i:
                    Y = _mode_product(Y, f.T, m)
            factors[i] = top_svd(_unfold(Y, mode), rank)

        core = X
        for mode, f in zip(modes, factors):
            core = _mode_product(core, f.T, mode)

        core_norm_sq = (core ** 2).sum().item()
        n_iter_done = it + 1
        if prev_core_norm_sq is not None:
            if norm_X_sq > 0 and abs(core_norm_sq - prev_core_norm_sq) / norm_X_sq < tol:
                break
        prev_core_norm_sq = core_norm_sq

    return core, factors, n_iter_done


# ---------------------------------------------------------------------------
# 重建
# ---------------------------------------------------------------------------
def reconstruct_custom(core, factors, modes):
    X_hat = core
    for mode, f in zip(modes, factors):
        X_hat = torch.tensordot(f, X_hat, dims=[[1], [mode]]).moveaxis(0, mode)
    return X_hat

def reconstruct_tensorly(core, factors, modes):
    X_hat = core
    for mode, f in zip(modes, factors):
        X_hat = tl.tenalg.mode_dot(X_hat, f, mode)
    return X_hat


# ---------------------------------------------------------------------------
# 计时辅助
# ---------------------------------------------------------------------------
def _timed(fn, n_runs, device):
    ts = []
    out = None
    for _ in range(n_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts), out


# ---------------------------------------------------------------------------
# 单组基准
# ---------------------------------------------------------------------------
def benchmark_one(shape, rank_ratio, device,
                  n_iter_max=50, tol=1e-7, seed=0, n_runs=2,
                  noise=0.05, run_tensorly=True, run_svd_version=True):
    torch.manual_seed(seed)
    N, T, D = shape

    if isinstance(rank_ratio, (int, float)):
        rT = rD = float(rank_ratio)
    else:
        rT, rD = float(rank_ratio[0]), float(rank_ratio[1])

    R_T = max(1, int(T * rT))
    R_D = max(1, int(D * rD))
    ranks = (R_T, R_D)
    modes = [1, 2]

    # 构造低秩 + 噪声张量 (真实秩 <= ranks, 便于看出非零误差)
    true_R_T = max(1, R_T // 2)
    true_R_D = max(1, R_D // 2)
    G_true = torch.randn(N, true_R_T, true_R_D, device=device, dtype=torch.float32)
    U_T = torch.linalg.qr(torch.randn(T, true_R_T, device=device))[0]
    U_D = torch.linalg.qr(torch.randn(D, true_R_D, device=device))[0]
    X = torch.einsum('nrs,tr,ds->ntd', G_true, U_T, U_D)
    X = X + noise * torch.randn_like(X)
    X = X.contiguous()
    X_norm = torch.linalg.norm(X).item()

    results = {'shape': shape, 'rank_ratio': (rT, rD), 'ranks': ranks,
               'device': str(device)}

    # ---- (A) custom (gram) ----
    try:
        t, (core_g, factors_g, iters_g) = _timed(
            lambda: _partial_tucker_torch(X, modes, ranks, n_iter_max, tol,
                                          svd_impl='gram'),
            n_runs, device,
        )
        X_hat = reconstruct_custom(core_g, factors_g, modes)
        results.update({
            'time_gram':     t,
            'iters_gram':    iters_g,
            'err_gram':      torch.linalg.norm(X - X_hat).item() / X_norm,
            'energy_gram':   (core_g ** 2).sum().item() / (X_norm ** 2),
        })
        del core_g, factors_g, X_hat
    except Exception as e:
        results['time_gram'] = None
        results['err_gram']  = f"FAIL: {type(e).__name__}"

    # ---- (B) custom (svd) 可能 OOM / cusolver fail ----
    if run_svd_version:
        try:
            t, (core_s, factors_s, iters_s) = _timed(
                lambda: _partial_tucker_torch(X, modes, ranks, n_iter_max, tol,
                                              svd_impl='svd'),
                n_runs, device,
            )
            X_hat = reconstruct_custom(core_s, factors_s, modes)
            results.update({
                'time_svd':   t,
                'iters_svd':  iters_s,
                'err_svd':    torch.linalg.norm(X - X_hat).item() / X_norm,
                'energy_svd': (core_s ** 2).sum().item() / (X_norm ** 2),
            })
            del core_s, factors_s, X_hat
        except Exception as e:
            results['time_svd'] = None
            results['err_svd']  = f"FAIL: {type(e).__name__}"

    # ---- (C) tensorly ----
    if run_tensorly:
        tl.set_backend('pytorch')
        try:
            t, out = _timed(
                lambda: partial_tucker(X, rank=list(ranks), modes=modes,
                                       n_iter_max=n_iter_max, tol=tol, init='svd'),
                n_runs, device,
            )
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], tuple):
                core_b, factors_b = out[0]
            elif isinstance(out, tuple) and len(out) == 2:
                core_b, factors_b = out
            else:
                core_b, factors_b = out.core, out.factors
            X_hat = reconstruct_tensorly(core_b, factors_b, modes)
            results.update({
                'time_tl':   t,
                'err_tl':    torch.linalg.norm(X - X_hat).item() / X_norm,
                'energy_tl': (core_b ** 2).sum().item() / (X_norm ** 2),
            })
            del core_b, factors_b, X_hat
        except Exception as e:
            results['time_tl'] = None
            results['err_tl']  = f"FAIL: {type(e).__name__}"

    del X
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _fmt_t(x):
    return f"{x:8.3f}s" if isinstance(x, (int, float)) else f"{str(x):>9}"

def _fmt_e(x):
    return f"{x:10.3e}" if isinstance(x, (int, float)) else f"{str(x):>10}"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"== device = {device} ==\n")

    cases = [
        ((10000, 720,   7), 1.0),
        ((10000, 720, 174), 1.0),
        ((10000, 720, 322), 1.0),
        ((10000, 720, 864), 1.0),
    ]

    header = (f"{'shape':>22} {'ratio':>10} {'ranks':>12} | "
              f"{'t_gram':>10} {'t_svd':>10} {'t_tl':>10} | "
              f"{'err_gram':>10} {'err_svd':>10} {'err_tl':>10}")
    print(header)
    print("-" * len(header))

    for shape, ratio in cases:
        r = benchmark_one(shape, ratio, device,
                          n_iter_max=20, tol=1e-6, n_runs=1,
                          run_tensorly=True, run_svd_version=True)
        print(
            f"{str(r['shape']):>22} {str(r['rank_ratio']):>10} "
            f"{str(r['ranks']):>12} | "
            f"{_fmt_t(r.get('time_gram'))} "
            f"{_fmt_t(r.get('time_svd'))} "
            f"{_fmt_t(r.get('time_tl'))} | "
            f"{_fmt_e(r.get('err_gram'))} "
            f"{_fmt_e(r.get('err_svd'))} "
            f"{_fmt_e(r.get('err_tl'))}"
        )

if __name__ == "__main__":
    main()
