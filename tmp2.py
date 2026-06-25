import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch

import numpy as np
from numpy.polynomial import Chebyshev as C
from numpy.polynomial import Hermite as H
from numpy.polynomial import Legendre as L
from numpy.polynomial import Laguerre as La
from robustica import RobustICA
from scipy import linalg
from sklearn.decomposition import FastICA, FactorAnalysis, TruncatedSVD
from sklearn.preprocessing import StandardScaler

from utils.rpca import RobustPCA


def _unfold(X, mode):
    """将张量 X 沿 mode 轴展开为矩阵 [size_mode, -1]"""
    return X.moveaxis(mode, 0).reshape(X.shape[mode], -1)


def _mode_product(X, M, mode):
    """mode-k 乘积：X ×_mode M，M shape [R, size_mode]，结果 mode 维替换为 R"""
    return torch.tensordot(M, X, dims=[[1], [mode]]).moveaxis(0, mode)


def _top_left_singvecs(M, rank, svd_impl="auto"):
    """
    取矩阵 M [d, big] 的前 rank 个左奇异向量, 返回 [d, rank].

    svd_impl:
      - "svd" : 直接 torch.linalg.svd (适合方阵 / 窄矩阵, 但 big 极大时
                cuSOLVER gesvdj 会 INVALID_VALUE / 显存爆)
      - "gram": 用 G = M M^T (shape [d,d]) + eigh, 只依赖 d, 不怕 big 大
      - "auto": big >= d 时走 gram, 否则走 svd  (HOOI unfold 几乎都是 big >> d)
    """
    d, big = M.shape[0], M.shape[1]
    use_gram = (svd_impl == "gram") or (svd_impl == "auto" and big >= d)

    if use_gram:
        # G = M M^T, [d, d]
        G = M @ M.T
        # eigh 升序, 翻转后取前 rank
        eigvals, eigvecs = torch.linalg.eigh(G)
        U = eigvecs.flip(-1)[:, :rank].contiguous()
        return U
    else:
        U, _, _ = torch.linalg.svd(M, full_matrices=False)
        return U[:, :rank].contiguous()


def _partial_tucker_torch(X, modes, ranks, n_iter_max=100, tol=1e-6,
                          svd_impl="auto"):
    """
    纯 PyTorch Partial HOOI Tucker 分解, 全程 GPU, float32.
    svd_impl 见 _top_left_singvecs.
    """
    # HOSVD 初始化
    factors = []
    for mode, rank in zip(modes, ranks):
        factors.append(_top_left_singvecs(_unfold(X, mode), rank, svd_impl))

    norm_X_sq = (X ** 2).sum().item()
    prev_core_norm_sq = None

    for _ in range(n_iter_max):
        for i, (mode, rank) in enumerate(zip(modes, ranks)):
            Y = X
            for j, (m, f) in enumerate(zip(modes, factors)):
                if j != i:
                    Y = _mode_product(Y, f.T, m)
            factors[i] = _top_left_singvecs(_unfold(Y, mode), rank, svd_impl)

        core = X
        for mode, f in zip(modes, factors):
            core = _mode_product(core, f.T, mode)

        core_norm_sq = (core ** 2).sum().item()
        if prev_core_norm_sq is not None:
            if norm_X_sq > 0 and abs(core_norm_sq - prev_core_norm_sq) / norm_X_sq < tol:
                break
        prev_core_norm_sq = core_norm_sq

    return core, factors


def get_pca_base(data, rank_ratio=1.0, pca_dim="all", reinit=0, speedup_sklearn=0):
    if speedup_sklearn in [0, 1]:
        from sklearn.decomposition import PCA
    elif speedup_sklearn == 2:
        from cuml.decomposition import PCA

    N, T, D = data.shape

    if pca_dim == "all":
        full_rank = T * D
    elif pca_dim == "T":
        full_rank = T
    elif pca_dim == "D":
        full_rank = D

    if pca_dim != "Tucker":
        n_components = int(full_rank * rank_ratio)

    if pca_dim == "all":
        initializer = []
        data = data.reshape(N, -1)  # shape: [N, T * D]
        if reinit:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            initializer = [scaler.mean_, scaler.scale_]

        pca = PCA(n_components=n_components)
        pca.fit(data)
        # Convert to numpy immediately to avoid cuML CUDA resource lifecycle issues
        # (cupy arrays holding CUDA events can be GC'd after CUDA context is destroyed)
        base = _to_numpy(pca.components_)      # shape: [rank, T * D]
        weights = _to_numpy(pca.explained_variance_ratio_)  # shape: [rank]
        del pca

    elif pca_dim == "T":
        pca_components, initializer, weights = [], [], []
        for d in range(D):
            chunk = data[..., d]  # shape: [N, T]
            if reinit:
                scaler = StandardScaler()
                chunk = scaler.fit_transform(chunk)
                initializer.append((scaler.mean_, scaler.scale_))
            pca = PCA(n_components=n_components)
            pca.fit(chunk)
            pca_components.append(_to_numpy(pca.components_))  # shape: [rank, T]
            weights.append(_to_numpy(pca.explained_variance_ratio_))  # shape: [rank]
            del pca

        if reinit:
            mean = np.array([pair[0] for pair in initializer])  # shape: [D, T]
            std = np.array([pair[1] for pair in initializer])  # shape: [D, T]
            initializer = [mean.transpose(1, 0), std.transpose(1, 0)]

        base = np.array(pca_components)  # shape: [D, rank, T]
        weights = np.array(weights)  # shape: [D, rank]

    elif pca_dim == "D":
        pca_components, initializer, weights = [], [], []
        for t in range(T):
            chunk = data[:, t]  # shape: [N, D]
            if reinit:
                scaler = StandardScaler()
                chunk = scaler.fit_transform(chunk)
                initializer.append((scaler.mean_, scaler.scale_))
            pca = PCA(n_components=n_components)
            pca.fit(chunk)
            pca_components.append(_to_numpy(pca.components_))  # shape: [rank, D]
            weights.append(_to_numpy(pca.explained_variance_ratio_))  # shape: [rank]
            del pca

        if reinit:
            mean = np.array([pair[0] for pair in initializer])  # shape: [T, D]
            std = np.array([pair[1] for pair in initializer])  # shape: [T, D]
            initializer = [mean, std]

        base = np.array(pca_components)  # shape: [T, rank, D]
        weights = np.array(weights)  # shape: [T, rank]

    elif pca_dim == "Tucker":
        if isinstance(rank_ratio, (int, float)):
            rank_ratios = [rank_ratio] * 2
        else:
            rank_ratios = rank_ratio
        assert len(rank_ratios) == 2, "Tucker 需要 (rT, rD) 两个比例"
        n_components_T = max(1, int(T * rank_ratios[0]))
        n_components_D = max(1, int(D * rank_ratios[1]))

        # float32: 比 float64 省一半显存, SVD/eigh 更快
        if isinstance(data, np.ndarray):
            data_proc = torch.from_numpy(data).to(torch.float32)
        else:
            data_proc = data.to(torch.float32).clone() if reinit else data.to(torch.float32)

        initializer_T, initializer_D = [], []

        if reinit:
            data_T = data_proc.permute(0, 2, 1).reshape(-1, T)
            mean_T = data_T.mean(dim=0)
            std_T = data_T.std(dim=0, unbiased=False)
            std_T = torch.where(std_T == 0, torch.ones_like(std_T), std_T)
            initializer_T = [mean_T, std_T]
            data_proc = (data_proc - mean_T[None, :, None]) / std_T[None, :, None]

            data_D = data_proc.reshape(-1, D)
            mean_D = data_D.mean(dim=0)
            std_D = data_D.std(dim=0, unbiased=False)
            std_D = torch.where(std_D == 0, torch.ones_like(std_D), std_D)
            initializer_D = [mean_D, std_D]
            data_proc = (data_proc - mean_D[None, None, :]) / std_D[None, None, :]

        def _run_tucker(x, svd_impl="auto"):
            return _partial_tucker_torch(
                x, modes=[1, 2], ranks=[n_components_T, n_components_D],
                n_iter_max=500, tol=1e-6, svd_impl=svd_impl,
            )

        if torch.cuda.is_available():
            data_proc = data_proc.cuda()

        # 用 Gram + eigh, 避免 [d, N*其它维] 这种超宽矩阵触发 cuSOLVER INVALID_VALUE
        try:
            core, factors = _run_tucker(data_proc, svd_impl="auto")
        except (getattr(torch, "OutOfMemoryError", RuntimeError),) as e:
            # 兼容老 torch (无 torch.OutOfMemoryError)
            if "out of memory" not in str(e).lower() and not isinstance(
                e, getattr(torch, "OutOfMemoryError", tuple())
            ):
                raise
            torch.cuda.empty_cache()
            print("[Tucker] CUDA OOM, falling back to CPU.")
            data_proc = data_proc.cpu()
            core, factors = _run_tucker(data_proc, svd_impl="auto")

        U_T = factors[0]   # [T, R_T]
        U_D = factors[1]   # [D, R_D]

        base_T = U_T.T.contiguous()   # [R_T, T]
        base_D = U_D.T.contiguous()   # [R_D, D]

        # 权重: 核张量每个分量在能量中的占比 (向量化版)
        core_sq = core ** 2
        total_energy = core_sq.sum() + 1e-12
        weights_T = core_sq.sum(dim=(0, 2)) / total_energy   # [R_T]
        weights_D = core_sq.sum(dim=(0, 1)) / total_energy   # [R_D]

        base = [_to_numpy(base_T), _to_numpy(base_D)]
        initializer = [
            [_to_numpy(x) for x in initializer_T],
            [_to_numpy(x) for x in initializer_D],
        ]
        weights = [_to_numpy(weights_T), _to_numpy(weights_D)]

    else:
        raise NotImplementedError

    return base, initializer, weights


def _to_numpy(arr):
    """Convert cuML/cupy arrays to plain numpy to avoid CUDA resource lifecycle issues.

    cuML returns cupy arrays that hold CUDA events internally.  If those arrays
    are GC'd *after* the CUDA context has been destroyed (e.g. at process exit),
    libraft's C++ destructors will call ``cudaEventDestroy`` on an invalid
    context, producing the error:
        "cudaEventDestroy initialization error"
    Calling this immediately after reading any attribute from a cuML estimator
    ensures the cupy array is converted to a plain numpy array and no CUDA
    resource is retained.
    """
    if hasattr(arr, 'to_numpy'):          # cuml DataFrame / Series
        return arr.to_numpy()
    if hasattr(arr, 'numpy'):            # cuml array
        return arr.detach().cpu().numpy()
    return np.asarray(arr)                # cupy ndarray, numpy ndarray, list …


data = torch.randn(10000, 720, 864)

get_pca_base(data, pca_dim="Tucker", rank_ratio=1.0, reinit=True)