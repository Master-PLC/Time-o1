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
        import tensorly as tl
        from tensorly.decomposition import partial_tucker
        tl.set_backend("pytorch")

        if isinstance(rank_ratio, (int, float)):
            rank_ratios = [rank_ratio] * 2
        else:
            rank_ratios = rank_ratio
        n_components_T = max(1, int(T * rank_ratios[0]))
        n_components_D = max(1, int(D * rank_ratios[1]))

        # ---- 转 torch ----
        if isinstance(data, np.ndarray):
            data_proc = torch.from_numpy(data).to(torch.float64)
        else:
            data_proc = data.to(torch.float64).clone() if reinit else data
        data_proc = data_proc.to('cuda')

        # ---- 可选标准化（分别在 mode-T 和 mode-D 展开上做）----
        initializer_T, initializer_D = [], []

        if reinit:
            # Mode-T: 把 T 当特征维 -> [N*D, T]
            data_T = data_proc.permute(0, 2, 1).reshape(-1, T)
            mean_T = data_T.mean(dim=0)
            std_T = data_T.std(dim=0, unbiased=False)
            std_T = torch.where(std_T == 0, torch.ones_like(std_T), std_T)
            initializer_T = [mean_T, std_T]  # 长度 T

            data_proc = (data_proc - mean_T[None, :, None]) / std_T[None, :, None]

            # Mode-D: 把 D 当特征维 -> [N*T, D]
            data_D = data_proc.reshape(-1, D)
            mean_D = data_D.mean(dim=0)
            std_D = data_D.std(dim=0, unbiased=False)
            std_D = torch.where(std_D == 0, torch.ones_like(std_D), std_D)
            initializer_D = [mean_D, std_D]  # 长度 D

            data_proc = (data_proc - mean_D[None, None, :]) / std_D[None, None, :]

        # ---- partial_tucker：只在 mode 1 (T) 和 mode 2 (D) 上分解 ----
        # ---- partial_tucker：显存峰值阶段，OOM 自动回退 CPU ----
        def _run_tucker(x):
            return partial_tucker(
                tl.tensor(x),
                modes=[1, 2],
                rank=[n_components_T, n_components_D],
                n_iter_max=100,
                tol=1e-6,
                init="svd",
            )

        try:
            result = _run_tucker(data_proc)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("[Tucker] CUDA OOM, falling back to CPU.")
            data_proc = data_proc.detach().cpu()
            if reinit:
                initializer_T = [t.cpu() for t in initializer_T]
                initializer_D = [t.cpu() for t in initializer_D]
            result = _run_tucker(data_proc)

        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], tuple):
            (core, factors) = result[0]
        else:
            (core, factors) = result

        U_T = factors[0]   # [T, R_T]
        U_D = factors[1]   # [D, R_D]
        core_t = core      # [N, R_T, R_D]

        base_T = U_T.transpose(0, 1).contiguous()  # [R_T, T]
        base_D = U_D.transpose(0, 1).contiguous()  # [R_D, D]

        core_sq = core_t ** 2
        total_energy = core_sq.sum() + 1e-12
        weights_T = torch.stack([
            core_sq[:, r, :].sum() / total_energy for r in range(n_components_T)
        ])
        weights_D = torch.stack([
            core_sq[:, :, r].sum() / total_energy for r in range(n_components_D)
        ])

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


data = torch.randn(1000, 96, 7)

get_pca_base(data, pca_dim="Tucker", rank_ratio=1.0, reinit=True)