import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

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


def standard_laguerre(data, degree):
    tvals = np.linspace(0, 5, len(data))
    coeffs = La.fit(tvals, data, degree).coef

    laguerre_poly = La(coeffs)
    reconstructed_data = laguerre_poly(tvals)
    return coeffs, reconstructed_data.reshape(-1)


def laguerre_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(0, 5, T)
    laguerre_polys = np.array([La.basis(i)(tvals) for i in range(degree)])

    laguerre_polys = torch.from_numpy(
        laguerre_polys).float().to(device)  # shape: [degree, T]
    # tvals = torch.from_numpy(tvals).float().to(device)
    # scale = torch.diag(torch.exp(-tvals))
    coeffs_candidate = torch.mm(laguerre_polys, data) / T
    coeffs = coeffs_candidate.transpose(0, 1)  # shape: [B * D, degree]
    # coeffs = torch.linalg.lstsq(laguerre_polys.T, data).solution.T

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, laguerre_polys)
        reconstructed_data = reconstructed_data.reshape(
            B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def standard_hermite(data, degree):
    tvals = np.linspace(-5, 5, len(data))
    coeffs = H.fit(tvals, data, degree).coef

    hermite_poly = H(coeffs)
    reconstructed_data = hermite_poly(tvals)
    return coeffs, reconstructed_data.reshape(-1)


def hermite_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-5, 5, T)
    hermite_polys = np.array([H.basis(i)(tvals) for i in range(degree)])

    hermite_polys = torch.from_numpy(
        hermite_polys).float().to(device)  # shape: [degree, T]
    # tvals = torch.from_numpy(tvals).float().to(device)
    # scale = torch.diag(torch.exp(-tvals ** 2))
    coeffs_candidate = torch.mm(hermite_polys, data) / T
    coeffs = coeffs_candidate.transpose(0, 1)  # shape: [B * D, degree]
    # coeffs = torch.linalg.lstsq(hermite_polys.T, data).solution.T

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, hermite_polys)
        reconstructed_data = reconstructed_data.reshape(
            B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def standard_leg(data, degree):
    tvals = np.linspace(-1, 1, len(data))
    coeffs = L.fit(tvals, data, degree).coef

    legendre_poly = L(coeffs)
    reconstructed_data = legendre_poly(tvals)
    return coeffs, reconstructed_data.reshape(-1)


def leg_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-1, 1, T)  # The Legendre series are defined in t\in[-1, 1]
    legendre_polys = np.array([L.basis(i)(tvals) for i in range(degree)])  # Generate the basis functions which are sampled at tvals.
    # tvals = torch.from_numpy(tvals).to(device)
    legendre_polys = torch.from_numpy(legendre_polys).float().to(device)  # shape: [degree, T]

    # This is implemented for 1D series. 
    # For N-D series, here, the data matrix should be transformed as B,T,D -> B,D,T -> BD, T. 
    # The legendre polys should be T,degree
    # Then, the dot should be a matrix multiplication: (BD, T) * (T, degree) -> BD, degree, which is the result of legendre transform.
    coeffs_candidate = torch.mm(legendre_polys, data) / T * 2
    coeffs = torch.stack([coeffs_candidate[i] * (2 * i + 1) / 2 for i in range(degree)]).to(device)
    coeffs = coeffs.transpose(0, 1)  # shape: [B * D, degree]

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, legendre_polys)
        reconstructed_data = reconstructed_data.reshape(B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def standard_chebyshev(data, degree):
    tvals = np.linspace(-1, 1, len(data))
    coeffs = C.fit(tvals, data, degree).coef

    chebyshev_poly = C(coeffs)
    reconstructed_data = chebyshev_poly(tvals)
    return coeffs, reconstructed_data.reshape(-1)


def chebyshev_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-1, 1, T)
    chebyshev_polys = np.array([C.basis(i)(tvals) for i in range(degree)])

    chebyshev_polys = torch.from_numpy(chebyshev_polys).float().to(device)  # shape: [degree, T]
    # tvals = torch.from_numpy(tvals).float().to(device)
    # scale = torch.diag(1 / torch.sqrt(1 - tvals ** 2))
    coeffs_candidate = torch.mm(chebyshev_polys, data) / torch.pi / T * 2
    # coeffs_candidate = torch.mm(torch.mm(chebyshev_polys, scale), data) / torch.pi * 2
    coeffs = coeffs_candidate.transpose(0, 1)  # shape: [B * D, degree]
    # coeffs = torch.linalg.lstsq(chebyshev_polys.T, data).solution.T

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, chebyshev_polys)
        reconstructed_data = reconstructed_data.reshape(B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


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


def ensure_array(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        # Handle cupy arrays and any other array-like (e.g. cuML outputs)
        # to avoid holding CUDA resources beyond the CUDA context lifetime.
        return _to_numpy(data)


def get_cca_projection(X, Y, rank_ratio=1.0, pca_dim="D", speedup_sklearn=0, align_type=0, add_noise=False):
    if speedup_sklearn in [0, 1]:
        from sklearn.cross_decomposition import CCA
    elif speedup_sklearn == 2:
        from utils.cca import CCA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # N, T, D = Y.shape
    D = Y.shape[-1]

    if pca_dim == "D":
        full_rank = D

    n_components = int(full_rank * rank_ratio)

    if pca_dim == "D":
        if align_type == 0:
            X = X.mean(axis=1)  # shape: [N, D]
            Y = Y.mean(axis=1)  # shape: [N, D]
        elif align_type == 1:
            X = X[:, -1]  # shape: [N, D]
            Y = Y[:, 0]  # shape: [N, D]
        elif align_type == 2:
            X = X[:, -1]  # shape: [N, D]
            Y = Y[:, -1]  # shape: [N, D]
        elif align_type == 3:
            X = X[:, 0]  # shape: [N, D]
            Y = Y[:, 0]  # shape: [N, D]
        elif align_type == 4:
            X = X.sum(axis=1)  # shape: [N, D]
            Y = Y.sum(axis=1)  # shape: [N, D]
        elif align_type == 5:
            pass
        elif align_type == 6:
            X = X[np.arange(X.shape[0]), np.random.randint(X.shape[1], size=X.shape[0])]  # shape: [N, D]
            Y = Y[np.arange(Y.shape[0]), np.random.randint(Y.shape[1], size=Y.shape[0])]  # shape: [N, D]

        if add_noise:
            X += np.random.normal(0, 0.005, X.shape)
            Y += np.random.normal(0, 0.005, Y.shape)

        cca = CCA(n_components=n_components) if speedup_sklearn in [0, 1] else CCA(n_components=n_components, device=device)
        cca.fit(X, Y)

        Wx = ensure_array(cca.x_rotations_)  # shape: [D, rank]
        Wy = ensure_array(cca.y_loadings_)  # shape: [D, rank]
        means = [ensure_array(cca._x_mean), ensure_array(cca._y_mean)]
        stds = [ensure_array(cca._x_std), ensure_array(cca._y_std)]

    else:
        raise NotImplementedError

    return Wx, Wy, means, stds


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


def get_fa_base(data, rank_ratio=1.0, pca_dim="all", reinit=0, speedup_sklearn=0):
    N, T, D = data.shape

    if pca_dim == "all":
        full_rank = T * D
    elif pca_dim == "T":
        full_rank = T
    elif pca_dim == "D":
        full_rank = D

    n_components = int(full_rank * rank_ratio)

    if pca_dim == "all":
        initializer = []
        data = data.reshape(N, -1)  # shape: [N, T * D]
        if reinit:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            initializer = [scaler.mean_, scaler.scale_]

        fa = FactorAnalysis(n_components=n_components, rotation='varimax')
        fa.fit(data)
        Wpsi = fa.components_ / fa.noise_variance_
        cov_z = linalg.inv(np.eye(n_components) + np.dot(Wpsi, fa.components_.T))
        base = np.dot(Wpsi.T, cov_z)  # shape: [rank, T * D]
        fa_mean = fa.mean_                   # shape: [T*D]

    elif pca_dim == "T":
        fa_components, initializer, fa_mean = [], [], []
        for d in range(D):
            chunk = data[..., d]  # shape: [N, T]
            if reinit:
                scaler = StandardScaler()
                chunk = scaler.fit_transform(chunk)
                initializer.append((scaler.mean_, scaler.scale_))
            fa = FactorAnalysis(n_components=n_components)
            fa.fit(chunk)
            Wpsi = fa.components_ / fa.noise_variance_
            cov_z = linalg.inv(np.eye(n_components) + np.dot(Wpsi, fa.components_.T))
            fa_components.append(np.dot(Wpsi.T, cov_z))  # shape: [rank, T]
            fa_mean.append(fa.mean_)              # shape: [T]

        if reinit:
            mean = np.array([pair[0] for pair in initializer])  # shape: [D, T]
            std = np.array([pair[1] for pair in initializer])  # shape: [D, T]
            initializer = [mean.transpose(1, 0), std.transpose(1, 0)]

        base = np.array(fa_components)  # shape: [D, rank, T]
        fa_mean = np.array(fa_mean).transpose(1, 0)                   # shape: [T, D]

    elif pca_dim == "D":
        fa_components, initializer, fa_mean = [], [], []
        for t in range(T):
            chunk = data[:, t]  # shape: [N, D]
            if reinit:
                scaler = StandardScaler()
                chunk = scaler.fit_transform(chunk)
                initializer.append((scaler.mean_, scaler.scale_))
            fa = FactorAnalysis(n_components=n_components)
            fa.fit(chunk)
            Wpsi = fa.components_ / fa.noise_variance_
            cov_z = linalg.inv(np.eye(n_components) + np.dot(Wpsi, fa.components_.T))
            fa_components.append(np.dot(Wpsi.T, cov_z))  # shape: [rank, D]
            fa_mean.append(fa.mean_)              # shape: [D]

        if reinit:
            mean = np.array([pair[0] for pair in initializer])  # shape: [T, D]
            std = np.array([pair[1] for pair in initializer])  # shape: [T, D]
            initializer = [mean, std]

        base = np.array(fa_components)  # shape: [T, rank, D]
        fa_mean = np.array(fa_mean)                   # shape: [T, D]

    else:
        raise NotImplementedError

    return base, initializer, fa_mean


def get_robustpca_base(data, rank_ratio=1.0, pca_dim="all", reinit=0):
    N, T, D = data.shape

    if pca_dim == "all":
        full_rank = T * D
    elif pca_dim == "T":
        full_rank = T
    elif pca_dim == "D":
        full_rank = D

    n_components = int(full_rank * rank_ratio)

    if pca_dim == "all":
        initializer = []
        data = data.reshape(N, -1)  # shape: [N, T * D]
        if reinit:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            initializer = [scaler.mean_, scaler.scale_]

        pca = RobustPCA(n_components=n_components)
        pca.fit(data)
        base = pca.components_  # shape: [rank, T * D]
        rpca_mean = pca.mean_                   # shape: [T*D]

    elif pca_dim == "T":
        pca_components, initializer, rpca_mean = [], [], []
        for d in range(D):
            chunk = data[..., d]  # shape: [N, T]
            if reinit:
                scaler = StandardScaler()
                chunk = scaler.fit_transform(chunk)
                initializer.append((scaler.mean_, scaler.scale_))
            pca = RobustPCA(n_components=n_components)
            pca.fit(chunk)
            pca_components.append(pca.components_)  # shape: [rank, T]
            rpca_mean.append(pca.mean_)              # shape: [T]

        if reinit:
            mean = np.array([pair[0] for pair in initializer])  # shape: [D, T]
            std = np.array([pair[1] for pair in initializer])  # shape: [D, T]
            initializer = [mean.transpose(1, 0), std.transpose(1, 0)]

        base = np.array(pca_components)  # shape: [D, rank, T]
        rpca_mean = np.array(rpca_mean).transpose(1, 0)                   # shape: [T, D]

    elif pca_dim == "D":
        pca_components, initializer, rpca_mean = [], [], []
        for t in range(T):
            chunk = data[:, t]  # shape: [N, D]
            if reinit:
                scaler = StandardScaler()
                chunk = scaler.fit_transform(chunk)
                initializer.append((scaler.mean_, scaler.scale_))
            pca = RobustPCA(n_components=n_components)
            pca.fit(chunk)
            pca_components.append(pca.components_)  # shape: [rank, D]
            rpca_mean.append(pca.mean_)              # shape: [D]

        if reinit:
            mean = np.array([pair[0] for pair in initializer])  # shape: [T, D]
            std = np.array([pair[1] for pair in initializer])  # shape: [T, D]
            initializer = [mean, std]

        base = np.array(pca_components)  # shape: [T, rank, D]
        rpca_mean = np.array(rpca_mean)                   # shape: [T, D]

    else:
        raise NotImplementedError

    return base, initializer, rpca_mean


def get_svd_base(data, rank_ratio=1.0, pca_dim="all", reinit=0):
    N, T, D = data.shape

    if pca_dim == "all":
        full_rank = T * D
    elif pca_dim == "T":
        full_rank = T
    elif pca_dim == "D":
        full_rank = D

    n_components = int(full_rank * rank_ratio)

    if pca_dim == "all":
        initializer = []
        data = data.reshape(N, -1)  # shape: [N, T * D]
        if reinit:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            initializer = [scaler.mean_, scaler.scale_]

        svd = TruncatedSVD(n_components=n_components)
        svd.fit(data)
        base = svd.components_  # shape: [rank, T * D]

    elif pca_dim == "T":
        svd_components, initializer = [], []
        for d in range(D):
            chunk = data[..., d]  # shape: [N, T]
            if reinit:
                scaler = StandardScaler()
                chunk = scaler.fit_transform(chunk)
                initializer.append((scaler.mean_, scaler.scale_))
            svd = TruncatedSVD(n_components=n_components)
            svd.fit(chunk)
            svd_components.append(svd.components_)  # shape: [rank, T]

        if reinit:
            mean = np.array([pair[0] for pair in initializer])  # shape: [D, T]
            std = np.array([pair[1] for pair in initializer])  # shape: [D, T]
            initializer = [mean.transpose(1, 0), std.transpose(1, 0)]

        base = np.array(svd_components)  # shape: [D, rank, T]

    elif pca_dim == "D":
        svd_components, initializer = [], []
        for t in range(T):
            chunk = data[:, t]  # shape: [N, D]
            if reinit:
                scaler = StandardScaler()
                chunk = scaler.fit_transform(chunk)
                initializer.append((scaler.mean_, scaler.scale_))
            svd = TruncatedSVD(n_components=n_components)
            svd.fit(chunk)
            svd_components.append(svd.components_)  # shape: [rank, D]

        if reinit:
            mean = np.array([pair[0] for pair in initializer])  # shape: [T, D]
            std = np.array([pair[1] for pair in initializer])  # shape: [T, D]
            initializer = [mean, std]

        base = np.array(svd_components)  # shape: [T, rank, D]

    else:
        raise NotImplementedError

    return base, initializer


def get_ica_base(data, rank_ratio=1.0, pca_dim="all", reinit=0):
    """
    提取 ICA base 和 initializer，仿照 PCA 的写法。
    data: np.ndarray of shape [N, T, D]
    """
    N, T, D = data.shape

    if pca_dim == "all":
        full_rank = T * D
    elif pca_dim == "T":
        full_rank = T
    elif pca_dim == "D":
        full_rank = D
    else:
        raise NotImplementedError

    n_components = int(full_rank * rank_ratio)

    if pca_dim == "all":
        initializer = []
        data_ = data.reshape(N, -1)  # shape: [N, T * D]
        if reinit:
            scaler = StandardScaler()
            data_ = scaler.fit_transform(data_)
            initializer = [scaler.mean_, scaler.scale_]

        ica = FastICA(n_components=n_components)
        ica.fit(data_)
        base = ica.components_  # [rank, T * D]
        ica_mean = ica.mean_                   # [T*D]
        whitening = ica.whitening_         # [rank, T*D]

    elif pca_dim == "T":
        ica_components, initializer, ica_mean, whitening = [], [], [], []
        for d in range(D):
            chunk = data[..., d]  # [N, T]
            if reinit:
                scaler = StandardScaler()
                chunk = scaler.fit_transform(chunk)
                initializer.append((scaler.mean_, scaler.scale_))
            ica = FastICA(n_components=n_components)
            ica.fit(chunk)
            ica_components.append(ica.components_)  # [rank, T]
            ica_mean.append(ica.mean_)              # [T]
            whitening.append(ica.whitening_)    # [rank, T]

        if reinit:
            mean = np.array([pair[0] for pair in initializer])  # [D, T]
            std = np.array([pair[1] for pair in initializer])   # [D, T]
            initializer = [mean.transpose(1, 0), std.transpose(1, 0)]

        base = np.array(ica_components)  # [D, rank, T]
        ica_mean = np.array(ica_mean).transpose(1, 0)                   # [T, D]
        whitening = np.array(whitening)         # [D, rank, T]

    elif pca_dim == "D":
        ica_components, initializer, ica_mean, whitening = [], [], [], []
        for t in range(T):
            chunk = data[:, t]  # [N, D]
            if reinit:
                scaler = StandardScaler()
                chunk = scaler.fit_transform(chunk)
                initializer.append((scaler.mean_, scaler.scale_))
            ica = FastICA(n_components=n_components)
            ica.fit(chunk)
            ica_components.append(ica.components_)  # [rank, D]
            ica_mean.append(ica.mean_)              # [D]
            whitening.append(ica.whitening_)    # [rank, D]

        if reinit:
            mean = np.array([pair[0] for pair in initializer])  # [T, D]
            std = np.array([pair[1] for pair in initializer])   # [T, D]
            initializer = [mean, std]

        base = np.array(ica_components)  # [T, rank, D]
        ica_mean = np.array(ica_mean)                   # [T, D]
        whitening = np.array(whitening)         # [T, rank, D]

    else:
        raise NotImplementedError

    return base, initializer, ica_mean, whitening


def get_robustica_base(data, rank_ratio=1.0, pca_dim="all", reinit=0):
    """
    提取 ICA base 和 initializer，仿照 PCA 的写法。
    data: np.ndarray of shape [N, T, D]
    """
    N, T, D = data.shape

    if pca_dim == "all":
        full_rank = T * D
    elif pca_dim == "T":
        full_rank = T
    elif pca_dim == "D":
        full_rank = D
    else:
        raise NotImplementedError

    n_components = int(full_rank * rank_ratio)
    rica_params = {
        "robust_runs": 10,
        "robust_method": "AgglomerativeClustering"
    }

    if pca_dim == "all":
        initializer = []
        data_ = data.reshape(N, -1)  # shape: [N, T * D]
        if reinit:
            scaler = StandardScaler()
            data_ = scaler.fit_transform(data_)
            initializer = [scaler.mean_, scaler.scale_]

        ica = RobustICA(n_components=n_components, **rica_params)
        S, A = ica.fit_transform(data_)
        base = linalg.pinv(A, check_finite=False)  # [rank, T * D]

    elif pca_dim == "T":
        ica_components, initializer = [], []
        for d in range(D):
            chunk = data[..., d]  # [N, T]
            if reinit:
                scaler = StandardScaler()
                chunk = scaler.fit_transform(chunk)
                initializer.append((scaler.mean_, scaler.scale_))
            ica = RobustICA(n_components=n_components, **rica_params)
            S, A = ica.fit_transform(chunk)
            ica_components.append(linalg.pinv(A, check_finite=False))  # [rank, T]

        if reinit:
            mean = np.array([pair[0] for pair in initializer])  # [D, T]
            std = np.array([pair[1] for pair in initializer])   # [D, T]
            initializer = [mean.transpose(1, 0), std.transpose(1, 0)]

        base = np.array(ica_components)  # [D, rank, T]

    elif pca_dim == "D":
        ica_components, initializer = [], []
        for t in range(T):
            chunk = data[:, t]  # [N, D]
            if reinit:
                scaler = StandardScaler()
                chunk = scaler.fit_transform(chunk)
                initializer.append((scaler.mean_, scaler.scale_))
            ica = RobustICA(n_components=n_components, **rica_params)
            S, A = ica.fit_transform(chunk)
            ica_components.append(linalg.pinv(A, check_finite=False))  # [rank, D]

        if reinit:
            mean = np.array([pair[0] for pair in initializer])  # [T, D]
            std = np.array([pair[1] for pair in initializer])   # [T, D]
            initializer = [mean, std]

        base = np.array(ica_components)  # [T, rank, D]

    else:
        raise NotImplementedError

    return base, initializer


def _to_cache_tensor(value, device='cpu'):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_to_cache_tensor(item, device) for item in value]
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [_to_cache_tensor(item, device) for item in value.tolist()]
    if isinstance(value, torch.Tensor):
        return value.float().to(device)
    return torch.from_numpy(value).float().to(device)


class Basis_Cache:
    def __init__(self, components, initializer=None, weights=None, mean=None, whitening=None, device='cpu'):
        self.components = _to_cache_tensor(components, device)
        self.initializer = _to_cache_tensor(initializer, device)
        self.weights = _to_cache_tensor(weights, device)
        self.mean = _to_cache_tensor(mean, device)
        self.whitening = _to_cache_tensor(whitening, device)


class Random_Cache:
    def __init__(self, rank_ratio, pca_dim, pred_len, enc_in, device='cpu'):
        if pca_dim == "all":
            rank = int(rank_ratio * enc_in * pred_len)
            self.components = torch.randn(rank, enc_in * pred_len, device=device)
        elif pca_dim == "T":
            rank = int(rank_ratio * pred_len)
            self.components = torch.randn(enc_in, rank, pred_len, device=device)
        elif pca_dim == "D":
            rank = int(rank_ratio * enc_in)
            self.components = torch.randn(pred_len, rank, enc_in, device=device)


def random_torch_inverse(low_rank_data, pca_dim, random_cache, pred_len, chan_inp=0, device='cpu'):
    B = low_rank_data.shape[0]

    pca_components = random_cache.components
    if pca_dim == "all":
        # components: [rank, T*D]
        # low_rank_data: [B, rank]
        rule_inv = 'br,rt->bt'
        data = torch.einsum(rule_inv, low_rank_data, pca_components)
        data = data.reshape(B, pred_len, -1)

    elif pca_dim == "T":
        # components: [D, rank, T]
        # low_rank_data: [B, rank, D]
        if not chan_inp:
            rule_inv = 'brd,rt->btd'
        else:
            rule_inv = 'brd,drt->btd'
        data = torch.einsum(rule_inv, low_rank_data, pca_components)

    elif pca_dim == "D":
        # components: [T, rank, D]
        # low_rank_data: [B, T, rank]
        rule_inv = 'btr,trd->btd'
        data = torch.einsum(rule_inv, low_rank_data, pca_components)

    return data


def random_torch(data, pca_dim, random_cache, chan_inp=0, device='cpu'):
    B, T, D = data.shape

    if pca_dim == "all":
        data = data.reshape(B, -1)  # reshape to B, TD

    pca_components = random_cache.components
    if pca_dim == "all":
        # pca_components shape: [rank, T*D]
        rule_trans = 'bt,rt->br'
    elif pca_dim == "T":
        # pca_components shape: [D, rank, T]
        if not chan_inp:
            pca_components = pca_components.mean(dim=0)  # shape: [rank, T]
            rule_trans = 'btd,rt->br'
        else:
            rule_trans = 'btd,drt->brd'
    elif pca_dim == "D":
        # pca_components shape: [T, rank, D]
        rule_trans = 'btd,trd->btr'

    low_rank_data = torch.einsum(rule_trans, data, pca_components)
    return low_rank_data


def pca_torch_inverse(low_rank_data, pca_dim, pca_cache, use_weights=0, reinit=True, pred_len=None, chan_indep=0, device='cpu'):
    B = low_rank_data.shape[0]

    pca_components = pca_cache.components
    if pca_dim == "all":
        # components: [rank, T*D]
        # low_rank_data: [B, rank],  weights shape: [rank]
        if use_weights:
            weights = pca_cache.weights
            if use_weights == 2:
                weights = torch.sqrt(weights)
            elif use_weights == 3:
                weights = torch.pow(weights, 2)
            # forward: 'br,r->br'  =>  inverse: element-wise divide along rank dim
            low_rank_data = low_rank_data / weights  # [B, rank] / [rank]
        rule_inv = 'br,rt->bt'
        data = torch.einsum(rule_inv, low_rank_data, pca_components)
        data = data.reshape(B, pred_len, -1)
    elif pca_dim == "T":
        # components: [D, rank, T]
        # low_rank_data: [B, rank, D],  weights shape: [D, rank]
        if use_weights:
            weights = pca_cache.weights
            if not chan_indep:
                weights = weights.mean(dim=0, keepdim=True)  # shape: [1, rank]
            if use_weights == 2:
                weights = torch.sqrt(weights)
            elif use_weights == 3:
                weights = torch.pow(weights, 2)
            # forward: 'brd,dr->brd'  =>  inverse: divide by weights transposed to [rank, D] = weights.T
            low_rank_data = low_rank_data / weights.T  # [B, rank, D] / [rank, D] or [B, rank, D] / [rank, 1]
        if not chan_indep:
            pca_components = pca_components.mean(dim=0)  # shape: [rank, T]
            rule_inv = 'brd,rt->btd'
        else:
            rule_inv = 'brd,drt->btd'
        data = torch.einsum(rule_inv, low_rank_data, pca_components)
    elif pca_dim == "D":
        # components: [T, rank, D]
        # low_rank_data: [B, T, rank],  weights shape: [T, rank]
        if use_weights:
            weights = pca_cache.weights
            if use_weights == 2:
                weights = torch.sqrt(weights)
            elif use_weights == 3:
                weights = torch.pow(weights, 2)
            # forward: 'btr,tr->btr'  =>  inverse: divide by weights [T, rank]
            low_rank_data = low_rank_data / weights  # [B, T, rank] / [T, rank]
        rule_inv = 'btr,trd->btd'
        data = torch.einsum(rule_inv, low_rank_data, pca_components)

    if reinit:
        mean, std = pca_cache.initializer
        data = data * std + mean

    return data


def pca_torch(data, pca_dim, pca_cache, use_weights=0, reinit=True, chan_indep=0, device='cpu'):
    B, T, D = data.shape

    if pca_dim == "Tucker":
        if reinit:
            (mean_T, std_T), (mean_D, std_D) = pca_cache.initializer
            data = (data - mean_T[None, :, None]) / std_T[None, :, None]
            data = (data - mean_D[None, None, :]) / std_D[None, None, :]

        component_T, component_D = pca_cache.components
        low_rank_data = torch.einsum('btd,rt,sd->brs', data, component_T, component_D)
        if use_weights:
            weight_T, weight_D = pca_cache.weights
            if use_weights == 2:
                weight_T = torch.sqrt(weight_T)
                weight_D = torch.sqrt(weight_D)
            elif use_weights == 3:
                weight_T = torch.pow(weight_T, 2)
                weight_D = torch.pow(weight_D, 2)
            low_rank_data = low_rank_data * weight_T[None, :, None] * weight_D[None, None, :]
        return low_rank_data

    if pca_dim == "all":
        data = data.reshape(B, -1)  # reshape to B, TD

    if reinit:
        mean, std = pca_cache.initializer  # shape: [T * D]
        data = (data - mean) / std

    pca_components = pca_cache.components
    if pca_dim == "all":
        # pca_components shape: [rank, T*D]
        rule_trans = 'bt,rt->br'
        rule_weight = 'br,r->br'
    elif pca_dim == "T":
        # pca_components shape: [D, rank, T]
        if not chan_indep:
            pca_components = pca_components.mean(dim=0)  # shape: [rank, T]
            rule_trans = 'btd,rt->brd'
            rule_weight = 'brd,r->brd'
        else:
            rule_trans = 'btd,drt->brd'
            rule_weight = 'brd,dr->brd'
    elif pca_dim == "D":
        # pca_components shape: [T, rank, D]
        if not chan_indep:
            pca_components = pca_components.mean(dim=0)  # shape: [rank, D]
            rule_trans = 'btd,rd->btr'
            rule_weight = 'btr,r->btr'
        else:
            rule_trans = 'btd,trd->btr'
            rule_weight = 'btr,tr->btr'

    low_rank_data = torch.einsum(rule_trans, data, pca_components)
    if use_weights:
        weights = pca_cache.weights
        if pca_dim == "T" and not chan_indep:
            weights = weights.mean(dim=0)  # shape: [rank]
        elif pca_dim == 'D' and not chan_indep:
            weights = weights.mean(dim=0)  # shape: [rank]

        if use_weights == 2:
            weights = torch.sqrt(weights)
        elif use_weights == 3:
            weights = torch.pow(weights, 2)
        low_rank_data = torch.einsum(rule_weight, low_rank_data, weights)

    return low_rank_data


def fa_torch_inverse(low_rank_data, pca_dim, fa_cache, reinit=True, pred_len=None, chan_indep=0, device='cpu'):
    B = low_rank_data.shape[0]

    pca_components = fa_cache.components
    if pca_dim == "all":
        # components: [rank, T*D]
        # low_rank_data: [B, rank]
        rule_inv = 'br,rt->bt'
        data = torch.einsum(rule_inv, low_rank_data, pca_components)
        data = data + fa_cache.mean
        data = data.reshape(B, pred_len, -1)
    elif pca_dim == "T":
        # components: [D, rank, T]
        # low_rank_data: [B, rank, D]
        if not chan_indep:
            pca_components = pca_components.mean(dim=0)  # shape: [rank, T]
            rule_inv = 'brd,rt->btd'
        else:
            rule_inv = 'brd,drt->btd'
        data = torch.einsum(rule_inv, low_rank_data, pca_components)
        data = data + fa_cache.mean
    elif pca_dim == "D":
        # components: [T, rank, D]
        # low_rank_data: [B, T, rank]
        rule_inv = 'btr,trd->btd'
        data = torch.einsum(rule_inv, low_rank_data, pca_components)
        data = data + fa_cache.mean

    if reinit:
        mean, std = fa_cache.initializer
        data = data * std + mean

    return data


def fa_torch(data, pca_dim, fa_cache, reinit=True, chan_indep=0, device='cpu'):
    B, T, D = data.shape

    if pca_dim == "all":
        data = data.reshape(B, -1)  # reshape to B, TD

    if reinit:
        mean, std = fa_cache.initializer  # shape: [T * D]
        data = (data - mean) / std

    pca_components = fa_cache.components
    data = data - fa_cache.mean
    if pca_dim == "all":
        # pca_components shape: [rank, T*D]
        rule_trans = 'bt,rt->br'
    elif pca_dim == "T":
        # pca_components shape: [D, rank, T]
        if not chan_indep:
            pca_components = pca_components.mean(dim=0)  # shape: [rank, T]
            rule_trans = 'btd,rt->brd'
        else:
            rule_trans = 'btd,drt->brd'
    elif pca_dim == "D":
        # pca_components shape: [T, rank, D]
        rule_trans = 'btd,trd->btr'

    low_rank_data = torch.einsum(rule_trans, data, pca_components)

    return low_rank_data


def robust_pca_torch_inverse(low_rank_data, pca_dim, pca_cache, reinit=True, pred_len=None, chan_indep=0, device='cpu'):
    B = low_rank_data.shape[0]

    pca_components = pca_cache.components
    if pca_dim == "all":
        # components: [rank, T*D]
        # low_rank_data: [B, rank]
        rule_inv = 'br,rt->bt'
        data = torch.einsum(rule_inv, low_rank_data, pca_components)
        data = data + pca_cache.mean
        data = data.reshape(B, pred_len, -1)
    elif pca_dim == "T":
        # components: [D, rank, T]
        # low_rank_data: [B, rank, D]
        if not chan_indep:
            pca_components = pca_components.mean(dim=0)  # shape: [rank, T]
            rule_inv = 'brd,rt->btd'
        else:
            rule_inv = 'brd,drt->btd'
        data = torch.einsum(rule_inv, low_rank_data, pca_components)
        data = data + pca_cache.mean
    elif pca_dim == "D":
        # components: [T, rank, D]
        # low_rank_data: [B, T, rank]
        rule_inv = 'btr,trd->btd'
        data = torch.einsum(rule_inv, low_rank_data, pca_components)
        data = data + pca_cache.mean

    if reinit:
        mean, std = pca_cache.initializer
        data = data * std + mean

    return data


def robust_pca_torch(data, pca_dim, pca_cache, reinit=True, chan_indep=0, device='cpu'):
    B, T, D = data.shape

    if pca_dim == "all":
        data = data.reshape(B, -1)  # reshape to B, TD

    if reinit:
        mean, std = pca_cache.initializer  # shape: [T * D]
        data = (data - mean) / std

    pca_components = pca_cache.components
    data = data - pca_cache.mean
    if pca_dim == "all":
        # pca_components shape: [rank, T*D]
        rule_trans = 'bt,rt->br'
    elif pca_dim == "T":
        # pca_components shape: [D, rank, T]
        if not chan_indep:
            pca_components = pca_components.mean(dim=0)  # shape: [rank, T]
            rule_trans = 'btd,rt->brd'
        else:
            rule_trans = 'btd,drt->brd'
    elif pca_dim == "D":
        # pca_components shape: [T, rank, D]
        rule_trans = 'btd,trd->btr'

    low_rank_data = torch.einsum(rule_trans, data, pca_components)

    return low_rank_data


def svd_torch_inverse(low_rank_data, pca_dim, svd_cache, reinit=True, pred_len=None, chan_indep=0, device='cpu'):
    B = low_rank_data.shape[0]

    svd_components = svd_cache.components
    if pca_dim == "all":
        # components: [rank, T*D]
        # low_rank_data: [B, rank]
        rule_inv = 'br,rt->bt'
        data = torch.einsum(rule_inv, low_rank_data, svd_components)
        data = data.reshape(B, pred_len, -1)
    elif pca_dim == "T":
        # components: [D, rank, T]
        # low_rank_data: [B, rank, D]
        if not chan_indep:
            svd_components = svd_components.mean(dim=0)  # shape: [rank, T]
            rule_inv = 'brd,rt->btd'
        else:
            rule_inv = 'brd,drt->btd'
        data = torch.einsum(rule_inv, low_rank_data, svd_components)
    elif pca_dim == "D":
        # components: [T, rank, D]
        # low_rank_data: [B, T, rank]
        rule_inv = 'btr,trd->btd'
        data = torch.einsum(rule_inv, low_rank_data, svd_components)

    if reinit:
        mean, std = svd_cache.initializer
        data = data * std + mean

    return data


def svd_torch(data, pca_dim, svd_cache, reinit=True, chan_indep=0, device='cpu'):
    B, T, D = data.shape

    if pca_dim == "all":
        data = data.reshape(B, -1)  # reshape to B, TD

    if reinit:
        mean, std = svd_cache.initializer  # shape: [T * D]
        data = (data - mean) / std

    svd_components = svd_cache.components
    if pca_dim == "all":
        # svd_components shape: [rank, T*D]
        rule_trans = 'bt,rt->br'
    elif pca_dim == "T":
        # svd_components shape: [D, rank, T]
        if not chan_indep:
            svd_components = svd_components.mean(dim=0)  # shape: [rank, T]
            rule_trans = 'btd,rt->brd'
        else:
            rule_trans = 'btd,drt->brd'
    elif pca_dim == "D":
        # svd_components shape: [T, rank, D]
        rule_trans = 'btd,trd->btr'

    low_rank_data = torch.einsum(rule_trans, data, svd_components)

    return low_rank_data


def ica_torch_inverse(low_rank_data, pca_dim, ica_cache, reinit=1, pred_len=None, chan_indep=0, device='cpu'):
    B = low_rank_data.shape[0]

    ica_components = ica_cache.components
    if pca_dim == "all":
        # components: [rank, T*D]
        # low_rank_data: [B, rank]
        rule_inv = 'br,rt->bt'
        data = torch.einsum(rule_inv, low_rank_data, ica_components)
        data = data + ica_cache.mean
        data = data.reshape(B, pred_len, -1)
    elif pca_dim == "T":
        # components: [D, rank, T]
        # low_rank_data: [B, rank, D]
        if not chan_indep:
            ica_components = ica_components.mean(dim=0)  # shape: [rank, T]
            rule_inv = 'brd,rt->btd'
        else:
            rule_inv = 'brd,drt->btd'
        data = torch.einsum(rule_inv, low_rank_data, ica_components)
        data = data + ica_cache.mean
    elif pca_dim == "D":
        # components: [T, rank, D]
        # low_rank_data: [B, T, rank]
        rule_inv = 'btr,trd->btd'
        data = torch.einsum(rule_inv, low_rank_data, ica_components)
        data = data + ica_cache.mean

    if reinit:
        mean, std = ica_cache.initializer
        data = data * std + mean

    return data


def ica_torch(data, pca_dim, ica_cache, reinit=1, chan_indep=0, device='cpu'):
    B, T, D = data.shape

    if pca_dim == "all":
        data = data.reshape(B, -1)  # reshape to B, TD

    if reinit:
        mean, std = ica_cache.initializer  # shape: [T * D]
        data = (data - mean) / std

    ica_components = ica_cache.components
    data = data - ica_cache.mean
    if pca_dim == "all":
        # pca_components shape: [rank, T*D]
        rule_trans = 'bt,rt->br'
    elif pca_dim == "T":
        # pca_components shape: [D, rank, T]
        if not chan_indep:
            ica_components = ica_components.mean(dim=0)  # shape: [rank, T]
            rule_trans = 'btd,rt->brd'
        else:
            rule_trans = 'btd,drt->brd'
    elif pca_dim == "D":
        # pca_components shape: [T, rank, D]
        rule_trans = 'btd,trd->btr'

    low_rank_data = torch.einsum(rule_trans, data, ica_components)

    return low_rank_data


def robust_ica_torch_inverse(low_rank_data, pca_dim, ica_cache, reinit=1, pred_len=None, chan_indep=0, device='cpu'):
    B = low_rank_data.shape[0]

    ica_components = ica_cache.components
    if pca_dim == "all":
        # components: [rank, T*D]
        # low_rank_data: [B, rank]
        rule_inv = 'br,rt->bt'
        data = torch.einsum(rule_inv, low_rank_data, ica_components)
        data = data.reshape(B, pred_len, -1)
    elif pca_dim == "T":
        # components: [D, rank, T]
        # low_rank_data: [B, rank, D]
        if not chan_indep:
            ica_components = ica_components.mean(dim=0)  # shape: [rank, T]
            rule_inv = 'brd,rt->btd'
        else:
            rule_inv = 'brd,drt->btd'
        data = torch.einsum(rule_inv, low_rank_data, ica_components)
    elif pca_dim == "D":
        # components: [T, rank, D]
        # low_rank_data: [B, T, rank]
        rule_inv = 'btr,trd->btd'
        data = torch.einsum(rule_inv, low_rank_data, ica_components)

    if reinit:
        mean, std = ica_cache.initializer
        data = data * std + mean

    return data


def robust_ica_torch(data, pca_dim, ica_cache, reinit=1, chan_indep=0, device='cpu'):
    B, T, D = data.shape

    if pca_dim == "all":
        data = data.reshape(B, -1)  # reshape to B, TD

    if reinit:
        mean, std = ica_cache.initializer  # shape: [T * D]
        data = (data - mean) / std

    ica_components = ica_cache.components
    if pca_dim == "all":
        # pca_components shape: [rank, T*D]
        rule_trans = 'bt,rt->br'
    elif pca_dim == "T":
        # pca_components shape: [D, rank, T]
        if not chan_indep:
            ica_components = ica_components.mean(dim=0)  # shape: [rank, T]
            rule_trans = 'btd,rt->brd'
        else:
            rule_trans = 'btd,drt->brd'
    elif pca_dim == "D":
        # pca_components shape: [T, rank, D]
        rule_trans = 'btd,trd->btr'

    low_rank_data = torch.einsum(rule_trans, data, ica_components)

    return low_rank_data


def evd_torch_inverse(low_rank_data, pca_dim, evd_cache, reinit=1, pred_len=None, chan_indep=0, device='cpu'):
    B = low_rank_data.shape[0]

    qmat = evd_cache.components  # shape: [D, T_tgt, T_src]
    if not chan_indep:
        rule_inv = 'bvd,vt->btd'
    else:
        rule_inv = 'bvd,dvt->btd'
    data = torch.einsum(rule_inv, low_rank_data, qmat)
    return data


def evd_torch(data, pca_dim, evd_cache, reinit=1, chan_indep=0, device='cpu'):
    B, T, D = data.shape

    qmat = evd_cache.components  # shape: [D, T_tgt, T_src]
    qmat = qmat.transpose(-1, -2)  # shape: [D, T_src, T_tgt]
    if not chan_indep:
        rule_trans = 'btd,tv->bvd'
    else:
        rule_trans = 'btd,dtv->bvd'
    low_rank_data = torch.einsum(rule_trans, data, qmat)
    return low_rank_data