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


def _estimate_kronecker_covariance(data, reinit=0):
    """
    估计 Kronecker 结构的协方差矩阵 Σ_T 和 Σ_D。

    假设 TD×TD 协方差矩阵具有 Kronecker 结构 Σ ≈ Σ_T ⊗ Σ_D，
    分别沿 T 和 D 维度估计边际协方差矩阵。

    Args:
        data: numpy 或 torch 张量, shape [N, T, D]
        reinit: 是否进行标准化 (0 或 1)

    Returns:
        sigma_T: [T, T] 协方差矩阵
        sigma_D: [D, D] 协方差矩阵
        initializer_T: [mean_T, std_T] 标准化参数 (list, 如果 reinit=0 则为空)
        initializer_D: [mean_D, std_D] 标准化参数 (list, 如果 reinit=0 则为空)
        data_proc: 预处理后的数据张量 (用于后续的 Flip-Flop 迭代)
    """
    N, T, D = data.shape
    initializer_T, initializer_D = [], []

    if isinstance(data, np.ndarray):
        data_proc = torch.from_numpy(data).to(torch.float32)
    else:
        data_proc = data.to(torch.float32).clone() if reinit else data.to(torch.float32)

    if torch.cuda.is_available():
        data_proc = data_proc.cuda()

    # ---------- 标准化 ----------
    if reinit:
        # T 维标准化: reshape to [N*D, T]
        data_T = data_proc.permute(0, 2, 1).reshape(-1, T)
        mean_T = data_T.mean(dim=0)
        std_T = data_T.std(dim=0, unbiased=False)
        std_T = torch.where(std_T == 0, torch.ones_like(std_T), std_T)
        initializer_T = [mean_T, std_T]
        data_proc = (data_proc - mean_T[None, :, None]) / std_T[None, :, None]

        # D 维标准化: reshape to [N*T, D]
        data_D = data_proc.reshape(-1, D)
        mean_D = data_D.mean(dim=0)
        std_D = data_D.std(dim=0, unbiased=False)
        std_D = torch.where(std_D == 0, torch.ones_like(std_D), std_D)
        initializer_D = [mean_D, std_D]
        data_proc = (data_proc - mean_D[None, None, :]) / std_D[None, None, :]

    # ---------- 计算 Σ_T: 时间维协方差 ----------
    # 将数据重塑为 [N*D, T], 每行是一个通道在某样本下的时间序列
    data_T = data_proc.permute(0, 2, 1).reshape(-1, T)
    data_T_centered = data_T - data_T.mean(dim=0, keepdim=True)
    sigma_T = (data_T_centered.T @ data_T_centered) / max(data_T.shape[0] - 1, 1)
    sigma_T = sigma_T + 1e-6 * torch.eye(T, device=sigma_T.device, dtype=sigma_T.dtype)
    sigma_T = (sigma_T + sigma_T.T) / 2

    # ---------- 计算 Σ_D: 通道维协方差 ----------
    # 将数据重塑为 [N*T, D], 每行是一个时间步在某样本下的通道向量
    data_D = data_proc.reshape(-1, D)
    data_D_centered = data_D - data_D.mean(dim=0, keepdim=True)
    sigma_D = (data_D_centered.T @ data_D_centered) / max(data_D.shape[0] - 1, 1)
    sigma_D = sigma_D + 1e-6 * torch.eye(D, device=sigma_D.device, dtype=sigma_D.dtype)
    sigma_D = (sigma_D + sigma_D.T) / 2

    return sigma_T, sigma_D, initializer_T, initializer_D, data_proc


def _kronecker_eigen_decomposition(sigma_T, sigma_D, rank_ratio_T, rank_ratio_D):
    """
    对 Σ_T 和 Σ_D 分别做对称特征分解, 取前 r_T 和 r_D 个特征向量.

    理论保证:
        若 Σ = Σ_T ⊗ Σ_D, 则投影矩阵 P = V_T ⊗ V_D 满足
        P^T Σ P = Λ_T ⊗ Λ_D  (对角矩阵)

    Args:
        sigma_T: [T, T] 对称半正定协方差矩阵
        sigma_D: [D, D] 对称半正定协方差矩阵
        rank_ratio_T: T 维度保留比例 (0, 1]
        rank_ratio_D: D 维度保留比例 (0, 1]

    Returns:
        V_T: [T, r_T] 特征向量矩阵 (列正交)
        V_D: [D, r_D] 特征向量矩阵 (列正交)
        lambda_T: [r_T] 特征值 (降序)
        lambda_D: [r_D] 特征值 (降序)
    """
    T = sigma_T.shape[0]
    D = sigma_D.shape[0]

    # 特征分解 Σ_T  (eigh 返回升序特征值)
    lambda_T_all, V_T_all = torch.linalg.eigh(sigma_T)
    V_T_all = V_T_all.flip(-1)
    lambda_T_all = lambda_T_all.flip(-1)

    r_T = max(1, int(T * rank_ratio_T))
    V_T = V_T_all[:, :r_T].contiguous()
    lambda_T = torch.clamp(lambda_T_all[:r_T], min=1e-10)

    # 特征分解 Σ_D
    lambda_D_all, V_D_all = torch.linalg.eigh(sigma_D)
    V_D_all = V_D_all.flip(-1)
    lambda_D_all = lambda_D_all.flip(-1)

    r_D = max(1, int(D * rank_ratio_D))
    V_D = V_D_all[:, :r_D].contiguous()
    lambda_D = torch.clamp(lambda_D_all[:r_D], min=1e-10)

    return V_T, V_D, lambda_T, lambda_D


def _flip_flop_kronecker(data_proc, n_iter_max=100, tol=1e-6):
    """
    Flip-Flop 迭代算法, 交替优化 Σ_T 和 Σ_D 以改进 Kronecker 近似.

    给定数据 X ∈ ℝ^{N×T×D}, 求解:
        min_{Σ_T, Σ_D}  || Cov(vec(X)) - Σ_T ⊗ Σ_D ||_F

    算法步骤:
        1. 初始化 V_D = I_D
        2. 固定 V_D, 更新 Σ_T = (1/ND) Σ_n Σ_d (V_D^T x_{n,:,d})(V_D^T x_{n,:,d})^T
        3. 固定 V_T (从 Σ_T 分解得到), 更新 Σ_D
        4. 重复直到收敛

    Args:
        data_proc: [N, T, D] 已预处理的数据张量
        n_iter_max: 最大迭代次数
        tol: 收敛阈值

    Returns:
        sigma_T: [T, T] 优化后的协方差矩阵
        sigma_D: [D, D] 优化后的协方差矩阵
    """
    N, T, D = data_proc.shape
    # 中心化
    data_centered = data_proc - data_proc.mean(dim=0, keepdim=True)

    # 初始估计
    # Σ_T: 将数据看作 [N*D, T]
    X_T = data_centered.permute(0, 2, 1).reshape(-1, T)
    sigma_T = (X_T.T @ X_T) / max(X_T.shape[0] - 1, 1)
    sigma_T = (sigma_T + sigma_T.T) / 2 + 1e-6 * torch.eye(T, device=sigma_T.device)

    # Σ_D: 将数据看作 [N*T, D]
    X_D = data_centered.reshape(-1, D)
    sigma_D = (X_D.T @ X_D) / max(X_D.shape[0] - 1, 1)
    sigma_D = (sigma_D + sigma_D.T) / 2 + 1e-6 * torch.eye(D, device=sigma_D.device)

    prev_obj = None
    for iteration in range(n_iter_max):
        # ---------- 固定 Σ_D, 更新 Σ_T ----------
        # 用 Σ_D^{-1/2} 白化 D 维, 然后计算 T 维协方差
        eigvals_D, eigvecs_D = torch.linalg.eigh(sigma_D)
        eigvals_D = torch.clamp(eigvals_D, min=1e-10)
        # Σ_D^{-1/2}
        sigma_D_inv_sqrt = eigvecs_D @ torch.diag(1.0 / torch.sqrt(eigvals_D)) @ eigvecs_D.T

        # 白化后数据: [N, T, D] x [D, D] -> [N, T, D], 再 reshape 为 [N*D, T]
        data_whitened_D = torch.einsum('ntd,dk->ntk', data_centered, sigma_D_inv_sqrt)
        X_T_w = data_whitened_D.permute(0, 2, 1).reshape(-1, T)
        sigma_T_new = (X_T_w.T @ X_T_w) / max(X_T_w.shape[0] - 1, 1)
        sigma_T_new = (sigma_T_new + sigma_T_new.T) / 2 + 1e-6 * torch.eye(T, device=sigma_T.device)
        # 归一化使得 trace(Σ_T) = T
        sigma_T = sigma_T_new * (T / (sigma_T_new.trace() + 1e-12))

        # ---------- 固定 Σ_T, 更新 Σ_D ----------
        eigvals_T, eigvecs_T = torch.linalg.eigh(sigma_T)
        eigvals_T = torch.clamp(eigvals_T, min=1e-10)
        sigma_T_inv_sqrt = eigvecs_T @ torch.diag(1.0 / torch.sqrt(eigvals_T)) @ eigvecs_T.T

        data_whitened_T = torch.einsum('ntd,ts->nsd', data_centered, sigma_T_inv_sqrt)
        X_D_w = data_whitened_T.reshape(-1, D)
        sigma_D_new = (X_D_w.T @ X_D_w) / max(X_D_w.shape[0] - 1, 1)
        sigma_D_new = (sigma_D_new + sigma_D_new.T) / 2 + 1e-6 * torch.eye(D, device=sigma_D.device)
        sigma_D = sigma_D_new * (D / (sigma_D_new.trace() + 1e-12))

        # ---------- 收敛检查 ----------
        # 用特征值乘积的 Frobenius 范数作为代理目标
        obj = (sigma_T.trace() * sigma_D.trace()).item()
        if prev_obj is not None and abs(obj - prev_obj) / (abs(prev_obj) + 1e-12) < tol:
            print(f"[KronPCA Flip-Flop] Converged at iteration {iteration + 1}")
            break
        prev_obj = obj

    return sigma_T, sigma_D


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


def get_pca_base(data, rank_ratio=1.0, pca_dim="all", reinit=0, speedup_sklearn=0,
                 pca_iter_max=500, pca_tol=1e-6):
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

    if pca_dim not in ("Tucker", "KronPCA"):
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
                n_iter_max=pca_iter_max, tol=pca_tol, svd_impl=svd_impl,
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

    elif pca_dim == "KronPCA":
        # ===== Kronecker PCA =====
        # 假设 Σ = Σ_T ⊗ Σ_D, 对两者分别做特征分解
        # 投影矩阵 P = V_T ⊗ V_D 保证 P^T Σ P = Λ_T ⊗ Λ_D (对角阵)
        if isinstance(rank_ratio, (int, float)):
            rank_ratios = [rank_ratio] * 2
        else:
            rank_ratios = rank_ratio
        assert len(rank_ratios) == 2, "KronPCA 需要 (rT, rD) 两个比例"

        # 步骤 1: 估计 Kronecker 协方差
        sigma_T, sigma_D, initializer_T, initializer_D, data_proc = \
            _estimate_kronecker_covariance(data, reinit)

        # 步骤 2: (可选) Flip-Flop 迭代改进 Kronecker 近似
        if pca_iter_max > 0:
            sigma_T, sigma_D = _flip_flop_kronecker(
                data_proc, n_iter_max=pca_iter_max, tol=pca_tol
            )

        # 步骤 3: 特征分解
        V_T, V_D, lambda_T, lambda_D = _kronecker_eigen_decomposition(
            sigma_T, sigma_D, rank_ratios[0], rank_ratios[1]
        )

        # base 格式: [V_T^T, V_D^T] 与 Tucker 保持一致
        # V_T: [T, r_T] -> base_T: [r_T, T]
        # V_D: [D, r_D] -> base_D: [r_D, D]
        base_T = V_T.T.contiguous()
        base_D = V_D.T.contiguous()

        # 权重: 归一化特征值, 与 Tucker 格式一致
        total_energy_T = lambda_T.sum() + 1e-12
        total_energy_D = lambda_D.sum() + 1e-12
        weights_T = lambda_T / total_energy_T
        weights_D = lambda_D / total_energy_D

        base = [_to_numpy(base_T), _to_numpy(base_D)]
        initializer = [
            [_to_numpy(x) for x in initializer_T],
            [_to_numpy(x) for x in initializer_D],
        ]
        weights = [_to_numpy(weights_T), _to_numpy(weights_D)]

        print(f"[KronPCA] base_T shape: {base[0].shape}, base_D shape: {base[1].shape}")
        print(f"[KronPCA] Explained variance ratio T (top-5): {weights[0][:5]}")
        print(f"[KronPCA] Explained variance ratio D (top-5): {weights[1][:5]}")

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

    if pca_dim == "KronPCA":
        # KronPCA 前向投影:
        # Z = X ×_T V_T^T ×_D V_D^T  =>  Z_{b,r,s} = Σ_{t,d} X_{b,t,d} * V_T_{t,r} * V_D_{d,s}
        # 由于 base 存储为 [r_T, T] 和 [r_D, D] (转置形式), 需要转回 [T, r_T] 和 [D, r_D]
        if reinit:
            (mean_T, std_T), (mean_D, std_D) = pca_cache.initializer
            data = (data - mean_T[None, :, None]) / std_T[None, :, None]
            data = (data - mean_D[None, None, :]) / std_D[None, None, :]

        component_T, component_D = pca_cache.components  # [r_T, T], [r_D, D]
        # einsum: 'btd, rt, sd -> brs'  其中 component_T[r,t] = V_T[t,r]^T
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