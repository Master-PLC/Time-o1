import torch, tensorly as tl
from tensorly.decomposition import partial_tucker


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


torch.manual_seed(0)
X = torch.randn(50, 720, 864)
ranks = [X.shape[-2], X.shape[-1]]
device = 'cuda'
X = X.to(device)
core, factors = _partial_tucker_torch(X, modes=[1,2], ranks=ranks,
                                       n_iter_max=500, tol=1e-6,
                                       svd_impl="auto")

# 重构
X_rec = core.clone()
for m, f in zip([1,2], factors):
    X_rec = _mode_product(X_rec, f, m)
err_mine = (X - X_rec).norm() / X.norm()
print("mine reconstruction err:", err_mine.item())

# tensorly 参照
tl.set_backend('pytorch')
(core_ref, factors_ref), _ = partial_tucker(X, modes=[1,2], rank=ranks,
                                            n_iter_max=500, tol=1e-6)
X_rec_ref = tl.tenalg.multi_mode_dot(core_ref, factors_ref, modes=[1,2])
print("tensorly err:", ((X - X_rec_ref).norm() / X.norm()).item())

# 因子正交性
for f in factors:
    print("orth err:", (f.T @ f - torch.eye(f.shape[1], device=device)).abs().max().item())
