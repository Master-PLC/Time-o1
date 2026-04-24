import torch


class CCA:
    def __init__(
        self,
        n_components=2,
        scale=True,
        max_iter=500,
        tol=1e-6,
        device="cpu"
    ):
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

    def _center_scale_xy(self, X, Y):
        # 仿sklearn内部实现
        x_mean = X.mean(dim=0, keepdim=True)
        x_std = X.std(dim=0, unbiased=False, keepdim=True)
        y_mean = Y.mean(dim=0, keepdim=True)
        y_std = Y.std(dim=0, unbiased=False, keepdim=True)
        Xc = X - x_mean
        Yc = Y - y_mean
        if self.scale:
            x_std = torch.where(x_std == 0, torch.ones_like(x_std), x_std)
            y_std = torch.where(y_std == 0, torch.ones_like(y_std), y_std)
            Xc = Xc / x_std
            Yc = Yc / y_std
        return Xc, Yc, x_mean, y_mean, x_std, y_std

    def _pinv(self, A):
        # 对齐sklearn/scipy pinv2
        return torch.linalg.pinv(A)

    def fit(self, X, Y):
        X = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        Y = torch.as_tensor(Y, dtype=torch.float64, device=self.device)
        n, p = X.shape
        n2, q = Y.shape
        if n != n2:
            raise ValueError("X and Y must have the same number of samples.")
        k = self.n_components
        if k > min(n, p, q):
            raise ValueError(f"n_components must be <= min(n_samples, n_features, n_targets): {min(n, p, q)}")

        Xc, Yc, self._x_mean, self._y_mean, self._x_std, self._y_std = self._center_scale_xy(X, Y)

        self.x_weights_ = torch.zeros((p, k), dtype=torch.float64, device=self.device)
        self.y_weights_ = torch.zeros((q, k), dtype=torch.float64, device=self.device)
        self.x_loadings_ = torch.zeros((p, k), dtype=torch.float64, device=self.device)
        self.y_loadings_ = torch.zeros((q, k), dtype=torch.float64, device=self.device)
        self.n_iter_ = []
        self._x_scores = torch.zeros((n, k), dtype=torch.float64, device=self.device)
        self._y_scores = torch.zeros((n, k), dtype=torch.float64, device=self.device)

        Xk = Xc.clone()
        Yk = Yc.clone()

        for comp in range(k):
            # sklearn: mask出几乎为0的y列
            eps = torch.finfo(Yk.dtype).eps
            y_mask = torch.all(torch.abs(Yk) < 10 * eps, dim=0)
            if y_mask.any():
                Yk[:, y_mask] = 0.0

            # ----------- power method (mode B, CCA) -----------
            X_pinv = self._pinv(Xk)
            Y_pinv = self._pinv(Yk)
            # 初始化 y_score
            y_score = None
            for col in Yk.T:
                if torch.any(torch.abs(col) > eps):
                    y_score = col
                    break
            if y_score is None:
                raise RuntimeError("y residual is constant")

            x_weights_old = torch.full((Xk.shape[1],), 100.0, dtype=Xk.dtype, device=self.device)
            n_iter = 0
            for n_iter in range(self.max_iter):
                x_weights = X_pinv @ y_score
                x_weights = x_weights / (torch.norm(x_weights) + eps)
                x_score = Xk @ x_weights
                y_weights = Y_pinv @ x_score
                y_weights = y_weights / (torch.norm(y_weights) + eps)
                y_score = Yk @ y_weights / (torch.dot(y_weights, y_weights) + eps)
                x_weights_diff = x_weights - x_weights_old
                if torch.dot(x_weights_diff, x_weights_diff) < self.tol or Y.shape[1] == 1:
                    break
                x_weights_old = x_weights
            self.n_iter_.append(n_iter + 1)

            # svd_flip_1d
            maxidx = torch.argmax(torch.abs(x_weights)).item()
            sign = torch.sign(x_weights[maxidx])
            x_weights = x_weights * sign
            y_weights = y_weights * sign

            x_scores = Xk @ x_weights
            y_scores = Yk @ y_weights
            x_loadings = (Xk.T @ x_scores) / (x_scores @ x_scores)
            y_loadings = (Yk.T @ y_scores) / (y_scores @ y_scores)
            Xk = Xk - x_scores.unsqueeze(1) @ x_loadings.unsqueeze(0)
            Yk = Yk - y_scores.unsqueeze(1) @ y_loadings.unsqueeze(0)

            self.x_weights_[:, comp] = x_weights
            self.y_weights_[:, comp] = y_weights
            self._x_scores[:, comp] = x_scores
            self._y_scores[:, comp] = y_scores
            self.x_loadings_[:, comp] = x_loadings
            self.y_loadings_[:, comp] = y_loadings

        # Compute rotations and coef_
        xw = self.x_weights_
        xl = self.x_loadings_
        yw = self.y_weights_
        yl = self.y_loadings_
        self.x_rotations_ = xw @ self._pinv(xl.T @ xw)
        self.y_rotations_ = yw @ self._pinv(yl.T @ yw)
        coef_ = self.x_rotations_ @ yl.T
        coef_ = (coef_ * self._y_std.T) / self._x_std.T
        self.coef_ = coef_.T
        self.intercept_ = self._y_mean.squeeze()
        return self

    def transform(self, X, Y=None):
        X = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        Xc = (X - self._x_mean) / self._x_std
        x_scores = Xc @ self.x_rotations_
        if Y is not None:
            Y = torch.as_tensor(Y, dtype=torch.float64, device=self.device)
            Yc = (Y - self._y_mean) / self._y_std
            y_scores = Yc @ self.y_rotations_
            return x_scores.cpu().numpy(), y_scores.cpu().numpy()
        return x_scores.cpu().numpy()

    def inverse_transform(self, X, Y=None):
        X = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        X_rec = X @ self.x_loadings_.T
        X_rec = X_rec * self._x_std + self._x_mean
        if Y is not None:
            Y = torch.as_tensor(Y, dtype=torch.float64, device=self.device)
            Y_rec = Y @ self.y_loadings_.T
            Y_rec = Y_rec * self._y_std + self._y_mean
            return X_rec.cpu().numpy(), Y_rec.cpu().numpy()
        return X_rec.cpu().numpy()

    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        Xc = X - self._x_mean
        Y_pred = Xc @ self.coef_.T + self.intercept_
        return Y_pred.cpu().numpy()