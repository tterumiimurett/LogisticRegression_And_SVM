import numpy as np
from typing import Callable, Dict, Tuple, Optional

class GradDescent:
    """
    Simple gradient-descent trainer for a *linear* model z = X @ w + b.

    The `loss` you pass must be a callable taking (z, y) and returning a tuple:
        loss_value: float
        dL_dz: np.ndarray shape (n_samples,) — derivative of the loss w.r.t. z

    Examples of valid `loss` functions are provided at the bottom (mse_loss, logistic_loss).

    Attributes
    ----------
    lr : float
        Step size.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on gradient infinity-norm.
    weights : Optional[np.ndarray]
        Learned weight vector of shape (n_features,).
    bias : Optional[float]
        Learned bias scalar.
    """

    def __init__(self, lr: float = 0.1, max_iter: int = 10000, tol: float = 1e-5):
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def optimize(
        self,
        computeLoss_and_Grad: Callable[[np.ndarray], Tuple[float, np.ndarray]],
        init_params: np.ndarray,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None
    ) -> Dict[str, object]:
        
        params = {k: v.copy() for k, v in init_params.items()}

        for i in range(1, self.max_iter + 1):
            L, dL_dparams = computeLoss_and_Grad(params)

            if callback is not None:
                callback(i, L, params)

            for key in params:
                params[key] -= self.lr * dL_dparams[key]

            if self._check_convergence(dL_dparams):
                break

        return params

    def _check_convergence(self, dL_dparams: Dict[str, np.ndarray]) -> bool:
        max_grad_norm = max(np.max(np.abs(grad)) for grad in dL_dparams.values())
        return max_grad_norm < self.tol

    @staticmethod
    def central_diff_grad(func: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        """
        Coordinate-wise central difference gradient ∂f/∂x_i ≈ [f(x+he_i) - f(x-he_i)] / (2h).

        NOTE: This uses a for-loop over features; it's only for debugging or when an analytic gradient
        is unavailable. For training, prefer providing a loss that returns dL/dz.
        """
        eps = np.finfo(float).eps
        h = eps ** (1/3)  # Optimal step size for central difference
        x = np.asarray(x, dtype=float)
        g = np.empty_like(x, dtype=float)
        for i in range(x.size):
            ei = np.zeros_like(x)
            ei[i] = 1.0
            g[i] = (func(x + h * ei) - func(x - h * ei)) / (2.0 * h)
        return g
    
    @staticmethod
    def adaptive_grad(func: Callable[[np.ndarray], float], x: np.ndarray, stability_threshold: float = 1e-100) -> np.ndarray:
        """
        Coordinate-wise adaptive finite-difference gradient.

        Uses an estimate of the second derivative to choose an optimal step size h for each coordinate.
        """
        eps = np.finfo(float).eps
        x = np.asarray(x, dtype=float)
        g = np.empty_like(x, dtype=float)

        for i in range(x.size):
            ei = np.zeros_like(x)
            ei[i] = 1.0

            # 1. 估算 f''(x) using a small probe step
            h_probe = eps ** (1/5)
            fx_val = func(x)
            fppp_est = (func(x + 2*h_probe*ei) - 2*func(x + h_probe*ei) + 2*func(x - h_probe*ei) - func(x - 2*h_probe*ei)) / (2 * h_probe**3)
            
            # 健壮性: f'' 估算为 0, 退回到 v2 (简单前向)
            if abs(fppp_est) < stability_threshold or abs(fx_val) < 1e-100:
                h_final = np.sqrt(eps)
                h_final = eps**(1/3)
            else:
                h_final = (3 * eps * abs(fx_val) / abs(fppp_est))**(1/3)

            # 2. 计算有限差分梯度
            g[i] = (func(x + h_final * ei) - func(x - h_final * ei)) / (2 * h_final)

        return g