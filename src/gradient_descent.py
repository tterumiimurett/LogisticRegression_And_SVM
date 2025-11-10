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
    learning_rate : float
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

    def __init__(self, learning_rate: float, max_iter: int, tol: float, plot_progress: bool = False):
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.converged: bool = False
        self.iter = 0
        self.plot_progress = plot_progress
        self.loss_curve = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]],
        init_w: Optional[np.ndarray] = None,
        init_b: Optional[float] = None,
        average_gradients: bool = True,
    ) -> Dict[str, object]:
        """
        Fit the linear model by gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Design matrix.
        y : np.ndarray, shape (n_samples,)
            Targets.
        loss : callable
            Function (z, y) -> (loss_value, dL_dz) where z = X @ w + b.
        init_w : optional np.ndarray
            Initial weights; defaults to zeros.
        init_b : optional float
            Initial bias; defaults to 0.0.
        average_gradients : bool
            If True, use mean gradients; otherwise use sum.

        Returns
        -------
        Dict with keys:
            status : str
            iter : int
            loss : float
            weights : np.ndarray
            bias : float
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        w = np.zeros(n_features) if init_w is None else np.asarray(init_w, dtype=float).copy()
        b = 0.0 if init_b is None else float(init_b)

        lr = self.learning_rate

        for i in range(1, self.max_iter + 1):
            z = X @ w + b                          # shape (n_samples,)
            L, dL_dz = loss(z, y)                  # L: float, dL_dz: (n_samples,)
            if self.plot_progress:
                self.loss_curve.append(L)
            if dL_dz.shape != (n_samples,):
                raise ValueError("loss must return dL_dz with shape (n_samples,)")

            if average_gradients:
                scale = 1.0 / n_samples
            else:
                scale = 1.0

            # Vectorized gradients for linear model
            dw = (X.T @ dL_dz) * scale             # shape (n_features,)
            db = dL_dz.sum() * scale               # scalar

            # Convergence check (∞-norm of parameter gradients)
            grad_inf = max(np.linalg.norm(dw, ord=np.inf), abs(db))
            if grad_inf < self.tol:
                self.weights, self.bias = w, b
                self.converged = True
                self.iter = i
                return {"status": "converged", "iter": i, "loss": float(L), "weights": w, "bias": b}

            # Parameter update
            w -= lr * dw
            b -= lr * db

        # Save final params
        self.weights, self.bias = w, b
        self.iter = self.max_iter
        return {"status": "max_iter_reached", "iter": self.max_iter, "loss": float(L), "weights": w, "bias": b}

    # ---------- Finite-difference utilities (optional) ----------

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

# ---------------------- Example loss functions ----------------------

def mse_loss(z: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Mean squared error:
        L = 0.5 * mean((z - y)^2)
        dL/dz = (z - y) / n
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = z - y
    dL_dz = diff / z.size
    L = 0.5 * np.dot(diff, diff) / z.size
    return float(L), dL_dz