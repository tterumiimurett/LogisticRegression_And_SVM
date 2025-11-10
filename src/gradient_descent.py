import numpy as np
from typing import Callable, Dict, Tuple, Optional

class GradDescent:
    """
    梯度下降优化器类
    
    用于优化线性模型 z = X @ w + b 的简单梯度下降训练器。
    
    传入的损失函数必须是一个可调用对象，接受参数字典并返回一个元组：
        loss_value: float - 损失值
        dL_dparams: Dict[str, np.ndarray] - 损失函数对各参数的梯度字典
    
    属性
    ----------
    lr : float
        学习率（步长）
    max_iter : int
        最大迭代次数
    tol : float
        梯度无穷范数的收敛容差
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
        """
        执行梯度下降优化
        
        参数
        ----------
        computeLoss_and_Grad : Callable
            计算损失和梯度的函数，接受参数字典，返回 (loss_value, gradients_dict)
        init_params : dict
            初始参数字典，例如 {'weights': ..., 'bias': ...}
        callback : Optional[Callable], 可选
            每次迭代后调用的回调函数，接受 (iteration, loss, params)
            
        返回
        -------
        Dict[str, object]
            优化后的参数字典
        """
        
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
        """
        检查梯度下降是否收敛
        
        参数
        ----------
        dL_dparams : Dict[str, np.ndarray]
            各参数的梯度字典
            
        返回
        -------
        bool
            如果所有参数的梯度无穷范数都小于容差，返回 True
        """
        max_grad_norm = max(np.max(np.abs(grad)) for grad in dL_dparams.values())
        return max_grad_norm < self.tol

    @staticmethod
    def central_diff_grad(func: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        """
        使用中心差分法计算梯度
        
        按坐标逐个计算中心差分梯度：∂f/∂x_i ≈ [f(x+he_i) - f(x-he_i)] / (2h)
        
        注意：此方法使用for循环遍历所有特征；仅用于调试或无法获得解析梯度时使用。
        对于训练，建议使用能直接返回 dL/dz 的损失函数。
        
        参数
        ----------
        func : Callable
            目标函数，接受 np.ndarray 返回 float
        x : np.ndarray
            计算梯度的点
            
        返回
        -------
        np.ndarray
            梯度向量，与 x 形状相同
        """
        eps = np.finfo(float).eps
        h = eps ** (1/3)  # 中心差分的最优步长
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
        使用自适应有限差分法计算梯度
        
        利用二阶导数的估计值为每个坐标选择最优步长 h。
        
        参数
        ----------
        func : Callable
            目标函数，接受 np.ndarray 返回 float
        x : np.ndarray
            计算梯度的点
        stability_threshold : float, 默认 1e-100
            稳定性阈值，当估计的二阶导数或函数值过小时使用固定步长
            
        返回
        -------
        np.ndarray
            梯度向量，与 x 形状相同
        """
        eps = np.finfo(float).eps
        x = np.asarray(x, dtype=float)
        g = np.empty_like(x, dtype=float)

        for i in range(x.size):
            ei = np.zeros_like(x)
            ei[i] = 1.0

            # 1. 使用小探测步长估算 f'''(x)
            h_probe = eps ** (1/5)
            fx_val = func(x)
            fppp_est = (func(x + 2*h_probe*ei) - 2*func(x + h_probe*ei) + 2*func(x - h_probe*ei) - func(x - 2*h_probe*ei)) / (2 * h_probe**3)
            
            # 鲁棒性处理：f''' 估算为 0 时，退回到固定步长
            if abs(fppp_est) < stability_threshold or abs(fx_val) < 1e-100:
                h_final = np.sqrt(eps)
                h_final = eps**(1/3)
            else:
                h_final = (3 * eps * abs(fx_val) / abs(fppp_est))**(1/3)

            # 2. 计算有限差分梯度
            g[i] = (func(x + h_final * ei) - func(x - h_final * ei)) / (2 * h_final)

        return g