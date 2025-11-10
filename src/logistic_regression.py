import numpy as np
from src.gradient_descent import GradDescent
from typing import Tuple

# 逻辑回归：
# 输入 X 是一个 (n,d) 矩阵，n 个样本，d 个特征
# 标签 Y 是一个 (n,) 向量，二分类标签 {0,1}
# 模型：逻辑回归模型：p(y=1|x) = sigmoid(w.x + b)
# 损失函数：二元交叉熵损失：\sum -y log (pi) - (1-y) log(1-pi)
# 更新规则：梯度下降

class LogisticRegression:
    """
    逻辑回归分类器
    
    使用梯度下降优化二元交叉熵损失函数来训练逻辑回归模型。
    支持三种梯度计算方法：解析梯度、固定步长数值梯度和自适应数值梯度。
    
    参数
    ----------
    lr : float, 默认 0.1
        学习率
    max_iter : int, 默认 10000
        最大迭代次数
    tol : float, 默认 1e-4
        收敛容差
        
    属性
    ----------
    weights : np.ndarray
        训练后的权重向量
    bias : float
        训练后的偏置项
    """
    def __init__(self, lr=0.1, max_iter=10000, tol=1e-4):
        self.optimizer = GradDescent(lr=lr, max_iter=max_iter, tol=tol)
        self.weights = None
        self.bias = None
        self._training_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, method: str = 'analytic') -> 'LogisticRegression':
        """
        训练逻辑回归模型
        
        参数
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            训练数据特征矩阵
        y : np.ndarray, shape (n_samples,)
            训练数据标签向量，取值为 {0, 1}
        method : str, 默认 'analytic'
            梯度计算方法：
            - 'analytic': 使用解析梯度（最快、最准确）
            - 'numerical_fixed': 使用固定步长的数值梯度
            - 'numerical_adaptive': 使用自适应步长的数值梯度
            
        返回
        -------
        self : LogisticRegression
            训练后的模型实例
        """
        n_samples, n_features = X.shape
        self._training_history = [] # clear history

        self._X = np.asarray(X, float)
        self._y = np.asarray(y, float)

        init_params = {
            'weights': np.zeros(n_features),
            'bias': np.float64(0.0),
            }
        
        def record_history(iteration, loss, params):
            self._training_history.append({
                'iteration' : iteration, 
                'weights': params['weights'].copy(),
                'bias': params['bias'],
                'loss': loss, 
            })
        
        if method == 'analytic':
            loss_function = self.logistic_loss
        elif method == 'numerical_fixed':
            loss_function = self.logistic_numerical_fixed_derivative_loss
        elif method == 'numerical_adaptive':
            loss_function = self.logistic_numerical_adaptive_derivative_loss
        else:
            raise ValueError(f"Unknown method: {method}")
        
        optimized_params = self.optimizer.optimize(
            computeLoss_and_Grad=loss_function,
            init_params=init_params,
            callback=record_history
        )

        self.weights = optimized_params['weights']
        self.bias = optimized_params['bias']

        del self._X, self._y

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        对新数据进行预测
        
        参数
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            待预测的特征矩阵
            
        返回
        -------
        np.ndarray, shape (n_samples,)
            预测的类别标签 {0, 1}
        """
        z = X @ self.weights + self.bias
        probs = self.sigmoid(z)
        return (probs >= 0.5).astype(int)

    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本属于类别 1 的概率
        
        参数
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            待预测的特征矩阵
            
        返回
        -------
        np.ndarray, shape (n_samples,)
            每个样本属于类别 1 的概率值，范围 [0, 1]
        """
        z = X @ self.weights + self.bias
        probs = self.sigmoid(z)
        return probs
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        评估模型在给定数据集上的准确率
        
        参数
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            测试数据特征矩阵
        y : np.ndarray, shape (n_samples,)
            测试数据真实标签
            
        返回
        -------
        float
            分类准确率，范围 [0, 1]
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def get_loss_curve(self) -> np.ndarray:
        """
        获取训练过程中的损失值曲线
        
        返回
        -------
        list
            每次迭代的损失值列表
        """
        return [np_record['loss'] for np_record in self._training_history]


    def logistic_loss(self, params: dict) -> Tuple[float, dict[str, np.ndarray]]:
        """
        计算逻辑回归的损失函数和梯度（解析方法）
        
        使用二元交叉熵损失：L = -1/n * Σ[y*log(p) + (1-y)*log(1-p)]
        
        参数
        ----------
        params : dict
            参数字典，包含 'weights' 和 'bias'
            
        返回
        -------
        Tuple[float, dict]
            (损失值, 梯度字典)
        """
        
        X = self._X
        weights = params['weights']
        bias = params['bias']

        z = X @ weights + bias
        y = self._y

        n_samples = X.shape[0]

        pos = y > 0
        neg = ~pos
        probs = self.sigmoid(z)
        loss = -( (pos * np.log(probs)).sum() + (neg * np.log(1 - probs)).sum() ) / n_samples
        dL_dz = (probs - y) / n_samples

        dL_dw = X.T @ dL_dz
        dL_db = dL_dz.sum()

        grads_param = {
            'weights': dL_dw,
            'bias': dL_db,
        }

        return float(loss), grads_param

    def logistic_numerical_fixed_derivative_loss(self, params: dict) -> Tuple[float, dict[str, np.ndarray]]:
        """
        计算逻辑回归的损失函数和梯度（固定步长数值方法）
        
        使用中心差分法计算 dL/dz，然后通过链式法则得到 dL/dw 和 dL/db。
        
        参数
        ----------
        params : dict
            参数字典，包含 'weights' 和 'bias'
            
        返回
        -------
        Tuple[float, dict]
            (损失值, 梯度字典)
        """

        X = self._X
        weights = params['weights']
        bias = params['bias']

        z = X @ weights + bias
        y = self._y

        n_samples = X.shape[0]

        pos = y > 0
        neg = ~pos
        probs = self.sigmoid(z)
        loss = -( (pos * np.log(probs)).sum() + (neg * np.log(1 - probs)).sum() ) / n_samples
        dL_dz = GradDescent.central_diff_grad(self.logistic_pure_loss, z) / n_samples

        dL_dw = X.T @ dL_dz
        dL_db = dL_dz.sum()

        grads_param = {
            'weights': dL_dw,
            'bias': dL_db,
        }

        return float(loss), grads_param
    
    def logistic_numerical_adaptive_derivative_loss(self, params: dict ) -> Tuple[float, dict[str, np.ndarray]]:
        """
        计算逻辑回归的损失函数和梯度（自适应步长数值方法）
        
        使用自适应步长的有限差分法计算 dL/dz，然后通过链式法则得到参数梯度。
        
        参数
        ----------
        params : dict
            参数字典，包含 'weights' 和 'bias'
            
        返回
        -------
        Tuple[float, dict]
            (损失值, 梯度字典)
        """
        
        X = self._X
        weights = params['weights']
        bias = params['bias']

        z = X @ weights + bias
        y = self._y

        n_samples = X.shape[0]

        pos = y > 0
        neg = ~pos
        probs = self.sigmoid(z)
        loss = -( (pos * np.log(probs)).sum() + (neg * np.log(1 - probs)).sum() ) / n_samples
        dL_dz = GradDescent.adaptive_grad(self.logistic_pure_loss, z) / n_samples

        dL_dw = X.T @ dL_dz
        dL_db = dL_dz.sum()

        grads_param = {
            'weights': dL_dw,
            'bias': dL_db,
        }

        return float(loss), grads_param

    def logistic_pure_loss(self, z: np.ndarray) -> float:
        """
        计算纯损失值（不含梯度）
        
        用于数值梯度计算中的函数值评估。
        
        参数
        ----------
        z : np.ndarray, shape (n_samples,)
            线性预测值 z = X @ w + b
            
        返回
        -------
        float
            二元交叉熵损失值
        """
        y = self._y
        pos = y > 0
        neg = ~pos
        probs = self.sigmoid(z)

        n_samples = z.shape[0]

        loss = -( (pos * np.log(probs)).sum() + (neg * np.log(1 - probs)).sum() ) / n_samples
        return float(loss)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid 激活函数
        
        计算 sigmoid(z) = 1 / (1 + exp(-z))，并进行数值稳定性处理。
        
        参数
        ----------
        z : np.ndarray
            输入值
            
        返回
        -------
        np.ndarray
            sigmoid 函数的输出，范围 [eps, 1-eps]，其中 eps=1e-15
        """
        prob = 1 / (1 + np.exp(-z))
        eps = 1e-15
        prob = np.clip(prob, eps, 1 - eps)
        return prob

