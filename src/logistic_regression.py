import numpy as np
from src.gradient_descent import GradDescent
from typing import Tuple

# logistic regression:
# input X is a (n,d) matrix, n samples, d features
# label Y is a (n,) vector, binary labels {0,1}
# Model: logistic regression model:  p(y=1|x) = sigmoid(w.x + b)
# Loss function: Binary Cross Entropy Loss: \sum -y log (pi) - (1-y) log(1-pi)
# Update rule: gradient descent

class LogisticRegression: 
    def __init__(self, lr=0.1, max_iter=10000, tol=1e-4):
        self.optimizer = GradDescent(lr=lr, max_iter=max_iter, tol=tol)
        self.weights = None
        self.bias = None
        self._training_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, method: str = 'analytic') -> 'LogisticRegression':
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
        z = X @ self.weights + self.bias
        probs = self.sigmoid(z)
        return (probs >= 0.5).astype(int)

    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        probs = self.sigmoid(z)
        return probs
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def get_loss_curve(self) -> np.ndarray:
        return [np_record['loss'] for np_record in self._training_history]


    def logistic_loss(self, params: dict) -> Tuple[float, dict[str, np.ndarray]]:
        
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

        return float(loss), dL_dz

    def logistic_pure_loss(self, z: np.ndarray) -> float:
        y = self._y
        pos = y > 0
        neg = ~pos
        probs = self.sigmoid(z)

        n_samples = z.shape[0]

        loss = -( (pos * np.log(probs)).sum() + (neg * np.log(1 - probs)).sum() ) / n_samples
        return float(loss)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        prob = 1 / (1 + np.exp(-z))
        eps = 1e-15
        prob = np.clip(prob, eps, 1 - eps)
        return prob

