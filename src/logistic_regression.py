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
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.data = X
        self.labels = y
        self.gradient = GradDescent(learning_rate=1e-1, max_iter=10000, tol=1e-4, plot_progress=True)
        self.gradient.fit(X, y, loss=self.logistic_loss)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.gradient.weights + self.gradient.bias
        probs = self.sigmoid(z)
        return (probs >= 0.5).astype(int)

    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.gradient.weights + self.gradient.bias
        probs = self.sigmoid(z)
        return probs
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def get_loss_curve(self) -> np.ndarray:
        return np.array(self.gradient.loss_curve)


    def logistic_loss(self, z: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:

        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)

        pos = y > 0
        neg = ~pos
        loss = -(pos * self.sigmoid(z)).sum() - (neg * self.sigmoid(-z)).mean()
        dL_dz = self.sigmoid(z) - y

        return float(loss), dL_dz
    
    def logistic_numerical_fixed_derivative_loss(self, z: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:

        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)

        pos = y > 0
        neg = ~pos
        loss = (pos * self.sigmoid(z)).sum() + (neg * self.sigmoid(-z)).mean()
        dL_dz = GradDescent.central_diff_grad(self.logistic_pure_loss, z)

        return float(loss), dL_dz
    
    def logistic_numerical_adaptive_derivative_loss(self, z: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)

        pos = y > 0
        neg = ~pos
        loss = (pos * self.sigmoid(z)).sum() + (neg * self.sigmoid(-z)).mean()
        dL_dz = GradDescent.adaptive_grad(self.logistic_pure_loss, z)

        return float(loss), dL_dz
    
    def logistic_pure_loss(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        y = np.asarray(self.labels, dtype=float)
        pos = y > 0
        neg = ~pos
        loss = (pos * self.sigmoid(z)).sum() + (neg * self.sigmoid(-z)).sum()
        return float(loss)


    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

