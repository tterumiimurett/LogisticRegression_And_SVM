import os
print("CWD =", os.getcwd())
import numpy as np
import src.utils as utils
from src.logistic_regression import LogisticRegression
import time


if __name__ == "__main__":
    
    # 生成合成数据
    X, y = utils.generate_synthetic_data(n_samples=200, noise=0.4)
    logisticReg = LogisticRegression()

    start_time = time.time()
    logisticReg.fit(X, y, method='analytic')
    end_time = time.time()
    print(f"Training time for Logistic Regression on synthetic data: {end_time - start_time:.4f} seconds")

    f1xx, f2yy = np.meshgrid(np.linspace(min(X[:,0])-1, max(X[:,0])+1, 500),
                             np.linspace(min(X[:,1])-1, max(X[:,1])+1, 500))
    f1f2grid = np.c_[f1xx.ravel(), f2yy.ravel()]

    f1f2pred = logisticReg.predict(f1f2grid).reshape(f1xx.shape)

    utils.prediction_visualization_2d(f1xx, f2yy, f1f2pred, X, y, title="Logistic Regression Prediction")

    accuracy = utils.accuracy_score(y, logisticReg.predict(X))
    print(f"Accuracy for Logistic Regression on synthetic data: {accuracy:.4f}")

    loss_curve = logisticReg.get_loss_curve()
    utils.plot_loss_curve(loss_curve, title="Logistic Regression Loss Curve")
    # 生成圆形数据
    X, y = utils.generate_circular_data(n_samples=200, noise=0.1)
    
    start_time = time.time()
    logisticReg.fit(X, y)
    end_time = time.time()
    print(f"Training time for Logistic Regression on circular data: {end_time - start_time:.4f} seconds")

    f1xx, f2yy = np.meshgrid(np.linspace(min(X[:,0])-1, max(X[:,0])+1, 500),
                             np.linspace(min(X[:,1])-1, max(X[:,1])+1, 500))
    f1f2grid = np.c_[f1xx.ravel(), f2yy.ravel()]

    f1f2pred = logisticReg.predict(f1f2grid).reshape(f1xx.shape)

    utils.prediction_visualization_2d(f1xx, f2yy, f1f2pred, X, y, title="Logistic Regression Prediction")

    accuracy = utils.accuracy_score(y, logisticReg.predict(X))
    print(f"Accuracy for Logistic Regression on circular data: {accuracy:.4f}")

    loss_curve = logisticReg.get_loss_curve()
    utils.plot_loss_curve(loss_curve, title="Logistic Regression Loss Curve")