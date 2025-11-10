import numpy as np
import time
import argparse
from pathlib import Path

import src.utils as utils
from src.logistic_regression import LogisticRegression


def run_experiment(X, y, method='analytic', title_prefix=""):
    """
    运行单个实验：训练模型、评估、可视化
    
    Args:
        X: 特征数据
        y: 标签
        method: 训练方法
        title_prefix: 图表标题前缀
    
    Returns:
        dict: 包含训练时间和准确率的结果字典
    """
    print(f"\n{'='*60}")
    print(f"{title_prefix}")
    print(f"{'='*60}")
    
    # 训练模型
    model = LogisticRegression()
    start_time = time.time()
    model.fit(X, y, method=method)
    train_time = time.time() - start_time
    
    # 评估模型
    y_pred = model.predict(X)
    accuracy = utils.accuracy_score(y, y_pred)
    
    # 打印结果
    print(f"Training time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    
    # 可视化决策边界
    if X.shape[1] == 2:  # 只对2D数据可视化
        f1xx, f2yy = np.meshgrid(
            np.linspace(X[:,0].min()-1, X[:,0].max()+1, 500),
            np.linspace(X[:,1].min()-1, X[:,1].max()+1, 500)
        )
        f1f2grid = np.c_[f1xx.ravel(), f2yy.ravel()]
        f1f2pred = model.predict(f1f2grid).reshape(f1xx.shape)
        
        utils.prediction_visualization_2d(
            f1xx, f2yy, f1f2pred, X, y,
            title=f"{title_prefix} - Decision Boundary"
        )
    
    # 绘制损失曲线
    loss_curve = model.get_loss_curve()
    if loss_curve:
        utils.plot_loss_curve(loss_curve, title=f"{title_prefix} - Loss Curve")
    
    return {'train_time': train_time, 'accuracy': accuracy}


def main(n_samples=200):
    """
    主函数：运行所有合成数据实验
    
    Args:
        n_samples: 样本数量
    """
    results = {}
    
    # 实验1: 线性可分数据
    print("\n" + "🔬 Experiment 1: Linearly Separable Data")
    X, y = utils.generate_synthetic_data(n_samples=n_samples, noise=0.4)
    results['linear'] = run_experiment(
        X, y, 
        method='analytic',
        title_prefix="Linear Separable Data"
    )
    
    # 实验2: 圆形数据
    print("\n" + "🔬 Experiment 2: Circular Data")
    X, y = utils.generate_circular_data(n_samples=n_samples, noise=0.1)
    results['circular'] = run_experiment(
        X, y,
        method='analytic',
        title_prefix="Circular Data"
    )
    
    # 打印总结
    print("\n" + "="*60)
    print("📊 Experiment Summary")
    print("="*60)
    for exp_name, result in results.items():
        print(f"{exp_name.capitalize():15s} | "
              f"Time: {result['train_time']:6.4f}s | "
              f"Accuracy: {result['accuracy']:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run synthetic data experiments')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of samples to generate (default: 200)')
    args = parser.parse_args()
    
    main(n_samples=args.n_samples)