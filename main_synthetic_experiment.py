import numpy as np
import time
import argparse
import json
from datetime import datetime
from pathlib import Path

import src.utils as utils
from src.logistic_regression import LogisticRegression


def run_experiment(X, y, method='analytic', title_prefix="", save_dir=None):
    """
    运行单个实验：训练模型、评估、可视化
    
    Args:
        X: 特征数据
        y: 标签
        method: 训练方法
        title_prefix: 图表标题前缀
        save_dir: 保存结果的目录路径
    
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
    
    # 生成安全的文件名
    safe_filename = title_prefix.replace(" ", "_").replace(":", "")
    
    # 可视化决策边界
    if X.shape[1] == 2:  # 只对2D数据可视化
        f1xx, f2yy = np.meshgrid(
            np.linspace(X[:,0].min()-1, X[:,0].max()+1, 500),
            np.linspace(X[:,1].min()-1, X[:,1].max()+1, 500)
        )
        f1f2grid = np.c_[f1xx.ravel(), f2yy.ravel()]
        f1f2pred = model.predict(f1f2grid).reshape(f1xx.shape)
        
        if save_dir:
            boundary_path = save_dir / f"{safe_filename}_decision_boundary.png"
            utils.prediction_visualization_2d(
                f1xx, f2yy, f1f2pred, X, y,
                title=f"{title_prefix} - Decision Boundary",
                save_path=str(boundary_path)
            )
        else:
            utils.prediction_visualization_2d(
                f1xx, f2yy, f1f2pred, X, y,
                title=f"{title_prefix} - Decision Boundary"
            )
    
    # 绘制损失曲线
    loss_curve = model.get_loss_curve()
    if loss_curve:
        if save_dir:
            loss_path = save_dir / f"{safe_filename}_loss_curve.png"
            utils.plot_loss_curve(loss_curve, 
                                 title=f"{title_prefix} - Loss Curve",
                                 save_path=str(loss_path))
        else:
            utils.plot_loss_curve(loss_curve, title=f"{title_prefix} - Loss Curve")
    
    return {
        'train_time': train_time, 
        'accuracy': accuracy,
        'loss_curve': loss_curve,
        'n_iterations': len(loss_curve) if loss_curve else 0,
        'final_loss': loss_curve[-1] if loss_curve else None,
        'weights': model.weights.tolist() if model.weights is not None else None,
        'bias': float(model.bias) if model.bias is not None else None
    }


def main(n_samples=200, save_results=True):
    """
    主函数：运行所有合成数据实验
    
    Args:
        n_samples: 样本数量
        save_results: 是否保存结果到文件
    """
    # 创建结果保存目录
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results") / f"synthetic_experiment_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Results will be saved to: {results_dir}")
    else:
        results_dir = None
    
    results = {}
    experiment_metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': n_samples,
        'experiments': {}
    }
    
    # 实验1: 线性可分数据
    print("\n" + "🔬 Experiment 1: Linearly Separable Data")
    X_linear, y_linear = utils.generate_synthetic_data(n_samples=n_samples, noise=0.4)
    
    # 保存数据
    if save_results:
        np.savez(results_dir / "linear_data.npz", X=X_linear, y=y_linear)
    
    results['linear_analytic'] = run_experiment(
        X_linear, y_linear, 
        method='analytic',
        title_prefix="Linear Separable Data: Analytic",
        save_dir=results_dir
    )

    results['linear_numerical_fixed'] = run_experiment(
        X_linear, y_linear,
        method='numerical_fixed',
        title_prefix="Linear Separable Data: Numerical Fixed",
        save_dir=results_dir
    )

    results['linear_numerical_adaptive'] = run_experiment(
        X_linear, y_linear,
        method='numerical_adaptive',
        title_prefix="Linear Separable Data: Numerical Adaptive",
        save_dir=results_dir
    )

    # 实验2: 圆形数据
    print("\n" + "🔬 Experiment 2: Circular Data")
    X_circular, y_circular = utils.generate_circular_data(n_samples=n_samples, noise=0.1)
    
    # 保存数据
    if save_results:
        np.savez(results_dir / "circular_data.npz", X=X_circular, y=y_circular)
    
    results['circular_analytic'] = run_experiment(
        X_circular, y_circular,
        method='analytic',
        title_prefix="Circular Data: Analytic",
        save_dir=results_dir
    )
    
    # 打印总结
    print("\n" + "="*60)
    print("📊 Experiment Summary")
    print("="*60)
    for exp_name, result in results.items():
        print(f"{exp_name:30s} | "
              f"Time: {result['train_time']:8.4f}s | "
              f"Accuracy: {result['accuracy']:.4f} | "
              f"Iterations: {result['n_iterations']:5d}")
        
        # 添加到元数据
        experiment_metadata['experiments'][exp_name] = {
            'train_time': result['train_time'],
            'accuracy': result['accuracy'],
            'n_iterations': result['n_iterations'],
            'final_loss': result['final_loss'],
            'weights': result['weights'],
            'bias': result['bias']
        }
    print("="*60)
    
    # 保存实验结果到JSON
    if save_results:
        json_path = results_dir / "experiment_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_metadata, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to: {results_dir}")
        print(f"   - Figures: *.png")
        print(f"   - Data: *.npz")
        print(f"   - Summary: experiment_results.json")
    
    return results, results_dir if save_results else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run synthetic data experiments')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of samples to generate (default: 200)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to disk')
    args = parser.parse_args()
    
    main(n_samples=args.n_samples, save_results=not args.no_save)