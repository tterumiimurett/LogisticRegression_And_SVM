import argparse
import time
from pathlib import Path

import src.data_loading as data_loading
from src.logistic_regression import LogisticRegression
import src.utils as utils


def train_and_evaluate(X_train, y_train, X_test, y_test, method='analytic'):
    """
    训练并评估模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        method: 训练方法
    
    Returns:
        dict: 包含训练时间和准确率的结果
    """
    print(f"\n{'='*60}")
    print(f"Training Logistic Regression (method={method})")
    print(f"{'='*60}")
    
    # 训练模型
    model = LogisticRegression()
    start_time = time.time()
    model.fit(X_train, y_train, method=method)
    train_time = time.time() - start_time
    
    print(f"Training completed in {train_time:.4f} seconds")
    
    # 评估模型
    train_accuracy = model.evaluate(X_train, y_train)
    test_accuracy = model.evaluate(X_test, y_test)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # 绘制损失曲线
    loss_curve = model.get_loss_curve()
    if loss_curve:
        utils.plot_loss_curve(loss_curve, title="Gisette Dataset - Loss Curve")
    
    return {
        'train_time': train_time,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }


def main(method='analytic'):
    """
    主函数：加载数据并运行实验
    
    Args:
        method: 训练方法 ('analytic' 或 'gradient_descent')
    """
    print("\n🔬 Loading Gisette Dataset...")
    X_train, y_train, X_test, y_test = data_loading.load_gisette_local()
    
    print(f"Dataset loaded:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # 运行实验
    results = train_and_evaluate(X_train, y_train, X_test, y_test, method=method)
    
    # 打印总结
    print("\n" + "="*60)
    print("📊 Experiment Summary")
    print("="*60)
    print(f"Training time:     {results['train_time']:6.4f}s")
    print(f"Training accuracy: {results['train_accuracy']:.4f}")
    print(f"Test accuracy:     {results['test_accuracy']:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Gisette dataset experiment')
    parser.add_argument('--method', type=str, default='analytic',
                        choices=['analytic', 'gradient_descent'],
                        help='Training method (default: analytic)')
    args = parser.parse_args()
    
    main(method=args.method)
