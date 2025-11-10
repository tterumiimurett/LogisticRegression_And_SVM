import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(n_samples=200, n_features=2, noise=0.3, mu0 = np.array([-1,0], dtype=float), mu1 = np.array([1,0], dtype=float)):
    """
    生成线性可分的二分类合成数据
    
    从两个不同均值的高斯分布中采样数据点，形成两个类别。
    
    参数
    ----------
    n_samples : int, 默认 200
        总样本数
    n_features : int, 默认 2
        特征维度
    noise : float, 默认 0.3
        高斯噪声的标准差
    mu0 : np.ndarray, 默认 [-1, 0]
        类别 0 的均值向量
    mu1 : np.ndarray, 默认 [1, 0]
        类别 1 的均值向量
        
    返回
    -------
    Tuple[np.ndarray, np.ndarray]
        (data, labels) - 特征矩阵和标签向量
    """
    n_samples_H0 = n_samples // 2
    n_samples_H1 = n_samples // 2 + 1 if n_samples % 2 else n_samples // 2
    data_H0 = np.random.normal(loc = mu0, scale = noise, size = (n_samples_H0, n_features))
    data_H1 = np.random.normal(loc = mu1, scale = noise, size = (n_samples_H1, n_features))
    data = np.r_[data_H0, data_H1]
    labels = np.array([0]*n_samples_H0 + [1]*n_samples_H1)
    # Shuffle data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    return data[indices], labels[indices]

def generate_circular_data(n_samples=200, radius=1.0, noise=0.1, center=np.array([0, 0], dtype=float)):
    """
    生成圆形分布的合成数据
    
    生成以指定半径为边界的二分类数据，半径内为类别 1，半径外为类别 0。
    可添加噪声使边界更真实。
    
    参数
    ----------
    n_samples : int, 默认 200
        总样本数
    radius : float, 默认 1.0
        划分边界的半径
    noise : float, 默认 0.1
        给样本点添加的噪声强度
    center : np.ndarray, 默认 [0, 0]
        圆心坐标
    
    返回
    -------
    Tuple[np.ndarray, np.ndarray]
        (data, labels) - 特征矩阵 (n_samples, 2) 和类别标签 (0或1)
    """
    # 生成极坐标角度 (0到2π)
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    
    # 生成半径（故意让内外分布更均匀）
    radii = np.random.uniform(0, radius * 2, n_samples)
    
    # 转换为直角坐标并添加噪声
    x = center[0] + radii * np.cos(angles) + np.random.normal(0, noise, n_samples)
    y = center[1] + radii * np.sin(angles) + np.random.normal(0, noise, n_samples)
    
    data = np.column_stack((x, y))
    
    # 计算到圆心的实际距离（含噪声）
    distances = np.sqrt((data[:, 0] - center[0])**2 + (data[:, 1] - center[1])** 2)
    
    # 距离 <= 半径的为类别1，否则为类别0
    labels = (distances <= radius).astype(int)
    
    return data, labels


def data_visualization(X: np.ndarray, y: np.ndarray):
    """
    可视化二分类数据
    
    使用散点图展示两个类别的数据分布。
    
    参数
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        特征矩阵，至少包含 2 个特征用于可视化
    y : np.ndarray, shape (n_samples,)
        类别标签 {0, 1}
    """
    plt.figure()
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], c='blue', label='Class 0')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='red', label='Class 1')
    plt.title("Synthetic Data Visualization")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def prediction_visualization_2d(f1xx: np.ndarray, f2yy: np.ndarray, f1f2pred: np.ndarray, X: np.ndarray, y: np.ndarray, title: str):
    """
    可视化二维特征空间的预测结果
    
    在二维网格上绘制模型的预测边界，并叠加真实数据点。
    
    参数
    ----------
    f1xx : np.ndarray
        特征 1 的网格矩阵（meshgrid 的第一个输出）
    f2yy : np.ndarray
        特征 2 的网格矩阵（meshgrid 的第二个输出）
    f1f2pred : np.ndarray
        网格点上的预测值矩阵
    X : np.ndarray, shape (n_samples, 2)
        真实数据的特征矩阵
    y : np.ndarray, shape (n_samples,)
        真实数据的标签
    title : str
        图表标题
    """
    plt.figure()
    plt.contourf(f1xx, f2yy, f1f2pred, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], c='blue', label='True Class 0', marker='o')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='red', label='True Class 1', marker='o')
    plt.title("Prediction Visualization")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算分类准确率
    
    准确率定义为正确预测的样本数占总样本数的比例。

    参数
    ----------
    y_true : np.ndarray
        真实标签数组
    y_pred : np.ndarray
        预测标签数组

    返回
    -------
    float
        准确率，范围 [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = (y_true == y_pred).sum()
    total = y_true.shape[0]
    return correct / total if total > 0 else 0.0


def plot_loss_curve(loss_values: list, title: str = "Loss Curve"):
    """
    绘制损失函数曲线
    
    用于可视化模型训练过程中损失值的变化趋势。

    参数
    ----------
    loss_values : list
        损失值列表，每个元素对应一次迭代的损失
    title : str, 默认 "Loss Curve"
        图表标题
    """
    plt.figure()
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()