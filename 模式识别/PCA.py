import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Z_score 标准化函数
def my_z_score_normalization(X):
    mean = np.mean(X, axis = 0) #对每一列求平均值
    std = np.std(X, axis = 0) # 对每一列求标准差
    X_normaliztion = (X - mean) / std
    return X_normaliztion

# PCA 降维函数
def my_pca(X, n_components):
    # 中心化数据
    mean = np.mean(X, axis = 0)
    X_center = X- mean

    # 协方差矩阵
    X_cov = np.cov(X_center.T)

    # 计算矩阵的特征值和特征向量
    eig_value, eig_vector = np.linalg.eig(np.mat(X_cov))

    # 将特征值由大到小进行排序
    idx = eig_value.argsort()[::-1]
    eig_vector = eig_vector[:, idx]

    # 选择特征向量，并降维
    W = eig_vector[:, :n_components]
    X_pca = X_center.dot(W)

    return X_pca

# LDA 降维函数
def my_LDA(X, y, n_components):
    # Find class mean
    class_means = []
    for label in np.unique(y):
        X_label = X[y == label]
        class_mean = np.mean(X_label, axis = 0)
        class_means.append(class_mean)
    class_means = np.array(class_means)

    # 计算类内散度矩阵
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for label, mean in zip(np.unique(y), class_means):
        X_label = X[y == label]
        X_centered = X_label - mean
        Sw += X_centered.T.dot(X_centered)

    # 计算类间散度矩阵
    mean_total = np.mean(X, axis = 0)
    Sb = np.zeros((X.shape[1], X.shape[1]))
    for label, mean in zip(np.unique(y), class_means):
        n_samples = np.sum(y == label)
        mean_diff = (mean - mean_total).reshape(-1, 1)
        Sb += n_samples * mean_diff.dot(mean.diff.T)

    # 计算特征向量和特征值
    eig_value, eig_vector = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

    # 将特征值由大到小进行排序
    idx = eig_value.argsort()[::-1]
    eig_vector = eig_vector[:, idx]

    # 选择特征向量，并降维
    W = eig_vector[:, :n_components]
    X_LDA= X.dot(W)

    return X_LDA


# 导入数据
Red_wine = pd.read_csv('winequality-red.csv', delimiter = ';')
White_wine = pd.read_csv('winequality-white.csv', delimiter = ';')

# 添加分类标签并合并数据
Red_wine['Typre'] = 0
White_wine['Type'] = 1
data = pd.concat([Red_wine, White_wine])

# 数据预处理
# 1.1 特征值与标签分离
X = data.drop(['Type'], axis = 1)
y = data['Type']


# 1.2 数据集划分
np.random.seed(42)
print(len(X))
indices = np.random.permutation(len(X))
print(indices.shape)
train_size = int(0.7 * len(X))
X


# 2. 特征值标准化
X_red_norm = my_z_score_normalization(X_red)
X_white_norm = my_z_score_normalization(X_white)


# 3. 降维
# 3.1 PCA
