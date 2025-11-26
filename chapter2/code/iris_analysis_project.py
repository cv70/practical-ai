#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
机器学习项目实战：鸢尾花分类与聚类分析
=================================================

仅使用 numpy + pandas + matplotlib。
- 手动实现神经网络（带反向传播）
- 手动实现 StandardScaler
- 使用鸢尾花数据集进行分类任务（监督学习）
- 对数据进行聚类分析（无监督学习）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Maple Mono NF CN']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 输出目录
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== 纯 NumPy 机器学习项目实战 ===\n")

# ==============================
# 工具函数
# ==============================

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n = X.shape[0]
    indices = np.random.permutation(n)
    n_test = int(n * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # 防止除零
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# 加载鸢尾花数据
def load_iris_data():
    # 下载地址: https://archive.ics.uci.edu/dataset/53/iris
    # 特征: 花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度
    # 类别: Setosa, Versicolour, Virginica
    # 注意: 由于版权原因，我们使用数据集的公开特征描述
    # 将下载好的数据集放到datasets目录下，命名为iris.data
    df = pd.read_csv('./datasets/iris.data', header=None,
                     names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    df['species'] = df['species'].map(species_map)
    X = df.iloc[:, :4].values.astype(np.float32)
    y = df.iloc[:, 4].values.astype(np.int64)
    feature_names = ['花萼长度 (cm)', '花萼宽度 (cm)', '花瓣长度 (cm)', '花瓣宽度 (cm)']
    target_names = ['Setosa', 'Versicolour', 'Virginica']
    return X, y, feature_names, target_names

# 手动实现神经网络（MLP）
class MLPClassifier:
    def __init__(self, input_dim, hidden_dim, num_classes, lr=0.01, seed=42):
        np.random.seed(seed)
        self.lr = lr
        # 初始化权重（Xavier 初始化）
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))
        self.W3 = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros((1, num_classes))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 数值稳定
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_probs = -np.log(y_pred[range(m), y_true] + 1e-8)
        return np.sum(log_probs) / m

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]

        # 输出层梯度
        dz3 = y_pred.copy()
        dz3[range(m), y_true] -= 1
        dz3 /= m

        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 更新参数
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def fit(self, X, y, epochs=500):
        losses = []
        accuracies = []
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            acc = np.mean(np.argmax(y_pred, axis=1) == y)

            # 反向传播
            self.backward(X, y, y_pred)

            losses.append(loss)
            accuracies.append(acc)

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, Accuracy: {acc:.4f}')

        return losses, accuracies

# 加载数据
X_iris, y_iris, feature_names_iris, target_names_iris = load_iris_data()
print(f"鸢尾花数据集形状: {X_iris.shape}")
print(f"特征名称: {feature_names_iris}")
print(f"类别名称: {target_names_iris}")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.1, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练神经网络
input_dim = X_iris.shape[1]
hidden_dim = 16
num_classes = 3

mlp = MLPClassifier(input_dim, hidden_dim, num_classes, lr=0.01)
losses_iris, accuracies_iris = mlp.fit(X_train_scaled, y_train, epochs=1000)

# 测试准确率
y_pred_iris = mlp.predict(X_test_scaled)
accuracy_iris = np.mean(y_pred_iris == y_test)
print(f'测试集准确率: {accuracy_iris:.4f}')

# 可视化分类结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(losses_iris)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('鸢尾花分类训练损失')

plt.subplot(1, 3, 2)
plt.plot(accuracies_iris)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('鸢尾花分类训练准确率')

plt.subplot(1, 3, 3)
colors = ['red', 'green', 'blue']
for i in range(3):
    points = X_test[y_test == i]
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.7, label=target_names_iris[i])
plt.xlabel(feature_names_iris[0])
plt.ylabel(feature_names_iris[1])
plt.title('鸢尾花测试集分布')
plt.legend()

plt.tight_layout()
plt.savefig(f'{output_dir}/iris_classification_results.png')
plt.close()
print("已生成鸢尾花分类结果图 (iris_classification_results.png)")

print("\n" + "="*50 + "\n")

# K-means 聚类
class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        np.random.seed(42)
        n_samples, n_features = X.shape
        # 随机选择初始中心
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[idx].astype(np.float64)

        for i in range(self.max_iters):
            # 计算距离：(n_samples, k)
            diff = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]  # 广播
            distances = np.linalg.norm(diff, axis=2)
            labels = np.argmin(distances, axis=1)

            # 更新中心
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j]
                for j in range(self.k)
            ])

            # 收敛判断
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                print(f"K-means在第 {i+1} 次迭代后收敛")
                break

            self.centroids = new_centroids

        self.labels_ = labels
        return labels

# 应用 K-means
kmeans = KMeans(k=3)
labels_kmeans = kmeans.fit(X_iris)

print(f"聚类中心:\n{kmeans.centroids}")

# 可视化 K-means
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for i in range(3):
    points = X_iris[y_iris == i]
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.7, label=target_names_iris[i])
plt.xlabel(feature_names_iris[0])
plt.ylabel(feature_names_iris[1])
plt.title('鸢尾花原始数据')
plt.legend()

plt.subplot(1, 3, 2)
for i in range(3):
    points = X_iris[labels_kmeans == i]
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.7)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
            c='black', marker='x', s=200, linewidths=3, label='聚类中心')
plt.xlabel(feature_names_iris[0])
plt.ylabel(feature_names_iris[1])
plt.title('K-means聚类结果')
plt.legend()

plt.subplot(1, 3, 3)
markers = ['o', 's', '^']
for i in range(3):
    for j in range(3):
        mask = (y_iris == i) & (labels_kmeans == j)
        pts = X_iris[mask]
        if len(pts) > 0:
            plt.scatter(pts[:, 0], pts[:, 1], c=colors[i], alpha=0.7, marker=markers[j])
plt.xlabel(feature_names_iris[0])
plt.ylabel(feature_names_iris[1])
plt.title('聚类 vs 真实标签')

legend_elements = []
for i, name in enumerate(target_names_iris):
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                                     markersize=8, label=f'真实-{name}'))
for j, mk in enumerate(markers):
    legend_elements.append(plt.Line2D([0], [0], marker=mk, color='w', markerfacecolor='gray',
                                     markersize=8, label=f'聚类-{j+1}'))
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(f'{output_dir}/kmeans_clustering_results.png')
plt.close()
print("已生成K-means聚类结果图 (kmeans_clustering_results.png)")

# ==============================
# 项目总结
# ==============================
print("\n" + "="*50)
print("项目总结")
print("="*50)
print("1. 鸢尾花分类任务:")
print(f"   - 实现三层神经网络（含反向传播）")
print(f"   - 测试集准确率达到 {accuracy_iris:.2%}")
print()
print("2. K-means聚类分析:")
print(f"   - 实现 K-means 算法")
print(f"   - 成功识别出3个聚类中心")
print()
print("所有结果图表已保存到 output 目录中!")
print("\n程序执行完毕！")