#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度学习入门项目实战：鸢尾花分类与聚类
=========================================

这是一个综合性的机器学习项目，结合了第二章所学的监督学习和无监督学习知识。

项目目标：
1. 使用PyTorch搭建神经网络
2. 使用全连接网络实现鸢尾花分类任务
3. 使用PyTorch实现鸢尾花聚类任务
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Maple Mono NF CN']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 输出目录
output_dir = './output'
# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"=== 机器学习项目实战 (设备: {device}) ===\n")

# 自定义数据集划分函数
def train_test_split(X, y, test_size=0.2):
    """自定义数据集划分函数"""
    total = X.shape[0]
    test_num = int(total * test_size)
    X_train, X_test = X[test_num:], X[:test_num]
    y_train, y_test = y[test_num:], y[:test_num]

    return X_train, X_test, y_train, y_test

# 自定义标准化函数
class StandardScaler:
    """自定义标准化类"""
    def fit(self, X):
        # 处理PyTorch张量
        if torch.is_tensor(X):
            self.mean = torch.mean(X, dim=0)
            self.std = torch.std(X, dim=0)
        else:
            # 处理NumPy数组
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        
        # 避免除零错误
        if torch.is_tensor(self.std):
            self.std[self.std == 0] = 1
        else:
            self.std[self.std == 0] = 1
        return self
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# 鸢尾花分类数据集
class IrisDataset(Dataset):
    def __init__(self, filepath):
        # 读取dataset文件
        df = pd.read_csv(filepath, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

        # 将类别名称转换为数字
        species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        df['species'] = df['species'].map(species_map)

        # 分离特征和标签
        self.x_data = torch.tensor(df.iloc[:, 0:4].values, dtype=torch.float32)
        self.y_data = torch.tensor(df.iloc[:, 4].values, dtype=torch.long)

        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

# 加载鸢尾花分类数据集
def load_iris_data():
    """加载鸢尾花分类数据集"""
    # 下载地址: https://archive.ics.uci.edu/dataset/53/iris
    # 特征: 花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度
    # 类别: Setosa, Versicolour, Virginica
    # 注意: 由于版权原因，我们使用数据集的公开特征描述
    # 将下载好的数据集放到datasets目录下，命名为iris.data
    dataset = IrisDataset('./datasets/iris.data')
    
    feature_names = ['花萼长度 (cm)', '花萼宽度 (cm)', '花瓣长度 (cm)', '花瓣宽度 (cm)']
    target_names = ['Setosa', 'Versicolour', 'Virginica']

    return dataset.x_data, dataset.y_data, feature_names, target_names

# 1. 鸢尾花分类任务（监督学习 - 分类）
print("1. 鸢尾花分类任务")
print("-" * 30)

# 加载真实的鸢尾花数据集
X_iris, y_iris, feature_names_iris, target_names_iris = load_iris_data()

print(f"鸢尾花数据集形状: {X_iris.shape}")
print(f"特征名称: {feature_names_iris}")
print(f"类别名称: {target_names_iris}")

# 划分数据集
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.1
)

# 标准化特征
scaler_iris = StandardScaler()
X_train_iris_scaled = scaler_iris.fit_transform(X_train_iris)
X_test_iris_scaled = scaler_iris.transform(X_test_iris)

# 转换为PyTorch张量
X_train_iris_tensor = torch.FloatTensor(X_train_iris_scaled).to(device)
y_train_iris_tensor = torch.LongTensor(y_train_iris).to(device)
X_test_iris_tensor = torch.FloatTensor(X_test_iris_scaled).to(device)
y_test_iris_tensor = torch.LongTensor(y_test_iris).to(device)

# 定义神经网络分类模型
class IrisClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型
input_dim_iris = X_iris.shape[1]
hidden_dim_iris = 16
num_classes_iris = len(target_names_iris)
model_iris = IrisClassifier(input_dim_iris, hidden_dim_iris, num_classes_iris).to(device)

# 定义损失函数和优化器
criterion_iris = nn.CrossEntropyLoss()
optimizer_iris = optim.Adam(model_iris.parameters(), lr=0.01)

# 训练模型
epochs_iris = 500
losses_iris = []
accuracies_iris = []

for epoch in range(epochs_iris):
    # 前向传播
    outputs = model_iris(X_train_iris_tensor)
    loss = criterion_iris(outputs, y_train_iris_tensor)
    
    # 反向传播和优化
    optimizer_iris.zero_grad()
    loss.backward()
    optimizer_iris.step()
    
    # 计算训练准确率
    with torch.no_grad():
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_train_iris_tensor).float().mean()
        losses_iris.append(loss.item())
        accuracies_iris.append(accuracy.item())
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs_iris}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')

# 评估模型
model_iris.eval()
with torch.no_grad():
    y_pred_iris_logits = model_iris(X_test_iris_tensor)
    _, y_pred_iris = torch.max(y_pred_iris_logits.data, 1)
    accuracy_iris = (y_pred_iris == y_test_iris_tensor).float().mean()
    print(f'测试集准确率: {accuracy_iris.item():.4f}')

# 可视化鸢尾花分类结果
plt.figure(figsize=(15, 5))

# 训练损失曲线
plt.subplot(1, 3, 1)
plt.plot(losses_iris)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('鸢尾花分类训练损失')

# 训练准确率曲线
plt.subplot(1, 3, 2)
plt.plot(accuracies_iris)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('鸢尾花分类训练准确率')

# 测试集预测结果
plt.subplot(1, 3, 3)
colors = ['red', 'green', 'blue']
for i in range(num_classes_iris):
    points = X_test_iris[y_test_iris == i]
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

# 3. K-means聚类分析（无监督学习）
print("2. K-means聚类分析")
print("-" * 30)

# 使用鸢尾花数据进行聚类分析
print(f"使用鸢尾花数据进行聚类分析，数据形状: {X_iris.shape}")

# 实现K-means算法
class KMeans:
    def __init__(self, k, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
    
    def fit(self, X):
        # 随机初始化聚类中心
        n_samples, n_features = X.shape
        self.centroids = X[torch.randperm(n_samples)[:self.k]]
        for i in range(self.max_iters):
            # 计算每个点到聚类中心的距离
            distances = torch.cdist(X, self.centroids)
            
            # 分配每个点到最近的聚类中心
            labels = torch.argmin(distances, dim=1)
            
            # 更新聚类中心
            new_centroids = torch.stack([
                X[labels == j].mean(dim=0) for j in range(self.k)
            ])
            
            # 检查收敛
            if torch.norm(new_centroids - self.centroids) < self.tol:
                print(f"K-means在第 {i+1} 次迭代后收敛")
                break
                
            self.centroids = new_centroids
        
        return labels
    
    def predict(self, X):
        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, dim=1)

# 应用K-means
X_iris_tensor = torch.FloatTensor(X_iris).to(device)
kmeans = KMeans(k=3)
labels_kmeans = kmeans.fit(X_iris_tensor)

print(f"聚类中心:\n{kmeans.centroids}")

# 可视化K-means聚类结果
plt.figure(figsize=(15, 5))

# 原始数据（按真实标签着色）
plt.subplot(1, 3, 1)
colors = ['red', 'green', 'blue']
for i in range(3):
    points = X_iris[y_iris == i]
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.7, label=target_names_iris[i])
plt.xlabel(feature_names_iris[0])
plt.ylabel(feature_names_iris[1])
plt.title('鸢尾花原始数据')
plt.legend()

# K-means聚类结果
plt.subplot(1, 3, 2)
for i in range(3):
    points = X_iris[labels_kmeans.cpu().numpy() == i]
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.7)
# 绘制聚类中心
centroids_cpu = kmeans.centroids.cpu().numpy()
plt.scatter(centroids_cpu[:, 0], centroids_cpu[:, 1], 
           c='black', marker='x', s=200, linewidths=3, label='聚类中心')
plt.xlabel(feature_names_iris[0])
plt.ylabel(feature_names_iris[1])
plt.title('K-means聚类结果')
plt.legend()

# 聚类与真实标签对比
plt.subplot(1, 3, 3)
# 创建混淆矩阵风格的可视化
for i in range(3):
    for j in range(3):
        points = X_iris[(y_iris == i) & (labels_kmeans.cpu().numpy() == j)]
        if len(points) > 0:
            plt.scatter(points[:, 0], points[:, 1], 
                       c=colors[i], alpha=0.7, marker=['o', 's', '^'][j])
plt.xlabel(feature_names_iris[0])
plt.ylabel(feature_names_iris[1])
plt.title('聚类结果 vs 真实标签')
# 添加图例
legend_elements = []
for i, name in enumerate(target_names_iris):
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                                     markersize=8, label=f'真实-{name}'))
for i, marker in enumerate(['o', 's', '^']):
    legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray', 
                                     markersize=8, label=f'聚类-{i+1}'))
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(f'{output_dir}/kmeans_clustering_results.png')
plt.close()
print("已生成K-means聚类结果图 (kmeans_clustering_results.png)")

print("\n" + "="*50)
print("项目总结")
print("="*50)
print("1. 鸢尾花分类任务:")
print(f"   - 使用神经网络实现了多分类任务")
print(f"   - 测试集准确率达到 {accuracy_iris.item():.2%}")
print()
print("2. K-means聚类分析:")
print(f"   - 对鸢尾花数据进行了无监督聚类")
print(f"   - 成功识别出3个聚类中心")
print()
print("所有结果图表已保存到 output 目录中!")

print("\n程序执行完毕！")