#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
机器学习概述示例（纯PyTorch实现）
=============================

演示机器学习的基本流程，包括：
- 数据准备和预处理
- 特征工程
- 模型训练和评估
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
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
print(f"=== 机器学习概述示例 (设备: {device}) ===\n")

# 1. 数据准备和预处理
print("1. 数据准备和预处理")
print("-" * 30)

# 生成示例数据集（房价预测）
np.random.seed(42)
n_samples = 1000
n_features = 5

# 生成特征
X = np.random.randn(n_samples, n_features)

# 生成目标变量（房价）
# 假设房价与房屋面积、卧室数量、地理位置、房龄、周边设施相关
y = (50 * X[:, 0] +  # 房屋面积
     10 * X[:, 1] +  # 卧室数量
     30 * X[:, 2] +  # 地理位置
     -5 * X[:, 3] +  # 房龄
     15 * X[:, 4] +  # 周边设施
     np.random.randn(n_samples) * 10)  # 噪声

# 确保房价为正数
y = np.abs(y)

print(f"数据集形状: {X.shape}")
print(f"目标变量形状: {y.shape}")
print(f"特征示例 (前5行):\n{X[:5]}")
print(f"目标变量示例 (前5个):\n{y[:5]}")

# 数据可视化
plt.figure(figsize=(15, 10))

# 特征分布
for i in range(n_features):
    plt.subplot(2, 3, i+1)
    plt.hist(X[:, i], bins=30, alpha=0.7)
    plt.xlabel(f'特征 {i+1}')
    plt.ylabel('频率')
    plt.title(f'特征 {i+1} 分布')

plt.tight_layout()
plt.savefig(f'{output_dir}/features_distribution.png')
plt.close()
print("已生成特征分布图 (features_distribution.png)")

# 目标变量分布
plt.figure(figsize=(8, 6))
plt.hist(y, bins=50, alpha=0.7, color='green')
plt.xlabel('房价')
plt.ylabel('频率')
plt.title('房价分布')
plt.savefig(f'{output_dir}/price_distribution.png')
plt.close()
print("已生成房价分布图 (price_distribution.png)")

# 2. 特征工程
print("\n2. 特征工程")
print("-" * 30)

# 特征标准化
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / (X_std + 1e-8)

y_mean = np.mean(y)
y_std = np.std(y)
y_normalized = (y - y_mean) / (y_std + 1e-8)

print(f"标准化后的特征均值: {np.mean(X_normalized, axis=0)}")
print(f"标准化后的特征标准差: {np.std(X_normalized, axis=0)}")
print(f"标准化后的目标变量均值: {np.mean(y_normalized)}")
print(f"标准化后的目标变量标准差: {np.std(y_normalized)}")

# 划分数据集
train_size = int(0.8 * n_samples)
X_train = X_normalized[:train_size]
X_test = X_normalized[train_size:]
y_train = y_normalized[:train_size]
y_test = y_normalized[train_size:]

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1).to(device)

# 3. 模型训练和评估
print("\n3. 模型训练和评估")
print("-" * 30)

# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# 初始化模型
input_dim = n_features
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
losses = []

for epoch in range(epochs):
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch+1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    mse = criterion(y_pred, y_test_tensor)
    
    # 计算R²分数
    y_test_mean = torch.mean(y_test_tensor)
    ss_tot = torch.sum((y_test_tensor - y_test_mean) ** 2)
    ss_res = torch.sum((y_test_tensor - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f'测试集 MSE: {mse.item():.4f}')
    print(f'测试集 R²: {r2.item():.4f}')

# 可视化结果
plt.figure(figsize=(15, 5))

# 损失曲线
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')

# 预测值 vs 真实值
plt.subplot(1, 3, 2)
plt.scatter(y_test[:100], y_pred.cpu().numpy()[:100], alpha=0.7)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('预测值 vs 真实值')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

# 残差图
plt.subplot(1, 3, 3)
residuals = y_test[:100] - y_pred.cpu().numpy().flatten()[:100]
plt.scatter(y_pred.cpu().numpy()[:100], residuals, alpha=0.7)
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')
plt.axhline(y=0, color='r', linestyle='--')

plt.tight_layout()
plt.savefig(f'{output_dir}/ml_results.png')
plt.close()
print("已生成机器学习结果图 (ml_results.png)")

print("\n程序执行完毕！")