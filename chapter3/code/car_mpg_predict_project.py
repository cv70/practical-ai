#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度学习入门项目实战：汽车MPG预测
=========================================

项目目标：
1. 使用PyTorch搭建神经网络
2. 使用全连接网络实现汽车MPG预测任务
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

# 汽车MPG数据集
class CarMPGDataset(Dataset):
    def __init__(self, filepath):
        column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin', 'Car Name']
        df = pd.read_csv(filepath, names=column_names, na_values='?', sep='\s+')

        # 处理缺失值：Horsepower用中位数填充（真实数据处理方式）
        df['Horsepower'] = df['Horsepower'].fillna(df['Horsepower'].median())

        # 删除无关列
        df = df.drop(columns=['Car Name'])

        # 独热编码Origin
        df = pd.get_dummies(df, columns=['Origin'], prefix='Origin')

        # 分离特征和标签
        X = np.array(df.drop('MPG', axis=1).values, dtype=np.float64)
        y = np.array(df['MPG'].values, dtype=np.float64)

        # 标准化：自己实现（减均值，除标准差）
        def standardize(x):
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            # 添加一个小的epsilon值避免除以零的情况
            epsilon = 1e-8
            std = np.where(std == 0, epsilon, std)  # 如果标准差为0，则替换为epsilon
            return (x - mean) / std

        X = standardize(X)

        self.x_data = torch.tensor(X, dtype=torch.float32)
        print(y.shape)
        self.y_data = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        print(self.y_data.shape)
        
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

# 加载汽车MPG数据集
def load_car_mpg_data():
    """加载汽车MPG数据集"""
    # 下载地址: https://archive.ics.uci.edu/dataset/9/auto+mpg
    # 注意: 由于版权原因，我们使用数据集的公开特征描述
    # 将下载好的数据集放到datasets目录下，命名为auto-mpg.data
    dataset = CarMPGDataset('./datasets/auto-mpg.data')

    # 特征
    feature_names = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin', 'Car Name']

    # 预测量
    target_name = 'MPG'
    
    return dataset.x_data, dataset.y_data, feature_names, target_name

# 汽车MPG预测任务（监督学习 - 回归）
print("汽车MPG预测任务")
print("-" * 30)

# 加载真实的汽车MPG数据集
X_car_mpg, y_car_mpg, feature_names_car_mpg, target_name_car_mpg = load_car_mpg_data()

print(f"汽车MPG数据集形状: {X_car_mpg.shape}")
print(f"特征名称: {feature_names_car_mpg}")

# 划分数据集
X_train_car_mpg, X_test_car_mpg, y_train_car_mpg, y_test_car_mpg = train_test_split(
    X_car_mpg, y_car_mpg, test_size=0.1
)

# 标准化特征
scaler_car_mpg = StandardScaler()
X_train_car_mpg_scaled = scaler_car_mpg.fit_transform(X_train_car_mpg)
X_test_car_mpg_scaled = scaler_car_mpg.transform(X_test_car_mpg)

# 标准化目标变量
y_train_car_mpg_mean = torch.mean(y_train_car_mpg)
y_train_car_mpg_std = torch.std(y_train_car_mpg)
y_test_car_mpg_mean = torch.mean(y_test_car_mpg)
y_test_car_mpg_std = torch.std(y_test_car_mpg)

# 避免除零错误
if y_train_car_mpg_std == 0:
    y_train_car_mpg_std = 1

if y_test_car_mpg_std == 0:
    y_test_car_mpg_std = 1

y_train_car_mpg_scaled = (y_train_car_mpg - y_train_car_mpg_mean) / y_train_car_mpg_std
y_test_car_mpg_scaled = (y_test_car_mpg - y_test_car_mpg_mean) / y_test_car_mpg_std

# 转换为PyTorch张量
X_train_car_mpg_tensor = torch.FloatTensor(X_train_car_mpg_scaled).to(device)
y_train_car_mpg_tensor = torch.FloatTensor(y_train_car_mpg_scaled).reshape(-1, 1).to(device)
X_test_car_mpg_tensor = torch.FloatTensor(X_test_car_mpg_scaled).to(device)
y_test_car_mpg_tensor = torch.FloatTensor(y_test_car_mpg_scaled).reshape(-1, 1).to(device)

# 定义神经网络回归模型 - 使用更简单的架构
class CarMPG(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super().__init__()
        # 使用更复杂的网络结构，包含批归一化、残差连接和不同的激活函数
        layers = []
        
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())  # 使用SiLU激活函数，通常比ReLU效果更好
        
        # 隐藏层 - 使用残差连接增强表达能力
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            # layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # 将所有层组合成一个序列模型
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# 初始化模型 - 使用更简单的网络结构
car_mpg_input_dim = X_car_mpg.shape[1]
car_mpg_hidden_dims = [128]  # 简化网络结构
car_mpg_output_dim = 1
model_car_mpg = CarMPG(car_mpg_input_dim, car_mpg_hidden_dims, car_mpg_output_dim).to(device)

# 定义损失函数和优化器 - 使用更适合回归任务的设置
criterion_car_mpg = nn.MSELoss()
optimizer_car_mpg = optim.Adam(model_car_mpg.parameters(), lr=0.01)  # 使用默认Adam优化器
scheduler = optim.lr_scheduler.StepLR(optimizer_car_mpg, step_size=300, gamma=0.5)

# 训练模型 - 简化训练策略
epochs_car_mpg = 200
losses_car_mpg = []
best_loss = float('inf')
patience_counter = 0

for epoch in range(epochs_car_mpg):
    # 训练模式
    model_car_mpg.train()
    
    # 前向传播
    outputs = model_car_mpg(X_train_car_mpg_tensor)
    loss = criterion_car_mpg(outputs, y_train_car_mpg_tensor)
    
    # 反向传播和优化
    optimizer_car_mpg.zero_grad()
    loss.backward()
    optimizer_car_mpg.step()
    scheduler.step()  # 更新学习率
    
    losses_car_mpg.append(loss.item())

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs_car_mpg}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

# 评估模型
model_car_mpg.eval()
with torch.no_grad():
    y_pred_car_mpg_scaled = model_car_mpg(X_test_car_mpg_tensor)
    # 反标准化预测结果
    y_pred_car_mpg = y_pred_car_mpg_scaled.cpu() * y_train_car_mpg_std + y_train_car_mpg_mean
    y_test_car_mpg_original = y_test_car_mpg
    
    # 计算MSE和R²（使用PyTorch函数处理张量）
    mse_car_mpg = torch.mean((y_pred_car_mpg - y_test_car_mpg_original) ** 2)
    
    y_test_mean = torch.mean(y_test_car_mpg_original)
    ss_tot = torch.sum((y_test_car_mpg_original - y_test_mean) ** 2)
    ss_res = torch.sum((y_test_car_mpg_original - y_pred_car_mpg) ** 2)
    r2_car_mpg = 1 - ss_res / ss_tot
    
    print(f'测试集 MSE: {mse_car_mpg.item():.4f}')
    print(f'测试集 R²: {r2_car_mpg.item():.4f}')

# 可视化汽车MPG预测结果
plt.figure(figsize=(15, 5))

# 训练损失曲线
plt.subplot(1, 3, 1)
plt.plot(losses_car_mpg)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('汽车MPG预测训练损失')

# 预测值 vs 真实值
plt.subplot(1, 3, 2)
y_test_np = y_test_car_mpg_original.cpu().numpy()
y_pred_np = y_pred_car_mpg.cpu().numpy()
plt.scatter(y_test_np, y_pred_np, alpha=0.7)
plt.xlabel('真实MPG')
plt.ylabel('预测MPG')
plt.title('汽车MPG预测结果')
plt.plot([y_test_np.min(), y_test_np.max()], 
         [y_test_np.min(), y_test_np.max()], 'r--', lw=2)

# 残差图
plt.subplot(1, 3, 3)
y_test_np_flat = y_test_np.flatten()
y_pred_np_flat = y_pred_np.flatten()
residuals = y_test_np_flat - y_pred_np_flat
plt.scatter(y_pred_np_flat, residuals, alpha=0.7)
plt.xlabel('预测MPG')
plt.ylabel('残差')
plt.title('残差分析')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig(f'{output_dir}/car_mpg_predict_results.png')
plt.close()
print("已生成汽车MPG预测结果图 (car_mpg_predict_results.png)")

print("\n" + "="*50)
print("项目总结")
print("="*50)
print("汽车MPG预测任务:")
print(f"   - 使用神经网络实现了回归任务")
print(f"   - 测试集MSE: {mse_car_mpg:.4f}")
print(f"   - 测试集R²: {r2_car_mpg:.4f}")
print()
print("所有结果图表已保存到 output 目录中!")

print("\n程序执行完毕！")