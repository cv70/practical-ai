#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
机器学习项目实战：汽车MPG预测
=========================================

仅使用 numpy + pandas + matplotlib。
- 手动实现神经网络（带反向传播）
- 手动实现 StandardScaler
- 使用汽车MPG数据进行回归任务（监督学习）
"""

import os
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
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # 避免除零错误
        self.std[self.std == 0] = 1
        return self
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class CarMPGDataset:
    def __init__(self, filepath):
        column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin', 'Car Name']
        df = pd.read_csv(filepath, names=column_names, na_values='?', sep=r'\s+')

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
        self.X = X
        self.y = y
        self.feature_names = df.drop('MPG', axis=1).columns.tolist()
        self.target_name = 'MPG'
        
        # 标准化特征
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # 标准化目标
        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)
        self.y_scaled = (self.y - self.y_mean) / self.y_std

    def get_data(self):
        """返回标准化后的数据"""
        return self.X_scaled, self.y_scaled, self.feature_names, self.target_name

# 定义神经网络模型（包含前向传播和反向传播）
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        初始化神经网络
        :param input_size: 输入特征数量
        :param hidden_size: 隐藏层神经元数量
        :param output_size: 输出数量（回归问题为1）
        :param learning_rate: 学习率
        """
        self.learning_rate = learning_rate

        # 使用更好的权重初始化方法（Xavier初始化）
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU激活函数的导数"""
        return (x > 0).astype(float)
    
    def forward(self, X):
        """
        前向传播
        :param X: 输入数据 (m, input_size)
        :return: 预测值 (m, output_size)
        """
        # 隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # 输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # 回归任务使用线性激活函数
        
        return self.a2
    
    def backward(self, X, y, output):
        """
        反向传播
        :param X: 输入数据 (m, input_size)
        :param y: 真实标签 (m, output_size)
        :param output: 前向传播的输出 (m, output_size)
        """
        m = X.shape[0]  # 样本数量
        
        # 计算输出层的误差
        error = output - y
        
        # 计算输出层的梯度
        dW2 = np.dot(self.a1.T, error) / m
        db2 = np.sum(error, axis=0, keepdims=True) / m
        
        # 计算隐藏层的误差
        hidden_error = np.dot(error, self.W2.T) * self.relu_derivative(self.z1)
        
        # 计算隐藏层的梯度
        dW1 = np.dot(X.T, hidden_error) / m
        db1 = np.sum(hidden_error, axis=0, keepdims=True) / m
        
        # 更新参数
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

# 汽车MPG预测任务（监督学习 - 回归）
print("汽车MPG预测任务")
print("-" * 30)

# 加载真实的汽车MPG数据集
# 下载地址: https://archive.ics.uci.edu/dataset/9/auto+mpg
# 注意: 由于版权原因，我们使用数据集的公开特征描述
# 将下载好的数据集放到datasets目录下，命名为iris.data
dataset = CarMPGDataset('./datasets/auto-mpg.data')
X_car_mpg, y_car_mpg, feature_names_car_mpg, target_name_car_mpg = dataset.get_data()

print(f"汽车MPG数据集形状: {X_car_mpg.shape}")
print(f"特征名称: {feature_names_car_mpg}")

# 划分数据集
X_train_car_mpg, X_test_car_mpg, y_train_car_mpg, y_test_car_mpg = train_test_split(
    X_car_mpg, y_car_mpg, test_size=0.1
)

# 创建神经网络模型
input_size = X_train_car_mpg.shape[1]
hidden_size = 32  # 调整隐藏层神经元数量
output_size = 1

# 使用更合适的学习率
model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.01)

# 训练参数
epochs = 1000
batch_size = 32  # 增加批量大小
losses = []

# 训练模型
for epoch in range(epochs):
    # 随机打乱数据
    indices = np.random.permutation(X_train_car_mpg.shape[0])
    X_shuffled = X_train_car_mpg[indices]
    y_shuffled = y_train_car_mpg[indices]  # 注意：y_shuffled 是一维数组
    
    # 分批训练
    for i in range(0, X_shuffled.shape[0], batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]  # 保持为一维数组
        
        # 确保 y_batch 是二维数组 (batch_size, 1)
        y_batch = y_batch.reshape(-1, 1)
        
        # 前向传播
        output = model.forward(X_batch)
        
        # 反向传播
        model.backward(X_batch, y_batch, output)
    
    # 计算并存储损失
    train_output = model.forward(X_train_car_mpg)
    train_loss = np.mean((train_output - y_train_car_mpg.reshape(-1, 1)) ** 2)
    losses.append(train_loss)
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}')

# 评估模型
y_pred_car_mpg = model.forward(X_test_car_mpg)

# 计算MSE和R²
mse_car_mpg = np.mean((y_pred_car_mpg - y_test_car_mpg.reshape(-1, 1)) ** 2)
y_test_mean = np.mean(y_test_car_mpg)
ss_tot = np.sum((y_test_car_mpg - y_test_mean) ** 2)
ss_res = np.sum((y_test_car_mpg - y_pred_car_mpg.flatten()) ** 2)
r2_car_mpg = 1 - (ss_res / ss_tot)

# 反标准化预测值以获得实际MPG值
y_pred_actual = y_pred_car_mpg.flatten() * dataset.y_std + dataset.y_mean
y_test_actual = y_test_car_mpg * dataset.y_std + dataset.y_mean

# 计算实际值的MSE和R²
mse_actual = np.mean((y_pred_actual - y_test_actual) ** 2)
ss_tot_actual = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
ss_res_actual = np.sum((y_test_actual - y_pred_actual) ** 2)
r2_actual = 1 - (ss_res_actual / ss_tot_actual)

# 打印结果
print(f'标准化测试集 MSE: {mse_car_mpg:.4f}')
print(f'标准化测试集 R²: {r2_car_mpg:.4f}')
print(f'实际值测试集 MSE: {mse_actual:.4f}')
print(f'实际值测试集 R²: {r2_actual:.4f}')

# 可视化汽车MPG预测结果
plt.figure(figsize=(15, 5))

# 训练损失曲线
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('汽车MPG预测训练损失')

# 预测值 vs 真实值（实际值）
plt.subplot(1, 3, 2)
plt.scatter(y_test_actual, y_pred_actual, alpha=0.7)
plt.xlabel('真实MPG')
plt.ylabel('预测MPG')
plt.title('汽车MPG预测结果')
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)

# 残差图（实际值）
plt.subplot(1, 3, 3)
residuals = y_test_actual - y_pred_actual
plt.scatter(y_pred_actual, residuals, alpha=0.7)
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
print(f"   - 使用NumPy实现了神经网络模型（含前向传播和反向传播）")
print(f"   - 实际值测试集MSE: {mse_actual:.4f}")
print(f"   - 实际值测试集R²: {r2_actual:.4f}")
print()
print("所有结果图表已保存到 output 目录中!")

print("\n程序执行完毕！")