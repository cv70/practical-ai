#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
神经网络基础示例
==============

本文件演示了神经网络的基本原理，包括：
- 神经元和激活函数
- 前向传播和反向传播
- 损失函数和优化器
- 梯度下降算法
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

print("=== 神经网络基础示例 ===\n")

# 1. 神经元和激活函数
print("1. 神经元和激活函数")
print("-" * 30)

# 创建简单的神经元示例
def simple_neuron(x, weights, bias):
    """简单的神经元计算"""
    return torch.sigmoid(torch.dot(weights, x) + bias)

# 输入数据
x = torch.tensor([0.5, 0.3])
weights = torch.tensor([0.8, 0.2])
bias = torch.tensor(0.1)

# 计算神经元输出
output = simple_neuron(x, weights, bias)
print(f"输入: {x}")
print(f"权重: {weights}")
print(f"偏置: {bias}")
print(f"输出 (sigmoid激活): {output:.4f}")

# 可视化不同激活函数
x_vals = torch.linspace(-5, 5, 100)
activations = {
    'Sigmoid': torch.sigmoid(x_vals),
    'ReLU': torch.relu(x_vals),
    'Tanh': torch.tanh(x_vals),
    'Linear': x_vals
}

plt.figure(figsize=(15, 10))
for i, (name, y_vals) in enumerate(activations.items()):
    plt.subplot(2, 2, i+1)
    plt.plot(x_vals.numpy(), y_vals.numpy())
    plt.title(f'{name} 激活函数')
    plt.xlabel('输入')
    plt.ylabel('输出')
    plt.grid(True)

plt.tight_layout()
plt.savefig(f'{output_dir}/activation_functions.png')
plt.close()
print("已生成激活函数图 (activation_functions.png)")

# 2. 前向传播和反向传播
print("\n2. 前向传播和反向传播")
print("-" * 30)

# 定义简单的两层网络
class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.linear1 = nn.Linear(2, 3)
        self.linear2 = nn.Linear(3, 1)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

# 创建网络实例
net = SimpleNetwork()
print("网络结构:")
print(net)

# 创建示例输入和目标
input_data = torch.tensor([[0.5, 0.3]])
target = torch.tensor([[0.8]])

print(f"\n输入数据: {input_data}")
print(f"目标值: {target}")

# 前向传播
output = net(input_data)
print(f"前向传播输出: {output.item():.4f}")

# 计算损失
criterion = nn.MSELoss()
loss = criterion(output, target)
print(f"MSE损失: {loss.item():.4f}")

# 反向传播
loss.backward()
print("\n反向传播后的梯度:")
for name, param in net.named_parameters():
    print(f"{name}: {param.grad}")

# 3. 梯度下降算法
print("\n3. 梯度下降算法")
print("-" * 30)

# 手动实现梯度下降
learning_rate = 0.1
print(f"学习率: {learning_rate}")

# 保存原始参数
original_params = {}
for name, param in net.named_parameters():
    original_params[name] = param.clone()

# 执行一次梯度下降更新
with torch.no_grad():
    for param in net.parameters():
        param -= learning_rate * param.grad

# 显示参数更新
print("\n参数更新前后对比:")
for name, param in net.named_parameters():
    print(f"{name}:")
    print(f"  更新前: {original_params[name]}")
    print(f"  更新后: {param}")
    print(f"  变化量: {param - original_params[name]}")

# 4. PyTorch自动微分机制
print("\n4. PyTorch自动微分机制")
print("-" * 30)

# 创建需要计算梯度的张量
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 定义计算图
z = x**2 + 2*x*y + y**3
print(f"计算表达式: z = x² + 2xy + y³")
print(f"x = {x.item()}, y = {y.item()}")
print(f"z = {z.item()}")

# 计算梯度
z.backward()
print(f"\n自动微分结果:")
print(f"dz/dx = {x.grad.item()}")
print(f"dz/dy = {y.grad.item()}")

# 验证解析解
# dz/dx = 2x + 2y = 2*2 + 2*3 = 10
# dz/dy = 2x + 3y² = 2*2 + 3*9 = 31
print(f"\n解析解验证:")
print(f"dz/dx (解析): 2x + 2y = 2*{x.item()} + 2*{y.item()} = {2*x.item() + 2*y.item()}")
print(f"dz/dy (解析): 2x + 3y² = 2*{x.item()} + 3*{y.item()}² = {2*x.item() + 3*y.item()**2}")

print("\n程序执行完毕！")