#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch张量操作示例
=================

本文件演示了PyTorch中张量的基本操作，包括：
- 张量创建和基本属性
- 张量运算
- 自动微分机制
- GPU加速计算
"""

import torch
import numpy as np

print("=== PyTorch张量操作示例 ===\n")

# 1. 张量创建和基本属性
print("1. 张量创建和基本属性")
print("-" * 30)

# 从列表创建张量
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
print(f"从列表创建的张量: {tensor_from_list}")
print(f"张量形状: {tensor_from_list.shape}")
print(f"张量数据类型: {tensor_from_list.dtype}")
print(f"张量设备: {tensor_from_list.device}")

# 创建特定形状的张量
zeros_tensor = torch.zeros(3, 4)
ones_tensor = torch.ones(2, 3)
random_tensor = torch.rand(2, 3)
print(f"\n零张量 (3x4):\n{zeros_tensor}")
print(f"一张量 (2x3):\n{ones_tensor}")
print(f"随机张量 (2x3):\n{random_tensor}")

# 从NumPy数组创建张量
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"\n从NumPy数组创建的张量:\n{tensor_from_numpy}")

# 2. 张量运算
print("\n2. 张量运算")
print("-" * 30)

# 基本算术运算
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")

# 矩阵运算
matrix_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
matrix_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

print(f"\n矩阵A:\n{matrix_a}")
print(f"矩阵B:\n{matrix_b}")
print(f"矩阵乘法 A @ B:\n{torch.mm(matrix_a, matrix_b)}")
print(f"矩阵乘法 A @ B (另一种写法):\n{torch.matmul(matrix_a, matrix_b)}")

# 统计运算
data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"\n数据: {data}")
print(f"求和: {torch.sum(data)}")
print(f"平均值: {torch.mean(data)}")
print(f"最大值: {torch.max(data)}")
print(f"最小值: {torch.min(data)}")
print(f"标准差: {torch.std(data)}")

# 3. 张量索引和切片
print("\n3. 张量索引和切片")
print("-" * 30)

matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"原始矩阵:\n{matrix}")
print(f"第一行: {matrix[0]}")
print(f"第一列: {matrix[:, 0]}")
print(f"中间元素: {matrix[1, 1]}")
print(f"前两行:\n{matrix[:2, :]}")

# 4. 张量变换
print("\n4. 张量变换")
print("-" * 30)

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"原始张量:\n{tensor}")
print(f"转置:\n{tensor.T}")
print(f"重塑为 (3, 2):\n{tensor.reshape(3, 2)}")
print(f"展平:\n{tensor.flatten()}")

# 5. GPU加速计算
print("\n5. GPU加速计算")
print("-" * 30)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前设备: {device}")

if torch.cuda.is_available():
    # 将张量移动到GPU
    gpu_tensor = tensor.to(device)
    print(f"GPU上的张量: {gpu_tensor}")
    print(f"张量设备: {gpu_tensor.device}")
else:
    print("CUDA不可用，无法使用GPU加速")

# 6. 自动微分机制
print("\n6. 自动微分机制")
print("-" * 30)

# 创建需要计算梯度的张量
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 定义计算
z = x**2 + x*y + y**2
print(f"计算表达式: z = x² + xy + y²")
print(f"x = {x.item()}, y = {y.item()}")
print(f"z = {z.item()}")

# 计算梯度
z.backward()
print(f"\n梯度:")
print(f"dz/dx = {x.grad.item()}")
print(f"dz/dy = {y.grad.item()}")

# 验证解析解
# dz/dx = 2x + y = 2*2 + 3 = 7
# dz/dy = x + 2y = 2 + 2*3 = 8
print(f"\n解析解验证:")
print(f"dz/dx (解析): 2x + y = 2*{x.item()} + {y.item()} = {2*x.item() + y.item()}")
print(f"dz/dy (解析): x + 2y = {x.item()} + 2*{y.item()} = {x.item() + 2*y.item()}")

print("\n程序执行完毕！")