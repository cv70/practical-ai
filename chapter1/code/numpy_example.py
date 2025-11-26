#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NumPy 科学计算示例
=================

演示 NumPy 库的基础知识，包括：
- 数组创建与操作
- 数学运算
- 广播机制
- 线性代数运算
"""

import numpy as np

print("=== NumPy 科学计算示例 ===\n")

# 1. 数组创建与操作
print("1. 数组创建与操作")
print("-" * 30)

# 创建数组的不同方式
arr1 = np.array([1, 2, 3, 4, 5])
print(f"从列表创建数组: {arr1}")

arr2 = np.zeros(5)
print(f"创建全零数组: {arr2}")

arr3 = np.ones((2, 3))
print(f"创建全一数组:\n{arr3}")

arr4 = np.arange(0, 10, 2)
print(f"创建等差数组: {arr4}")

arr5 = np.linspace(0, 1, 5)
print(f"创建等间距数组: {arr5}")

# 数组属性
print(f"\n数组 arr1 的形状: {arr1.shape}")
print(f"数组 arr3 的形状: {arr3.shape}")
print(f"数组 arr1 的数据类型: {arr1.dtype}")

# 数组索引和切片
print(f"\narr1 的第一个元素: {arr1[0]}")
print(f"arr1 的最后三个元素: {arr1[-3:]}")
print(f"arr3 的第一行: {arr3[0, :]}")

print("\n" + "="*50 + "\n")

# 2. 数学运算
print("2. 数学运算")
print("-" * 30)

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"数组 a: {a}")
print(f"数组 b: {b}")

# 基本运算
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")

# 数学函数
print(f"a 的平方根: {np.sqrt(a)}")
print(f"b 的指数: {np.exp(b)}")
print(f"a 的正弦值: {np.sin(a)}")

# 统计运算
print(f"a 的总和: {np.sum(a)}")
print(f"b 的平均值: {np.mean(b)}")
print(f"a 的最大值: {np.max(a)}")
print(f"b 的标准差: {np.std(b)}")

print("\n" + "="*50 + "\n")

# 3. 广播机制
print("3. 广播机制")
print("-" * 30)

# 广播示例
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

vector = np.array([10, 20, 30])

print(f"矩阵:\n{matrix}")
print(f"向量: {vector}")

# 广播加法
result = matrix + vector
print(f"矩阵 + 向量 (广播):\n{result}")

print("\n" + "="*50 + "\n")

# 4. 线性代数运算
print("4. 线性代数运算")
print("-" * 30)

# 矩阵乘法
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print(f"矩阵 A:\n{A}")
print(f"矩阵 B:\n{B}")

# 矩阵乘法
C = np.dot(A, B)
print(f"A × B:\n{C}")

# 矩阵转置
print(f"A 的转置:\n{A.T}")

# 行列式
det_A = np.linalg.det(A)
print(f"A 的行列式: {det_A:.2f}")

# 逆矩阵
inv_A = np.linalg.inv(A)
print(f"A 的逆矩阵:\n{inv_A}")

# 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"A 的特征值: {eigenvalues}")
print(f"A 的特征向量:\n{eigenvectors}")

print("\n程序执行完毕！")