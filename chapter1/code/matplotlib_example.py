#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Matplotlib 数据可视化示例
========================

演示 Matplotlib 库的基础知识，包括：
- 基本图表绘制
- 图表美化
- 多子图布局
- 交互式图表
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Maple Mono NF CN']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 输出目录
output_dir = './output'
# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== Matplotlib 数据可视化示例 ===\n")

# 1. 基本图表绘制
print("1. 基本图表绘制")
print("-" * 30)

# 创建示例数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 折线图
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('正弦和余弦函数')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/sine_cosine.png')
plt.close()
print("已生成正弦和余弦函数图 (sine_cosine.png)")

# 散点图
np.random.seed(42)
x_scatter = np.random.randn(100)
y_scatter = np.random.randn(100)

plt.figure(figsize=(8, 6))
plt.scatter(x_scatter, y_scatter, alpha=0.6)
plt.xlabel('X 值')
plt.ylabel('Y 值')
plt.title('散点图示例')
plt.grid(True)
plt.savefig(f'{output_dir}/scatter_plot.png')
plt.close()
print("已生成散点图 (scatter_plot.png)")

# 柱状图
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(8, 6))
plt.bar(categories, values)
plt.xlabel('类别')
plt.ylabel('值')
plt.title('柱状图示例')
plt.savefig(f'{output_dir}/bar_chart.png')
plt.close()
print("已生成柱状图 (bar_chart.png)")

print("\n" + "="*50 + "\n")

# 2. 图表美化
print("2. 图表美化")
print("-" * 30)

# 创建更美观的图表
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, color='blue', linewidth=2, linestyle='-', marker='o', markersize=4, label='sin(x)')

# 美化设置
plt.xlabel('x (弧度)', fontsize=12)
plt.ylabel('sin(x)', fontsize=12)
plt.title('美化后的正弦函数图', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 设置坐标轴范围
plt.xlim(0, 2*np.pi)
plt.ylim(-1.2, 1.2)

# 添加注释
plt.annotate('最大值', xy=(np.pi/2, 1), xytext=(np.pi/2, 1.3),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, color='red')

plt.tight_layout()
plt.savefig(f'{output_dir}/beautiful_sine.png')
plt.close()
print("已生成美化后的正弦函数图 (beautiful_sine.png)")

print("\n" + "="*50 + "\n")

# 3. 多子图布局
print("3. 多子图布局")
print("-" * 30)

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 子图1: 折线图
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), 'b-')
axes[0, 0].set_title('正弦函数')
axes[0, 0].grid(True)

# 子图2: 散点图
x_scatter = np.random.randn(50)
y_scatter = np.random.randn(50)
axes[0, 1].scatter(x_scatter, y_scatter, c='red', alpha=0.6)
axes[0, 1].set_title('散点图')

# 子图3: 柱状图
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 2, 5]
axes[1, 0].bar(categories, values, color='green')
axes[1, 0].set_title('柱状图')

# 子图4: 饼图
sizes = [15, 30, 45, 10]
labels = ['类别A', '类别B', '类别C', '类别D']
axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('饼图')

plt.tight_layout()
plt.savefig(f'{output_dir}/subplots_example.png')
plt.close()
print("已生成多子图 (subplots_example.png)")

print("\n" + "="*50 + "\n")

# 4. 实际数据分析可视化
print("4. 实际数据分析可视化")
print("-" * 30)

# 创建示例数据集
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=12, freq='M')
sales = np.random.randint(800, 1200, 12)
profit = sales * np.random.uniform(0.1, 0.3, 12)

# 创建销售数据可视化
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制销售数据
color = 'tab:blue'
ax1.set_xlabel('月份')
ax1.set_ylabel('销售额', color=color)
ax1.plot(dates, sales, color=color, marker='o', linewidth=2, markersize=6)
ax1.tick_params(axis='y', labelcolor=color)

# 创建第二个y轴用于利润数据
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('利润', color=color)
ax2.bar(dates, profit, alpha=0.6, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('2023年销售与利润数据')
fig.tight_layout()
plt.savefig(f'{output_dir}/sales_profit.png')
plt.close()
print("已生成销售与利润数据图 (sales_profit.png)")

print("\n所有图表已生成并保存到 output 目录下！")