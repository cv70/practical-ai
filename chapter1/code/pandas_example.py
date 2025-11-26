#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pandas 数据分析示例
==================

演示 Pandas 库的基础知识，包括：
- Series 与 DataFrame
- 数据读取与保存
- 数据清洗与预处理
- 数据分组与聚合
"""

import os
import pandas as pd
import numpy as np

# 输出目录
output_dir = './output'
# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== Pandas 数据分析示例 ===\n")

# 1. Series 与 DataFrame
print("1. Series 与 DataFrame")
print("-" * 30)

# 创建 Series
series = pd.Series([1, 3, 5, np.nan, 6, 8])
print("创建 Series:")
print(series)
print()

# 创建 DataFrame
dates = pd.date_range('20230101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print("创建 DataFrame:")
print(df)
print()

# 从字典创建 DataFrame
data = {
    'name': ['张三', '李四', '王五', '赵六'],
    'age': [20, 21, 19, 22],
    'score': [85, 92, 78, 88],
    'city': ['北京', '上海', '广州', '深圳']
}
df2 = pd.DataFrame(data)
print("从字典创建 DataFrame:")
print(df2)
print()

print("\n" + "="*50 + "\n")

# 2. 数据读取与保存
print("2. 数据读取与保存")
print("-" * 30)

# 创建示例数据并保存为 CSV
sample_data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 70000, 55000, 65000],
    'department': ['IT', 'HR', 'Finance', 'IT', 'Marketing']
})

# 保存为 CSV 文件
sample_data.to_csv(f'{output_dir}/employee_data.csv', index=False)
print("数据已保存到 employee_data.csv")

# 读取 CSV 文件
df_csv = pd.read_csv(f'{output_dir}/employee_data.csv')
print("从 CSV 文件读取的数据:")
print(df_csv)
print()

print("\n" + "="*50 + "\n")

# 3. 数据清洗与预处理
print("3. 数据清洗与预处理")
print("-" * 30)

# 创建包含缺失值的数据
data_with_na = {
    'name': ['张三', '李四', None, '赵六'],
    'age': [20, np.nan, 19, 22],
    'score': [85, 92, 78, None],
    'city': ['北京', '上海', '广州', '深圳']
}
df_na = pd.DataFrame(data_with_na)
print("包含缺失值的数据:")
print(df_na)
print()

# 检查缺失值
print("缺失值统计:")
print(df_na.isnull().sum())
print()

# 处理缺失值
# 删除包含缺失值的行
df_dropped = df_na.dropna()
print("删除缺失值后的数据:")
print(df_dropped)
print()

# 填充缺失值
df_filled = df_na.fillna({'name': '未知', 'age': df_na['age'].mean(), 'score': df_na['score'].mean()})
print("填充缺失值后的数据:")
print(df_filled)
print()

print("\n" + "="*50 + "\n")

# 4. 数据分组与聚合
print("4. 数据分组与聚合")
print("-" * 30)

# 使用之前保存的员工数据
print("员工数据:")
print(df_csv)
print()

# 按部门分组
grouped = df_csv.groupby('department')
print("按部门分组的平均薪资:")
print(grouped['salary'].mean())
print()

# 多列聚合
agg_result = df_csv.groupby('department').agg({
    'salary': ['mean', 'min', 'max'],
    'age': 'mean'
})
print("按部门分组的多列聚合统计:")
print(agg_result)
print()

# 创建透视表
pivot_table = df_csv.pivot_table(values='salary', index='department', aggfunc='mean')
print("薪资透视表:")
print(pivot_table)
print()

print("\n程序执行完毕！")