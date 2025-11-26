#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
学生 grades 分析项目
==================

这是一个综合性的数据分析项目，结合了 Python 基础、NumPy、Pandas 和 Matplotlib 的知识点。
项目目标：分析学生 grades 数据，生成报告和可视化图表。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Maple Mono NF CN']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 输出目录
output_dir = './output'
# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def generate_sample_data():
    """生成示例学生数据"""
    np.random.seed(42)
    
    # 学生信息
    student_ids = [f"STU{i:03d}" for i in range(1, 101)]
    names = [f"学生{i}" for i in range(1, 101)]
    genders = np.random.choice(['男', '女'], 100, p=[0.55, 0.45])
    ages = np.random.randint(18, 25, 100)
    
    # grades 数据 (假设满分为100分)
    math_scores = np.random.normal(75, 10, 100)
    english_scores = np.random.normal(78, 12, 100)
    physics_scores = np.random.normal(72, 15, 100)
    
    # 确保分数在合理范围内
    math_scores = np.clip(math_scores, 0, 100)
    english_scores = np.clip(english_scores, 0, 100)
    physics_scores = np.clip(physics_scores, 0, 100)
    
    # 创建DataFrame
    data = {
        'student_id': student_ids,
        'name': names,
        'gender': genders,
        'age': ages,
        'math_score': math_scores,
        'english_score': english_scores,
        'physics_score': physics_scores
    }
    
    df = pd.DataFrame(data)
    return df

def analyze_data(df):
    """分析学生数据"""
    print("=" * 60)
    print("学生 grades 分析报告")
    print("=" * 60)
    
    # 基本信息
    print(f"总学生数: {len(df)}")
    print(f"男生数量: {len(df[df['gender'] == '男'])}")
    print(f"女生数量: {len(df[df['gender'] == '女'])}")
    print(f"平均年龄: {df['age'].mean():.1f} 岁")
    print()
    
    # 各科目统计
    subjects = ['math_score', 'english_score', 'physics_score']
    subject_names = ['数学', '英语', '物理']
    
    print("各科目统计信息:")
    print("-" * 40)
    for subject, name in zip(subjects, subject_names):
        mean_score = df[subject].mean()
        median_score = df[subject].median()
        std_score = df[subject].std()
        max_score = df[subject].max()
        min_score = df[subject].min()
        
        print(f"{name}:")
        print(f"  平均分: {mean_score:.2f}")
        print(f"  中位数: {median_score:.2f}")
        print(f"  标准差: {std_score:.2f}")
        print(f"  最高分: {max_score:.2f}")
        print(f"  最低分: {min_score:.2f}")
        print()
    
    # 计算总分和平均分
    df['total_score'] = df[subjects].sum(axis=1)
    df['average_score'] = df[subjects].mean(axis=1)
    
    # 成绩等级划分
    def get_grade(score):
        if score >= 90:
            return '优秀'
        elif score >= 80:
            return '良好'
        elif score >= 70:
            return '中等'
        elif score >= 60:
            return '及格'
        else:
            return '不及格'
    
    df['grade_level'] = df['average_score'].apply(get_grade)
    
    # 成绩等级分布
    grade_distribution = df['grade_level'].value_counts()
    print("成绩等级分布:")
    print("-" * 20)
    for level, count in grade_distribution.items():
        percentage = count / len(df) * 100
        print(f"{level}: {count}人 ({percentage:.1f}%)")
    print()
    
    return df

def visualize_data(df):
    """可视化学生数据"""
    # 1. 各科目成绩分布直方图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    subjects = ['math_score', 'english_score', 'physics_score']
    subject_names = ['数学', '英语', '物理']
    
    for i, (subject, name) in enumerate(zip(subjects, subject_names)):
        axes[i].hist(df[subject], bins=20, alpha=0.7, color=plt.cm.Set3(i))
        axes[i].set_xlabel('分数')
        axes[i].set_ylabel('人数')
        axes[i].set_title(f'{name}成绩分布')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/score_distributions.png')
    plt.close()
    print("已生成各科目成绩分布图 (score_distributions.png)")
    
    # 2. 成绩等级饼图
    grade_counts = df['grade_level'].value_counts()
    plt.figure(figsize=(8, 8))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    plt.pie(grade_counts.values, labels=grade_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('成绩等级分布')
    plt.savefig(f'{output_dir}/grade_distribution.png')
    plt.close()
    print("已生成成绩等级分布图 (grade_distribution.png)")
    
    # 3. 各科目平均分对比柱状图
    subject_means = [df[subj].mean() for subj in subjects]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(subject_names, subject_means, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('平均分')
    plt.title('各科目平均分对比')
    
    # 在柱状图上添加数值标签
    for bar, mean in zip(bars, subject_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{mean:.1f}', ha='center', va='bottom')
    
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f'{output_dir}/subject_comparison.png')
    plt.close()
    print("已生成各科目平均分对比图 (subject_comparison.png)")
    
    # 4. 性别与成绩关系箱线图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (subject, name) in enumerate(zip(subjects, subject_names)):
        df.boxplot(column=subject, by='gender', ax=axes[i])
        axes[i].set_title(f'{name}成绩按性别分布')
        axes[i].set_xlabel('性别')
        axes[i].set_ylabel('分数')
    
    plt.suptitle('')  # 移除默认标题
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gender_score_comparison.png')
    plt.close()
    print("已生成性别与成绩关系图 (gender_score_comparison.png)")
    
    # 5. 总分与各科成绩散点图矩阵
    plt.figure(figsize=(10, 8))
    pd.plotting.scatter_matrix(df[subjects + ['total_score']], 
                              alpha=0.6, figsize=(10, 8), diagonal='hist')
    plt.suptitle('成绩散点图矩阵')
    plt.savefig(f'{output_dir}/scatter_matrix.png')
    plt.close()
    print("已生成成绩散点图矩阵 (scatter_matrix.png)")
    
    print(f"\n所有图表已保存到 {output_dir} 目录下!")

def save_results(df):
    """保存分析结果"""
    # 保存处理后的数据
    df.to_csv(f'{output_dir}/student_grades_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"分析结果已保存到 {output_dir}/student_grades_analysis.csv")
    
    # 输出前10名学生
    top_students = df.nlargest(10, 'total_score')[['name', 'math_score', 'english_score', 
                                                   'physics_score', 'total_score', 'average_score']]
    print("\n前10名学生:")
    print("-" * 60)
    print(top_students.to_string(index=False))

def main():
    """主函数"""
    print("开始学生 grades 分析项目...")
    
    # 生成示例数据
    print("正在生成示例数据...")
    df = generate_sample_data()
    
    # 分析数据
    print("正在进行数据分析...")
    df = analyze_data(df)
    
    # 可视化数据
    print("正在生成可视化图表...")
    visualize_data(df)
    
    # 保存结果
    print("正在保存分析结果...")
    save_results(df)
    
    print("\n项目完成! 请查看 output 目录下的结果文件。")

if __name__ == "__main__":
    main()