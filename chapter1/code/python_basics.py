#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python 基础知识示例
==================

演示 Python 编程的基础知识，包括：
- 变量与数据类型
- 控制结构（条件语句、循环）
- 函数定义与调用
- 类与对象
"""

# 1. 变量与数据类型
print("=== 1. 变量与数据类型 ===")

# 数值类型
integer_var = 42
float_var = 3.14159
complex_var = 3 + 4j

print(f"整数: {integer_var} (类型: {type(integer_var)})")
print(f"浮点数: {float_var} (类型: {type(float_var)})")
print(f"复数: {complex_var} (类型: {type(complex_var)})")

# 字符串类型
string_var = "Hello, AI!"
print(f"字符串: {string_var} (类型: {type(string_var)})")

# 布尔类型
bool_var = True
print(f"布尔值: {bool_var} (类型: {type(bool_var)})")

# 列表
list_var = [1, 2, 3, 4, 5]
print(f"列表: {list_var} (类型: {type(list_var)})")

# 元组
tuple_var = (1, 2, 3, 4, 5)
print(f"元组: {tuple_var} (类型: {type(tuple_var)})")

# 字典
dict_var = {"name": "AI", "version": 1.0}
print(f"字典: {dict_var} (类型: {type(dict_var)})")

# 集合
set_var = {1, 2, 3, 4, 5}
print(f"集合: {set_var} (类型: {type(set_var)})")

print("\n" + "="*50 + "\n")

# 2. 控制结构
print("=== 2. 控制结构 ===")

# 条件语句
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "D"

print(f"分数 {score} 对应等级: {grade}")

# 循环语句
print("\n使用 for 循环打印 1 到 5:")
for i in range(1, 6):
    print(f"  {i}")

print("\n使用 while 循环打印倒计时:")
count = 5
while count > 0:
    print(f"  {count}")
    count -= 1

print("\n" + "="*50 + "\n")

# 3. 函数定义与调用
print("=== 3. 函数定义与调用 ===")

def greet(name, greeting="Hello"):
    """简单的问候函数"""
    return f"{greeting}, {name}!"

def calculate_area(length, width):
    """计算矩形面积"""
    return length * width

def factorial(n):
    """递归计算阶乘"""
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)

# 调用函数
print(greet("AI 学习者"))
print(greet("Python", "你好"))
print(f"矩形面积 (5x3): {calculate_area(5, 3)}")
print(f"5 的阶乘: {factorial(5)}")

print("\n" + "="*50 + "\n")

# 4. 类与对象
print("=== 4. 类与对象 ===")

class Student:
    """学生类示例"""
    
    def __init__(self, name, age, major):
        """初始化学生对象"""
        self.name = name
        self.age = age
        self.major = major
        self.grades = []
    
    def add_grade(self, grade):
        """添加成绩"""
        self.grades.append(grade)
    
    def get_average_grade(self):
        """计算平均成绩"""
        if not self.grades:
            return 0
        return sum(self.grades) / len(self.grades)
    
    def introduce(self):
        """自我介绍"""
        return f"我是 {self.name}，{self.age} 岁，专业是 {self.major}"

# 创建学生对象
student1 = Student("张三", 20, "计算机科学")
student1.add_grade(85)
student1.add_grade(92)
student1.add_grade(78)

print(student1.introduce())
print(f"平均成绩: {student1.get_average_grade():.2f}")

print("\n程序执行完毕！")