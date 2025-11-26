#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
人工智能概述与Python基础
================================

演示 Python 基础知识，为后续的人工智能学习打下基础。

内容包括：
- 基本输出语句
- 变量和数据类型
- 简单的数学运算
- 字符串操作
"""

# 1. 基本输出
print("=" * 50)
print("欢迎来到人工智能学习之旅！")
print("=" * 50)

# 2. 变量和数据类型
name = "AI学习者"
age = 25
height = 1.75
is_student = True

print(f"姓名: {name}")
print(f"年龄: {age}")
print(f"身高: {height}米")
print(f"是否为学生: {is_student}")

# 3. 基本数学运算
print("\n基本数学运算:")
a, b = 10, 3
print(f"{a} + {b} = {a + b}")
print(f"{a} - {b} = {a - b}")
print(f"{a} * {b} = {a * b}")
print(f"{a} / {b} = {a / b:.2f}")
print(f"{a} // {b} = {a // b}")
print(f"{a} % {b} = {a % b}")
print(f"{a} ** {b} = {a ** b}")

# 4. 字符串操作
print("\n字符串操作:")
message = "人工智能"
print(f"字符串长度: {len(message)}")
print(f"字符串重复: {message * 3}")
print(f"字符串切片: {message[0:2]}")

# 5. 列表操作
print("\n列表操作:")
ai_topics = ["机器学习", "深度学习", "自然语言处理", "计算机视觉"]
print(f"AI主题列表: {ai_topics}")
print(f"第一个主题: {ai_topics[0]}")
ai_topics.append("机器人学")
print(f"添加后的列表: {ai_topics}")

# 6. 简单函数
def greet_person(name):
    """问候函数"""
    return f"你好, {name}! 欢迎学习人工智能!"

print("\n函数调用:")
greeting = greet_person("AI学习者")
print(greeting)

# 7. 条件语句
print("\n条件语句:")
if age >= 18:
    print("你已经成年，可以开始学习人工智能了！")
else:
    print("你还年轻，是学习的好时机！")

# 8. 循环语句
print("\n循环语句:")
print("AI学习的步骤:")
for i, step in enumerate(["理论学习", "实践编程", "项目开发", "持续优化"], 1):
    print(f"{i}. {step}")

print("\n学习进度:")
progress = 0
while progress < 5:
    print(f"学习进度: {progress * 20}%")
    progress += 1
print("学习进度: 100% - 完成!")

print("\n" + "=" * 50)
print("Python基础知识演示完成！")
print("现在你已经准备好开始深入学习人工智能了！")
print("=" * 50)