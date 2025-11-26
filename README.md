# 实用人工智能：知行合一

欢迎来到实用人工智能教程！本项目旨在通过动手实践帮助初学者理解和掌握人工智能的核心概念和技术。

## 📚 教程简介

本教程采用循序渐进的方式，从基础概念出发，逐步深入到实际应用。每一章节都包含理论讲解和代码实践，帮助读者更好地理解人工智能的工作原理。

## 🎯 学习目标

完成本教程后，你将能够：
- 理解人工智能的基本概念和发展历程
- 掌握机器学习的基础知识和常用算法
- 使用 Python 和相关库实现 AI 应用
- 构建和训练神经网络模型
- 解决实际的人工智能问题

## 📂 项目结构

```
practical-ai/
├── chapter0/           # 人工智能概述
├── chapter1/           # Python 基础与数据处理
├── chapter2/           # 机器学习基础
├── chapter3/           # 深度学习入门
├── chapter4/           # 计算机视觉
├── chapter5/           # 自然语言处理
├── chapter6/           # Transformer架构
├── README.md           # 教程说明
```

## 🚀 环境配置

### 克隆项目

```bash
git clone https://github.com/cv70/practical-ai.git
cd practical-ai
```

### uv安装
[参考文档：https://uv.doczh.com/getting-started/installation/](https://uv.doczh.com/getting-started/installation/)

### 创建虚拟环境

```bash
# 安装 Python 3.12
uv python install 3.12

# 使用 uv 创建 Python 3.12 虚拟环境
uv venv --python 3.12

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows
```

### 安装依赖

- 安装PyTorch

[参考文档：https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

- 安装NumPy
```bash
# 正常安装PyTorch后，NumPy会自动安装，这一步也能跳过
uv pip install numpy
```

- 安装Pandas
```bash
uv pip install pandas
```

- 安装Matplotlib
```bash
uv pip install matplotlib
```

## 📖 教程章节

- [**Chapter 0**: 人工智能概述](./chapter0/README.md)
  - 什么是人工智能？
  - 人工智能的发展历史
  - 人工智能的应用领域
  - Python环境准备

- [**Chapter 1**: Python 基础与数据处理](./chapter1/README.md)
  - Python 编程基础
  - NumPy 科学计算
  - Pandas 数据分析
  - Matplotlib 数据可视化

- [**Chapter 2**: 机器学习基础](./chapter2/README.md)
  - 机器学习概念
  - 监督学习-回归任务
  - 监督学习-分类任务
  - 无监督学习-聚类任务
  - 模型评估与优化

- [**Chapter 3**: 深度学习入门](./chapter3/README.md)
  - 神经网络基础
  - 全连接神经网络 (FCN)
  - PyTorch 框架介绍
  - 构建神经网络
  - 训练与调试技巧

- [**Chapter 4**: 计算机视觉](./chapter4/README.md)
  - 计算机视觉概述
  - 图像处理基础
  - 卷积神经网络 (CNN)
  - 图像分类任务
  - 目标检测任务

- [**Chapter 5**: 自然语言处理](./chapter5/README.md)
  - 自然语言处理概述
  - 文本预处理
  - 词向量表示
  - 循环神经网络 (RNN)
  - 文本生成任务
  - 文本分类任务

- [**Chapter 6**: Transformer架构](./chapter6/README.md)
  - Transformer 架构
  - 注意力机制
  - 位置编码
  - BERT
  - 大语言模型概述
  - 计算文本相似度
  - 动手实现大语言模型

## 🧪 实践项目

每个章节都配有相应的编程练习和小项目，帮助巩固所学知识。

## 📝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进本教程！

## 📄 许可证

本项目采用 Apache-2.0 许可证，详情请见 [LICENSE](./LICENSE-APACHE) 文件。
