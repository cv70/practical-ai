# 第 0 章：人工智能概述与Python环境准备

## 📚 学习目标

完成本章后，你将能够：
- 理解人工智能的基本定义和概念
- 了解人工智能的发展历史和重要里程碑
- 认识人工智能在各个领域的应用
- 区分不同类型的人工智能系统
- 准备Python开发环境
- 运行基本的Python代码示例

## 📖 章节内容

### 1. 什么是人工智能？

人工智能（Artificial Intelligence，简称AI）是指由人类制造出来的机器所表现出来的智能。它能够通过感知环境、理解语言、学习经验并采取行动来实现特定目标。

### 2. 人工智能的发展简史

- **1950年代**：图灵测试提出，标志着人工智能概念的诞生
- **1956年**：达特茅斯会议正式确立"人工智能"这一术语
- **1960-70年代**：专家系统的兴起
- **1980-90年代**：机器学习方法的发展
- **2000年代至今**：深度学习和大数据推动AI快速发展

### 3. 人工智能的主要分支

- **机器学习**：让计算机通过数据自动学习模式
- **自然语言处理**：使计算机理解和生成人类语言
- **计算机视觉**：让计算机"看懂"图像和视频
- **机器人学**：结合感知、规划和控制技术
- **专家系统**：模拟人类专家决策过程

### 4. 人工智能的应用领域

- **医疗健康**：疾病诊断、药物研发
- **金融服务**：风险评估、算法交易
- **交通运输**：自动驾驶、路径规划
- **教育**：个性化学习、智能辅导
- **娱乐**：推荐系统、游戏AI

### 5. Python环境准备

为了进行人工智能开发，我们需要准备以下工具：
- Python环境管理（uv）

[参考链接](https://uv.doczh.com/getting-started/installation/)

- Python 3.12
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

- 深度学习框架（PyTorch）

[参考文档：https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

- 核心科学计算库（NumPy, Pandas, Matplotlib）
```bash
uv pip install numpy pandas matplotlib
```

### 6. Python基础示例

查看 `code/` 目录中的示例代码，了解Python在AI开发中的基本用法。

## 💻 代码示例

### [Hello World示例 code/python_hello_world.py](./code/python_hello_world.py)
- 最简单的Python程序
- 演示基本的输出语句
- 变量和数据类型
- 简单的数学运算
- 字符串操作

## 🧪 实践练习

1. 调研一个你感兴趣的人工智能应用案例
2. 分析该应用解决了什么问题，使用了哪些AI技术
3. 思考该应用可能面临的挑战和限制
4. 安装Python环境并运行本章的示例代码
5. 修改示例代码，添加自己的功能

## 🔍 扩展阅读

- 《人工智能：一种现代的方法》 - Stuart Russell & Peter Norvig
- 《深度学习》 - Ian Goodfellow, Yoshua Bengio & Aaron Courville
- [AI发展趋势报告](https://www.example.com/ai-trends)

---
**[下一章：Python 基础与数据处理](../chapter1/README.md)**
