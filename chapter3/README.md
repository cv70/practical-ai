# 第 3 章：深度学习入门

## 📚 学习目标

完成本章后，你将能够：
- 理解神经网络的基本原理和结构
- 使用 PyTorch 构建和训练神经网络
- 掌握深度学习中的关键概念和技术
- 解决实际的深度学习问题

## 📖 章节内容

### 1. 神经网络基础
- 神经元和激活函数
- 前向传播和反向传播
- 损失函数和优化器
- 梯度下降算法

### 2. PyTorch 框架介绍
- 张量(Tensor)操作
- 自动微分机制
- 模型定义和训练流程
- GPU 加速计算

### 3. 构建第一个神经网络
- 数据准备和预处理
- 网络架构设计
- 训练和验证过程
- 结果分析和可视化

### 4. 训练技巧和最佳实践
- 过拟合和欠拟合
- 正则化技术
- 学习率调度
- 批量归一化

## 💻 代码示例

### [PyTorch张量操作示例 code/pytorch_tensors.py](./code/pytorch_tensors.py)
- 张量创建和基本属性
- 张量运算
- 自动微分机制
- GPU加速计算

### [神经网络基础示例 code/neural_network_basics.py](./code/neural_network_basics.py)
- 神经元和激活函数
- 前向传播和反向传播
- 损失函数和优化器
- 梯度下降算法

### [汽车MPG预测项目 code/car_mpg_predict_project.py](./code/car_mpg_predict_project.py)
- 数据集加载和预处理
- 神经网络模型构建
- 模型训练和验证
- 结果分析和可视化
- PyTorch模型定义和训练流程
- 使用神经网络实现回归任务

### [鸢尾花分析项目 code/iris_analysis_project.py](./code/iris_analysis_project.py)
- 数据集加载和预处理
- 神经网络模型构建
- 模型训练和验证
- 结果分析和可视化
- PyTorch模型定义和训练流程
- 使用神经网络实现分类任务与聚类任务

## 🧪 实践练习

### [手写数字识别项目(code/mnist_fully_connected.py)](./code/mnist_fully_connected.py)

这是一个使用全连接神经网络实现的手写数字识别项目，帮助你深入理解神经网络的基本原理：

1. **数据处理**：加载MNIST手写数字数据集，将28x28的图像展平为784维向量
2. **模型构建**：使用PyTorch构建全连接神经网络，包含输入层、隐藏层和输出层
3. **模型训练**：实现完整的训练流程，包括交叉熵损失函数和Adam优化器
4. **模型评估**：在测试集上评估模型性能，计算准确率
5. **结果分析**：分析模型在不同数字上的表现
6. **可视化**：生成多种图表展示训练过程和预测结果

运行项目：
```bash
cd code && python mnist_fully_connected.py
```

项目输出：
- 控制台显示详细的训练过程和结果
- `output/` 目录下生成多个可视化图表：
  - `mnist_samples_fc.png`：MNIST数据集样本
  - `mnist_fc_training_curves.png`：训练和测试损失及准确率曲线
  - `mnist_fc_predictions.png`：预测结果示例

## 🔍 扩展阅读

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- 《动手学深度学习》 - 李沐等著

---
**[上一章：机器学习基础](../chapter2/README.md)** | **[下一章：计算机视觉](../chapter4/README.md)**