#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用全连接神经网络实现手写数字识别(MNIST)
=====================================

本文件演示了如何使用PyTorch构建和训练一个全连接神经网络来识别手写数字。
使用全连接网络将图像展平为一维向量进行处理。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Maple Mono NF CN']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 输出目录
output_dir = './output'
# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"=== 手写数字识别 (全连接网络) (设备: {device}) ===\n")

# 1. 数据准备和预处理
print("1. 数据准备和预处理")
print("-" * 30)

# 定义数据转换 - 只将图像转换为张量并展平
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # 展平为784维向量
])

# 下载和加载训练数据集
train_dataset = torchvision.datasets.MNIST(
    root='./datasets',
    train=True,
    download=False, # 安装后就可以置为False
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

# 下载和加载测试数据集
test_dataset = torchvision.datasets.MNIST(
    root='./datasets',
    train=False,
    download=False, # 安装后就可以置为False
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=1000,
    shuffle=False
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"输入特征维度: 28*28 = {28*28}")
print(f"类别数量: 10 (0-9)")

# 可视化一些样本
def imshow(img):
    img = img.view(28, 28)  # 重塑为28x28图像
    plt.imshow(img, cmap='gray')
    plt.show()

# 获取一批随机训练图像
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 显示图像
plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    img = images[i].view(28, 28)  # 重塑为28x28图像
    plt.imshow(img, cmap='gray')
    plt.title(f'标签: {labels[i].item()}')
    plt.axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/mnist_samples_fc.png')
plt.close()
print("已生成MNIST样本图 (mnist_samples_fc.png)")

# 2. 定义全连接神经网络
print("\n2. 定义全连接神经网络")
print("-" * 30)

class FullyConnectedNet(nn.Module):
    def __init__(self, input_size=784, hidden_size1=512, hidden_size2=256, num_classes=10):
        super(FullyConnectedNet, self).__init__()
        # 全连接层
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        # Dropout层
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 第一个全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # 第二个全连接层
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # 输出层
        x = self.fc3(x)
        return x

# 初始化网络
model = FullyConnectedNet().to(device)
print("网络结构:")
print(model)

# 3. 定义损失函数和优化器
print("\n3. 定义损失函数和优化器")
print("-" * 30)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("损失函数: CrossEntropyLoss")
print("优化器: Adam (学习率: 0.001)")

# 4. 训练模型
print("\n4. 训练模型")
print("-" * 30)

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    accuracy = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    print(f'Epoch {epoch} - 平均损失: {avg_loss:.4f}, 准确率: {correct}/{total} ({accuracy:.2f}%)')
    return avg_loss, accuracy

# 5. 测试模型
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f'测试集 - 平均损失: {avg_loss:.4f}, 准确率: {correct}/{total} ({accuracy:.2f}%)')
    return avg_loss, accuracy

# 训练多个epoch
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

epochs = 10
for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test()
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

# 6. 可视化训练过程
print("\n6. 可视化训练过程")
print("-" * 30)

plt.figure(figsize=(15, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='训练损失', marker='o')
plt.plot(range(1, epochs + 1), test_losses, label='测试损失', marker='s')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.title('训练和测试损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='训练准确率', marker='o')
plt.plot(range(1, epochs + 1), test_accuracies, label='测试准确率', marker='s')
plt.xlabel('Epoch')
plt.ylabel('准确率 (%)')
plt.title('训练和测试准确率曲线')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/mnist_fc_training_curves.png')
plt.close()
print("已生成训练曲线图 (mnist_fc_training_curves.png)")

# 7. 预测示例
print("\n8. 预测示例")
print("-" * 30)

# 获取测试集的一批数据
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# 预测
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 显示前12个图像及其预测结果
plt.figure(figsize=(15, 8))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    img = images[i].view(28, 28).cpu()  # 重塑为28x28图像
    plt.imshow(img, cmap='gray')
    plt.title(f'真实: {labels[i].item()}\n预测: {predicted[i].item()}')
    plt.axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/mnist_fc_predictions.png')
plt.close()
print("已生成预测结果图 (mnist_fc_predictions.png)")

print("\n" + "="*50)
print("项目总结")
print("="*50)
print("1. 数据处理:")
print("   - 使用MNIST数据集进行手写数字识别")
print("   - 将28x28的图像展平为784维向量")
print()
print("2. 模型架构:")
print("   - 构建了一个包含3个全连接层的神经网络")
print("   - 使用ReLU激活函数和Dropout防止过拟合")
print()
print("3. 训练过程:")
print(f"   - 训练了{epochs}个epoch")
print(f"   - 最终测试准确率: {test_accuracies[-1]:.2f}%")
print()
print("4. 结果分析:")
print("   - 通过损失曲线和准确率曲线监控训练过程")
print("   - 使用混淆矩阵分析模型在各类别上的表现")
print()
print("所有结果图表已保存到 output 目录中!")

print("\n程序执行完毕！")