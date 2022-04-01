# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 23:44
# @Author  : Justus
# @FileName: GoogLeNet.py
# @Software: PyCharm

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import torch.nn.functional


# 准备数据集
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307, ), (0.3081, ))
                                ])
train_dataset = datasets.MNIST(root="./mnist", train=True,
                               transform=transform,
                               download=True)
test_dataset = datasets.MNIST(root="./mnist",
                              train=False,
                              transform=transform,
                              download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# 定义模型
# 复用块
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 分支1
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))
        # 分支2
        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=(5, 5), padding=2)
        # 分支3
        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=(3, 3), padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=(3, 3), padding=1)
        # 池化
        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=(1, 1))

    def forward(self, x):
        # 分支1
        branch1x1 = self.branch1x1(x)
        # 分支2
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        # 分支3
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        # 池化
        branch_pool = torch.nn.functional.avg_pool2d(x, kernel_size=(3, 3), stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        # 输入维度N，C，H，W，按C维度方向进行拼接dim=1
        return torch.cat(outputs, dim=1)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=(5, 5))

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        in_size = x.size(0)
        x = self.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = self.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


model = Net()

# GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 交叉熵损失已包含
criterion = torch.nn.CrossEntropyLoss()
# SGD优化器, momentum冲量值
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# 训练
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 加载数据
        inputs, target = data
        # inputs和target迁移到GPU，注意要在同一块显卡上
        inputs, target = inputs.to(device), target.to(device)
        # 预测
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, target)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("[%d, %4d]loss:%.3f" % (epoch+1, batch_idx+1, running_loss/300))


# 测试
def test():
    correct = 0
    total = 0
    # 不计算梯度，强制之后的内容不进行计算图构建
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # inputs和target迁移到GPU，注意要在同一块显卡上
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 取每行最大值的下标，dim=1按行
            _, predicted = torch.max(outputs.data, dim=1)
            # 取labels的第0个元素，total最终值为样本总数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy on test set:%d %%" % (100*correct/total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
