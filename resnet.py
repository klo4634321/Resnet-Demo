import torch

import torch.nn as nn
import torch.nn.functional as F


# 定義一個基本的殘差塊
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一個卷積層
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批量正規化
        # 第二個卷積層
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # 批量正規化
        self.downsample = downsample  # 下採樣，用於匹配維度
        self.relu = nn.ReLU(inplace=True)  # 激活函數

    def forward(self, x):
        identity = x  # 保存輸入值
        if self.downsample is not None:
            identity = self.downsample(x)  # 如果需要下採樣，調整維度

        out = self.conv1(x)  # 第一個卷積
        out = self.bn1(out)  # 批量正規化
        out = self.relu(out)  # 激活函數

        out = self.conv2(out)  # 第二個卷積
        out = self.bn2(out)  # 批量正規化

        out += identity  # 殘差連接
        out = self.relu(out)  # 激活函數

        return out

# 定義ResNet模型
class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # 初始卷積層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # 批量正規化
        self.relu = nn.ReLU(inplace=True)  # 激活函數
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化層

        # 堆疊多個殘差塊
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化層和全連接層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # 如果輸入和輸出的維度不匹配，進行下採樣
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 第一個殘差塊
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        # 後續的殘差塊
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 初始卷積
        x = self.bn1(x)  # 批量正規化
        x = self.relu(x)  # 激活函數
        x = self.maxpool(x)  # 最大池化

        x = self.layer1(x)  # 第一層殘差塊
        x = self.layer2(x)  # 第二層殘差塊
        x = self.layer3(x)  # 第三層殘差塊
        x = self.layer4(x)  # 第四層殘差塊

        x = self.avgpool(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 展平
        x = self.fc(x)  # 全連接層

        return x

# 定義ResNet18
def resnet18(num_classes=1000):
    return ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# 測試模型
if __name__ == "__main__":
    model = resnet18(num_classes=10)  # 定義ResNet18模型，分類數為10
    print(model)