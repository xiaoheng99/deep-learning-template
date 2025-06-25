import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.short_cut = nn.Sequential()

        if stride != 1 or in_planes != planes * self.expansion:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, stride=stride, bias=False),  # in_planes-->planes
                nn.BatchNorm2d(planes*self.expansion)
                )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))  # out
        out += self.short_cut(x)
        out = F.relu(out)
        return out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.short_cut = nn.Sequential()

        if stride != 1 or in_planes != planes * self.expansion:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*self.expansion)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.short_cut(x)
        out = F.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, in_planes: int, image_channels: int, block, num_blocks, num_classes: int = 10):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(image_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def ResNet18():
    return ResNet(64, 3, BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(64, 3, BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(64, 3, Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(64, 3, Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(64, 3, Bottleneck, [3, 8, 36, 3])

def count_parameters(model):
    return sum([param.numel() for name, param in model.named_parameters() if param.requires_grad])


# 计算模型参数大小和显存占用
def check_memory_usage(model, input_shape, batch_size):
    # 计算模型参数大小 (MB)
    param_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)
    print(f"Batch Size: {batch_size}")
    print(f"Parameter Size (MB): {param_size:.2f}")

    # 计算显存占用
    if torch.cuda.is_available():
        model.cuda()
        input_data = torch.randn(batch_size, *input_shape).cuda()

        # 清空缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        # 前向传播
        output = model(input_data)

        # 获取显存占用 (MB)
        memory_used = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"Memory Used (MB): {memory_used:.2f}")
    else:
        print("CUDA is not available.")

def main():
    net = ResNet18().to(device)
    print(f"Number of parameters: {count_parameters(net) / 1e6}M")
    # y = net(torch.randn(64, 3, 224, 224).to(device))
    # print(y.size())
    # summary(net, (3, 224, 224))
    x = torch.randn(1, 3, 224, 224).to(device)
    # print(net(x).shape)

    input_shape = (3, 224, 224)  # 输入数据的形状

    # 尝试不同的 batch size
    batch_sizes = [1, 8, 16, 32, 64]
    for batch_size in batch_sizes:
        check_memory_usage(net, input_shape, batch_size)
        print("-" * 50)

    # 得到的数据：
    # Number of parameters: 11.173962M 可训练的参数量
    # Params size (MB): 42.63M 参数量大小  参数量×4
    # Estimated Total Size (MB): 686.33  模型在前向传播和反向传播过程中临时变量占用的内存量。
   

if __name__ == '__main__':
    main()


