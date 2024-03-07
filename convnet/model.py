import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 20)
        self.linear2 = nn.Linear(20, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 28 * 28)
        x = F.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x)


class MNIST_2Layer(nn.Module):
    def __init__(self):
        super(MNIST_2Layer, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 20)
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(20, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 28 * 28)
        x = F.sigmoid(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return F.log_softmax(x)


class MNIST_Big(nn.Module):
    def __init__(self):
        super(MNIST_Big, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 40)
        self.linear2 = nn.Linear(40, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 28 * 28)
        x = F.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x)


class MNIST_Relu(nn.Module):
    def __init__(self):
        super(MNIST_Relu, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 20)
        self.linear2 = nn.Linear(20, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 28 * 28)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 8 * 8, 1024)

        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.bn1(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.bn2(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.sigmoid(out)

        out = self.fc2(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 看上面的信息是否需要卷积修改，从而满足相加条件
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits
        # probas = F.softmax(logits, dim=1)
        # return logits, probas


def resnet18(num_classes):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   num_classes=num_classes,
                   grayscale=False)
    return model

def resnet50(num_classes):
    model = ResNet(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=False)
    return model
