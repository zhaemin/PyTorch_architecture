import torch
import torch.nn as nn
import torch.nn.functional as F

import densenet
import fractalnet
import visiontransformer
import mlpmixer

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_classes)
        
    def forward(self,x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock,self).__init__()
        
        if in_features == out_features:
            stride = 1
            self.identity = nn.Identity()
        else:
            stride = 2
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_features)
                )
            
        self.conv1 = nn.Conv2d(in_features, out_features, 3, stride=stride, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_features, out_features, 3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_features)
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.batchnorm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.batchnorm2(y)
        y += self.identity(x)
        y = self.relu(y)
        
        return y

class PreactResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(PreactResidualBlock,self).__init__()
        
        if in_features == out_features:
            stride = 1
            self.identity = nn.Identity()
        else:
            stride = 2
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_features)
                )
        
        self.batchnorm1 = nn.BatchNorm2d(in_features)
        self.conv1 = nn.Conv2d(in_features, out_features, 3, stride=stride, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        y = self.batchnorm1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.batchnorm2(y)
        y = self.relu(y)
        y = self.conv2(y)
        y += self.identity(x)
        
        return y

class BottleneckBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(BottleneckBlock,self).__init__()
        
        if in_features == out_features*4:
            stride = 1
            self.identity = nn.Identity()
        elif in_features == out_features:
            stride = 1
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features*4, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_features*4)
                )
        else:
            stride = 2
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features*4, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_features*4)
                )
        
        self.conv1 = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_features)
        self.conv3 = nn.Conv2d(out_features, out_features*4, 1, stride=1)
        self.batchnorm3 = nn.BatchNorm2d(out_features*4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.batchnorm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.batchnorm2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.batchnorm3(y)
        y += self.identity(x)
        y = self.relu(y)
        
        return y

class PreactBottleneckBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(PreactBottleneckBlock,self).__init__()
        
        if in_features == out_features*4:
            stride = 1
            self.identity = nn.Identity()
        elif in_features == out_features:
            stride = 1
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features*4, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_features*4)
                )
        else:
            stride = 2
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features*4, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_features*4)
                )

        self.batchnorm1 = nn.BatchNorm2d(in_features)
        self.conv1 = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, stride=1, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_features)
        self.conv3 = nn.Conv2d(out_features, out_features*4, 1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.batchnorm1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.batchnorm2(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.batchnorm3(y)
        y = self.relu(y)
        y = self.conv3(y)
        y += self.identity(x)
        
        return y

class PreactResNet(nn.Module):
    def __init__(self, layer_num, num_classes):
        super(PreactResNet, self).__init__()
        
        self.block_type = 'Residual'
        
        if layer_num == 18:
            layers = [2,2,2,2]
        elif layer_num == 34:
            layers = [3,4,6,3]
        elif layer_num == 50:
            layers = [3,4,6,3]
            self.block_type = 'Bottleneck'
        elif layer_num == 101:
            layers = [3,4,23,3]
            self.block_type = 'Bottleneck'
        elif layer_num == 152:
            layers = [3,8,36,3]
            self.block_type = 'Bottleneck'
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        self.conv2_x = self.build_blocks(layers[0], 64, 64)
        self.conv3_x = self.build_blocks(layers[1], 64, 128)
        self.conv4_x = self.build_blocks(layers[2], 128, 256)
        self.conv5_x = self.build_blocks(layers[3], 256, 512)
        self.avgpool = nn.AvgPool2d(4)
        self.relu = nn.ReLU()
        if self.block_type == 'Bottleneck':
            self.fc = nn.Linear(2048, num_classes)
        else:
            self.fc = nn.Linear(512, num_classes)
            
    def build_blocks(self, layer, in_features, out_features):
        module_list = []
        if self.block_type == 'Residual':
            for i in range(layer):
                if i == 0:
                    module_list.append(PreactResidualBlock(in_features, out_features))
                else:
                    module_list.append(PreactResidualBlock(out_features, out_features))
        elif self.block_type == 'Bottleneck':
            for i in range(layer):
                if i == 0:
                    module_list.append(PreactBottleneckBlock(in_features, out_features))
                else:
                    module_list.append(PreactBottleneckBlock(out_features*4, out_features))
        return nn.Sequential(*module_list)
            
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def set_net(model, num_classes):
    if model == 'LeNet':
        net = LeNet(num_classes)
    elif model == 'ResNet-18':
        net = PreactResNet(18, num_classes)
    elif model == 'ResNet-34':
        net = PreactResNet(34, num_classes)
    elif model == 'ResNet-50':
        net = PreactResNet(50, num_classes)
    elif model == 'ResNet-101':
        net = PreactResNet(101, num_classes)
    elif model == 'ResNet-152':
        net = PreactResNet(152, num_classes)
    elif model == 'DenseNet_BC-121':
        net = densenet.DenseNetBC(121, num_classes=num_classes)
    elif model == 'DenseNet_BC-169':
        net = densenet.DenseNetBC(169, num_classes=num_classes)
    elif model == 'DenseNet_BC-201':
        net = densenet.DenseNetBC(201, num_classes=num_classes)
    elif model == 'DenseNet_BC-264':
        net = densenet.DenseNetBC(264, num_classes=num_classes)
    elif model == 'FractalNet-20':
        net = fractalnet.FractalNet(20, num_classes=num_classes)
    elif model == 'FractalNet-40':
        net = fractalnet.FractalNet(40, num_classes=num_classes)
    elif model == 'ViT-B':
        net = visiontransformer.VisionTransformer('Base', num_classes)
    elif model == 'ViT-L':
        net = visiontransformer.VisionTransformer('Large', num_classes)
    elif model == 'ViT-H':
        net = visiontransformer.VisionTransformer('Huge', num_classes)
    elif model == 'MLPMixer-S':
        net = mlpmixer.MLPMixer('S', 4, num_classes)
    elif model == 'MLPMixer-B':
        net = mlpmixer.MLPMixer('B', 4, num_classes)
    elif model == 'MLPMixer-L':
        net = mlpmixer.MLPMixer('L', 4, num_classes)
    return net