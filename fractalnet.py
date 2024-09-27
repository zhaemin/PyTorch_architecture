import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

class ConvBlock(nn.Sequential):
    def __init__(self, in_features, out_features, dropout_rate):
        super(ConvBlock,self).__init__(
            nn.Conv2d(in_features, out_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

class TransitionLayer(nn.Sequential):
    def __init__(self, in_features, out_features):
        super(TransitionLayer,self).__init__(
            nn.Conv2d(in_features, out_features, 1, bias=False),
            nn.MaxPool2d(2, 2)
        )

mode = 'test'
global_drop_column = -1

class FractalBlock(nn.Module):
    def __init__(self, original_columns, num_columns, num_features, out_features, dropout_rate):
        super(FractalBlock,self).__init__()
        
        self.original_columns = original_columns
        self.num_columns = num_columns
        
        if self.num_columns == 1:
            self.conv = ConvBlock(num_features,num_features,dropout_rate)
        else:
            self.conv = ConvBlock(num_features,num_features,dropout_rate)
            self.block1 = FractalBlock(original_columns, num_columns-1, num_features, out_features, dropout_rate)
            self.block2 = FractalBlock(original_columns, num_columns-1, num_features, out_features, dropout_rate)
            self.transition = TransitionLayer(num_features,out_features)
    
    def forward(self,x):
        global mode
        global global_drop_column
        
        if self.original_columns == self.num_columns:
            if self.training:
                if random.random() > 0.5:
                    mode = 'drop_local'
                    global_drop_column = -1
                else:
                    mode = 'drop_global'
                    global_drop_column = random.randint(0, self.original_columns-1)

        if self.num_columns == 1:
            y = self.conv(x)
            return y
        else:
            x = join(x)
            y1 = self.conv(x)
            y2 = self.block1(x)
            y2 = join(y2)
            y2 = self.block2(y2)
            
            if isinstance(y2, list):
                res = [y1]+y2
            else:
                res = [y1,y2]
            
            if self.original_columns == self.num_columns:
                return self.transition(join(res))
            else:
                return res

def join(inputs):
    if not isinstance(inputs, list):
        return inputs
    
    if mode == 'drop_local':
        mask = np.random.rand(len(inputs)) > 0.15
        if not any(mask):
            mask[random.randint(0,len(inputs)-1)] = True
        inputs = [i for i,m in zip(inputs,mask) if m]
    
    if mode == 'drop_global' and len(inputs)-1 >= global_drop_column:
        return inputs[global_drop_column]
    
    total = 0
    for tensors in inputs:
        total += tensors
    return total / len(inputs)

class FractalNet(nn.Module):
    def __init__(self, layer_num, num_classes):
        super(FractalNet,self).__init__()
        
        dropout_per_block = [0, 0.1, 0.2, 0.3, 0.4]
        num_columns = int(math.log2(layer_num//5))+1
        
        self.conv = ConvBlock(3, 64, 0)
        self.block1 = FractalBlock(num_columns, num_columns, 64, 128, dropout_per_block[0])
        self.block2 = FractalBlock(num_columns, num_columns, 128, 256, dropout_per_block[1])
        self.block3 = FractalBlock(num_columns, num_columns, 256, 512, dropout_per_block[2])
        self.block4 = FractalBlock(num_columns, num_columns, 512, 512, dropout_per_block[3])
        self.block5 = FractalBlock(num_columns, num_columns, 512, 512, dropout_per_block[4])
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self,x):
        y = self.conv(x)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y