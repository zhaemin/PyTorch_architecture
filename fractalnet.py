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

class FractalBlock(nn.Module):
    def __init__(self, original_columns, num_columns, num_features, out_features, dropout_rate, drop_local, drop_global, globaldrop_column):
        super(FractalBlock,self).__init__()
        
        self.original_columns = original_columns
        self.num_columns = num_columns
        self.drop_local = drop_local
        self.drop_global = drop_global
        self.globaldrop_column = globaldrop_column
        
        if original_columns == num_columns:
            if self.training:
                if random.random() > 0.5:
                    self.drop_local = False
                    self.drop_global = True
                    self.globaldrop_column = random.randint(0, self.original_columns-1)
                else:
                    self.drop_local = True
                    self.drop_global = False
        
        if self.num_columns == 1:
            self.conv = ConvBlock(num_features,num_features,dropout_rate)
        else:
            self.conv = ConvBlock(num_features,num_features,dropout_rate)
            self.block1 = FractalBlock(original_columns, num_columns-1, num_features, out_features, dropout_rate, self.drop_local, self.drop_global, self.globaldrop_column)
            self.block2 = FractalBlock(original_columns, num_columns-1, num_features, out_features, dropout_rate, self.drop_local, self.drop_global, self.globaldrop_column)
            self.transition = TransitionLayer(num_features, out_features)
        
    def join(self, input1, input2):
        inputs = [input1,input2]
        mask = [True,True]
        if self.drop_local:
            mask = np.random.rand(2) > 0.15
            if mask[0]==False and mask[1] == False:
                mask[random.randint(0,1)] = True
            dropped_input = [i for i,m in zip(inputs, mask) if m]
        elif self.drop_global:
            mask = self.create_global_mask()
            dropped_input = [i for i,m in zip(inputs, mask) if m]
        
        if len(dropped_input) == 1:
            return dropped_input[0]
        else:
            return (dropped_input[0]+dropped_input[1])/2
    
    def create_global_mask(self):
        if self.num_columns == self.original_columns - self.globaldrop_column:
            global_mask = [1,0]
        else:
            global_mask = [0,1]
        return global_mask
    
    def forward(self,x):
        if self.num_columns == 1:
            y = self.conv(x)
            return y
        else:
            y1 = self.conv(x)
            y2 = self.block1(x)
            y2 = self.block2(y2)
            y2 = self.join(y1, y2)
            if self.num_columns == self.original_columns:
                y2 = self.transition(y2)
            return y2

class FractalNet(nn.Module):
    def __init__(self, layer_num, num_classes):
        super(FractalNet,self).__init__()
        
        dropout_per_block = [0, 0.1, 0.2, 0.3, 0.4]
        num_columns = int(math.log2(layer_num//5))+1
        
        self.conv = ConvBlock(3, 64, 0)
        self.block1 = FractalBlock(num_columns, num_columns, 64, 128, dropout_per_block[0], False, False, 0)
        self.block2 = FractalBlock(num_columns, num_columns, 128, 256, dropout_per_block[1], False, False, 0)
        self.block3 = FractalBlock(num_columns, num_columns, 256, 512, dropout_per_block[2], False, False, 0)
        self.block4 = FractalBlock(num_columns, num_columns, 512, 512, dropout_per_block[3], False, False, 0)
        self.block5 = FractalBlock(num_columns, num_columns, 512, 512, dropout_per_block[4], False, False, 0)
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