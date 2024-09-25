import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class MakePatch(nn.Module):
    def __init__(self, patch_size, channels):
        super(MakePatch,self).__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.conv = nn.Conv2d(3, channels, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        self.batch_size = x.shape[0]
        y = self.conv(x)
        y = y.view(self.batch_size, self.channels, -1)
        y = y.transpose(1, 2)
        return y

class MLP(nn.Sequential):
    def __init__(self, in_features, D):
        super(MLP,self).__init__(
            nn.Linear(in_features, D),
            nn.GELU(),
            nn.Linear(D, in_features)
        )

class MixerLayer(nn.Module):
    def __init__(self, channels, patches, dimension_s, dimension_c):
        super(MixerLayer,self).__init__()
        
        self.layernorm1 = nn.LayerNorm(channels)
        self.token_mixing_MLP = MLP(patches, dimension_s)
        self.layernorm2 = nn.LayerNorm(channels)
        self.channel_mixing_MLP = MLP(channels, dimension_c)
        
    def forward(self, x):
        y1 = self.layernorm1(x)
        y1 = y1.transpose(1, 2)
        y1 = self.token_mixing_MLP(y1)
        y1 = y1.transpose(1, 2)
        y1 += x
        y2 = self.layernorm2(y1)
        y2 = self.channel_mixing_MLP(y2)
        y2 += y1
        
        return y2

class MLPMixer(nn.Module):
    def __init__(self, layers, patch_size, num_classes):
        super(MLPMixer,self).__init__()
        
        if layers == 'S':
            self.num_layers = 8
            self.channels = 512
            self.patches = 64
            self.dimension_c = 2048
            self.dimension_s = 256
        if layers == 'B':
            self.num_layers = 12
            self.channels = 768
            self.patches = 64
            self.dimension_c = 3072
            self.dimension_s = 384
        if layers == 'L':
            self.num_layers = 24
            self.channels = 1024
            self.patches = 64
            self.dimension_c = 4096
            self.dimension_s = 512
            
        self.make_patch = MakePatch(patch_size, self.channels)
        mixerlayers = [MixerLayer(self.channels, self.patches, self.dimension_s, self.dimension_c) for i in range(self.num_layers)]
        self.mixerlayers = nn.Sequential(*mixerlayers)
        self.fc = nn.Linear(self.channels, num_classes)
    
    def forward(self, x):
        y = self.make_patch(x)
        y = self.mixerlayers(y)
        y = y.mean(dim=1)
        y = self.fc(y)
        
        return y