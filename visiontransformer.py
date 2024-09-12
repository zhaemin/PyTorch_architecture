import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class MakePatch(nn.Module):
    def __init__(self, patch_size, D):
        super(MakePatch,self).__init__()
        self.D = D
        self.patch_size = patch_size
        self.conv = nn.Conv2d(3, D, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        self.batch_size = x.shape[0]
        y = self.conv(x)
        y = y.view(self.batch_size, self.D, -1)
        y = y.transpose(1, 2)
        return y

class SelfAttention(nn.Module):
    def __init__(self, D, head_numb, device):
        super(SelfAttention,self).__init__()
        self.D = D
        self.head_numb = head_numb
        self.make_qkv = nn.Linear(D,3*D).to(device)
    
    def forward(self, x):
        qkv = self.make_qkv(x)
        q = qkv[:,:,:self.D]
        k = qkv[:,:,self.D:self.D*2]
        v = qkv[:,:,self.D*2:]
        scailing = (self.D/self.head_numb)**(-0.5)
        attention = torch.softmax((torch.matmul(q, torch.transpose(k, 1, 2))*scailing),dim=2)
        attention = torch.matmul(attention, v)
        
        return attention

class MSA(nn.Module):
    def __init__(self, D, head_numb, device):
        super(MSA,self).__init__()
        self.D = D
        self.head_numb = head_numb
        self.linear = nn.Linear(D*head_numb, self.D).to(device)
        self.attention_heads = nn.ModuleList([SelfAttention(self.D, self.head_numb, device) for _ in range(self.head_numb)])
        
    def forward(self, x):
        results = []
        for i in range(self.head_numb):
            results.append(self.attention_heads[i](x))
        results = torch.cat(results, dim=2)
        y = self.linear(results)
        return y

class Encoder(nn.Module):
    def __init__(self, D, head_numb, mlp_size, device):
        super(Encoder, self).__init__()
        self.D = D
        self.head_numb = head_numb
        self.mlp_size = mlp_size
        self.device = device
        self.layer_norm1 = nn.LayerNorm(self.D)
        self.layer_norm2 = nn.LayerNorm(self.D)
        self.mlp = nn.Sequential(
            nn.Linear(self.D, self.mlp_size),
            nn.GELU(),
            nn.Linear(self.mlp_size, self.D),
            nn.GELU(),
        )
        
    def forward(self, x):
        y1 = self.layer_norm1(x)
        y1 = MSA(self.D, self.head_numb, self.device)(y1)
        y1 += x
        y2 = self.layer_norm2(y1)
        y2 = self.mlp(y2)
        
        return y2 + y1

class VisionTransformer(nn.Module):
    def __init__(self, model, num_classes, device):
        super(VisionTransformer,self).__init__()
        if model == 'Base':
            self.D = 48
            self.head_numb = 12
            self.mlp_size = 192
            self.layers = 8
        elif model == 'Large':
            self.D = 256
            self.head_numb = 16
            self.mlp_size = 1024
            self.layers = 24
        elif model == 'Huge':
            self.D = 320
            self.head_numb = 16
            self.mlp_size = 1280
            self.layers = 32
        self.patch_size = 4
        self.device = device
        self.make_patch = MakePatch(self.patch_size, self.D)
        self.mlp_head = nn.Sequential(
            nn.Linear(self.D, self.mlp_size),
            nn.GELU(),
            nn.Linear(self.mlp_size, self.D),
            nn.GELU(),
            nn.Linear(self.D, num_classes)
            )
        encoders = [Encoder(self.D, self.head_numb, self.mlp_size, self.device) for _ in range(self.layers)]
        self.encoder = nn.Sequential(*encoders)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.D))
        n = (32*32)//self.patch_size**2
        self.position_embedding = nn.Parameter(torch.randn(1, n+1, self.D))
        
    def forward(self, x):
        batch_size = x.shape[0]
        y = self.make_patch(x)
        class_token = self.class_token.repeat(batch_size, 1, 1)
        position_embedding = self.position_embedding.repeat(batch_size, 1, 1)
        y = torch.cat((class_token, y), dim=1)
        y = y+position_embedding
        y = self.encoder(y)
        y = y[:, 0, :]
        y = self.mlp_head(y)
        return y