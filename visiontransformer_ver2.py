import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class MakePatch(nn.Module):
    def __init__(self, patch_size, dim):
        super(MakePatch,self).__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.conv = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        self.batch_size = x.shape[0]
        y = self.conv(x)
        y = y.view(self.batch_size, self.dim, -1)
        y = y.transpose(1, 2)
        return y

class MSA(nn.Module):
    def __init__(self, dim, num_heads):
        super(MSA,self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim//num_heads
        self.make_qkv = nn.Linear(dim, 3*dim)
        self.scaling = (self.dim_head)**(-0.5)
        self.linear = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, n, dim = x.shape
        qkv = self.make_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # b n d
        q = q.view(batch_size, n, self.num_heads, self.dim_head).permute(0,2,1,3)
        k = k.view(batch_size, n, self.num_heads, self.dim_head).permute(0,2,1,3)
        v = v.view(batch_size, n, self.num_heads, self.dim_head).permute(0,2,1,3)
        att = torch.softmax(torch.einsum('bhqd, bhkd -> bhqk', q, k)*self.scaling, dim=-1)
        y = torch.einsum('bhan, bhnd -> bhad', att, v) 
        y = y.permute(0,2,1,3).reshape(batch_size, n, dim)
        y = self.linear(y)
        return y

class Encoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super(Encoder, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio*dim),
            nn.GELU(),
            nn.Linear(mlp_ratio*dim, dim),
        )
        self.msa = MSA(dim, num_heads)
        
    def forward(self, x):
        y1 = self.norm(x)
        y1 = self.msa(y1)
        y1 += x
        y2 = self.norm(y1)
        y2 = self.mlp(y2)
        
        return y2 + y1

class VisionTransformer(nn.Module):
    def __init__(self, num_classes, patch_size, dim, num_heads, mlp_ratio, layers):
        super(VisionTransformer,self).__init__()
        self.make_patch = MakePatch(patch_size, dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(dim*mlp_ratio, num_classes)
        )
        self.norm = nn.LayerNorm(dim)
        encoders = [Encoder(dim, num_heads, dim*mlp_ratio) for _ in range(layers)]
        self.encoder = nn.Sequential(*encoders)
        self.class_token = nn.Parameter(torch.randn(1, 1, dim).mul_(0.02))
        n = (32*32)//patch_size**2
        self.position_embedding = nn.Parameter(torch.randn(1, n+1, dim).mul_(0.02))
        
    def forward(self, x):
        batch_size = x.shape[0]
        y = self.make_patch(x)
        class_token = self.class_token.repeat(batch_size, 1, 1)
        position_embedding = self.position_embedding.repeat(batch_size, 1, 1)
        y = torch.cat((class_token, y), dim=1)
        y = y+position_embedding
        y = self.norm(self.encoder(y))
        y = y[:, 0, :]
        y = self.mlp_head(y)
        return y
    
def ViT_Base(num_classes):
    return VisionTransformer(num_classes, patch_size=4, dim=48, num_heads=12, mlp_ratio=4, layers=12)

def ViT_Large(num_classes):
    return VisionTransformer(num_classes, patch_size=4, dim=1024, num_heads=16, mlp_ratio=4, layers=24)

def ViT_Huge(num_classes):
    return VisionTransformer(num_classes, patch_size=4, dim=1280, num_heads=16, mlp_ratio=4, layers=32)