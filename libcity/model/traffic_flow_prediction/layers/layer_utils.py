import torch
from torch import nn
from torch import Tensor


class Permution(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x:Tensor):
        if self.contiguous: return x.permute(*self.dims).contiguous()
        else: return x.permute(*self.dims)

def Norm(norm,embed_dim):
    if "batch" in norm.lower():
        return nn.Sequential(Permution(0,3,1,2), nn.BatchNorm2d(embed_dim), Permution(0,2,3,1))
    else:
        return nn.LayerNorm(embed_dim)