from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv,SimpleConv
from torch_geometric.utils import dense_to_sparse
import numpy as np
import math
from math import sqrt
from libcity.model.traffic_flow_prediction.layers.layer_utils import *
"""
    short range time series patch generation
    3/13: consider missing as a time-spatiol series signal.
"""


class MixhopConv(nn.Module):
    def __init__(self, gdep=3, alpha=0):
        super(MixhopConv, self).__init__()
        # self.mlp = nn.Linear((gdep+1)*c_in, c_out)
        self.gdep = gdep
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        adj = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h1 = torch.einsum('bntc,nm->bmtc', (h, adj))
            # h1 = torch.einsum('ncwl,vw->ncvl', (h, adj))
            # batch_size (bs) * node_num x time_seq_len x input_channel
            h = self.alpha*x + (1-self.alpha)*h1
            out.append(h)
        ho = torch.cat(out, dim=-1)
        # ho = self.mlp(ho)
        return ho


# tested
class patching_conv(nn.Module):
    """
    Input/Output shape:
        input: [batch_size (bs) * node_num x time_seq_len x input_channel] 
        output: [batch_size (bs) * node_num x patch_num x embed_dim(out_channel*kernel_size)]
    """
    def __init__(self, in_channel, embed_dim, in_seq_len, kernel_sizes:list=[1,2,3,6], stride=1, activation_fn=nn.Tanh):
        super(patching_conv, self).__init__()
        self.kernel_num = len(kernel_sizes)
        assert (
            embed_dim % self.kernel_num == 0
        ), "Embedding dim needs to be divisible by kernel_size"
        self.kernel_size = kernel_sizes
        self.in_channel = in_channel
        self.out_channel = embed_dim // self.kernel_num
        self.embed_dim = embed_dim
        self.in_seq_len = in_seq_len
        self.out_seq_len = math.ceil(in_seq_len / stride)

        # pad seq for unified patch_num / len(shape)<=3
        self.paddings = nn.ModuleList([
            nn.ReplicationPad1d((round((ks - 1) / 2), (ks-1)-round((ks - 1) / 2))) 
            for ks in kernel_sizes
        ])
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.in_channel,out_channels=self.out_channel,kernel_size=ks,stride=stride) 
            for ks in kernel_sizes
        ])
        self.activation = activation_fn()

    def _t_patch_reshape(self, batch_size, node_num, x:Tensor, mode=0):
        if mode ==0: return x.view(batch_size*node_num,x.shape[2],x.shape[3])
        else: return x.reshape(batch_size,node_num,x.shape[2],x.shape[3])

    def forward(self, x:Tensor):
        batch_size,node_num,t_len,in_channel = x.shape
        x = x.view([-1,t_len,in_channel])
        x = x.permute(0,2,1) # b n i t
        out = list()
        for i in range(self.kernel_num):
            xi = self.paddings[i](x)
            xi = self.convs[i](xi) # b nn ed pn
            xi = xi.permute(0,2,1)
            out.append(xi)
        out = torch.cat(out,dim=-1)
        out = out.reshape([batch_size,node_num,-1,self.embed_dim]).contiguous()
        out = self.activation(out)
        return out




class patching_STconv(nn.Module):
    """
    adding k timeseq filter and a/k graph filter
    change to MixhopConv
    Input/Output shape:
        input: [batch_size (bs) x node_num x time_seq_len x input_channel] 
        output: [batch_size (bs) x node_num x patch_num x embed_dim(out_channel*kernel_size)]
    """
    def __init__(self, in_channel, embed_dim, in_seq_len, kernel_sizes:list=[1,2,3,6], stride=1,
                 gdep = 3, alpha = 0, norm = 'BatchNorm',
                 activation_fn=nn.Tanh,device=torch.device('cpu')):
        super(patching_STconv, self).__init__()
        self.device = device
        self.kernel_num = len(kernel_sizes)
        assert (
            embed_dim % (self.kernel_num*(gdep+1)) == 0
        ), "Embedding dim needs to be divisible by kernel_size"
        self.kernel_size = kernel_sizes
        self.in_channel = in_channel
        self.out_channel = embed_dim // (self.kernel_num*(gdep+1))
        self.embed_dim = embed_dim
        self.in_seq_len = in_seq_len
        self.out_seq_len = math.ceil(in_seq_len / stride)
        self.gdep = gdep
        self.alpha = alpha
        self.norm = norm

        # pad seq for unified patch_num / len(shape)<=3
        self.paddings = nn.ModuleList([
            nn.ReplicationPad1d((round((ks - 1) / 2), (ks-1)-round((ks - 1) / 2))) 
            for ks in kernel_sizes
        ])
        self.tconvs = nn.ModuleList([
            nn.Conv1d(in_channels=self.in_channel,out_channels=self.out_channel,kernel_size=ks,stride=stride) 
            for ks in kernel_sizes
        ])
        self.gconv = MixhopConv(gdep = self.gdep,alpha = self.alpha)
        self.norm = Norm(self.norm,self.embed_dim)
        self.activation = activation_fn()
    
    def forward(self, x:Tensor,dense_adj_mx): 
        # edge_index,edge_weight = dense_to_sparse(torch.from_numpy(dense_adj_mx))
        # edge_index,edge_weight = edge_index.to(self.device),edge_weight.to(self.device)
        batch_size,node_num,t_len,in_channel = x.shape
        x = x.view([-1,t_len,in_channel])
        x = x.permute(0,2,1) # b*n i t
        out = list()
        for i in range(self.kernel_num):
            # timeseq pattern extraction
            xi:Tensor = self.paddings[i](x)
            xi = self.tconvs[i](xi) # b*n ed pn
            xi = xi.permute(0,2,1)
            xi = xi.reshape([batch_size,node_num,-1,self.out_channel]).contiguous()
            # neighborhood pattern extraction
            out.append(xi)
        out = torch.cat(out,dim=-1)
        # out = out.permute(0,2,1,3).contiguous()
        # out = self.gconv(out,edge_index,edge_weight) # b t n c
        out = self.gconv(out,torch.from_numpy(dense_adj_mx).to(self.device))
        # out = out.permute(0,2,1,3)
        out = self.activation(out)
        return out



class depatching_conv(nn.Module):
    """
    depatch conv transpose layer with linear decoder
    """
    def __init__(self, embed_dim, unpatch_channel, out_channel, hid_seq_len, out_seq_len, kernal_size=None,stride=None, activation_fn=nn.Tanh):
        super(depatching_conv, self).__init__()
        self.embed_dim = embed_dim
        self.unpatch_channel = unpatch_channel
        self.out_channel = out_channel
        self.in_len = hid_seq_len
        self.out_len = out_seq_len
        self.stride = stride or math.ceil(self.out_len/self.in_len)
        self.kernal = kernal_size or math.ceil(self.out_len/self.in_len)
        assert (
            self.kernal >= self.stride
        ), "Bad kernal size"
        # self.unpatch_seq_len = self.stride * (self.in_len+self.kernal-1)+self.stride-self.kernal
        self.unpatch_seq_len = self.stride * self.in_len


        self.padding =  nn.ReplicationPad1d((round((self.kernal - 1) / 2), (self.kernal-1)-round((self.kernal - 1) / 2))) 
        self.tconv = nn.ConvTranspose1d(in_channels=self.embed_dim,out_channels=self.unpatch_channel,
                                         kernel_size=self.kernal,stride=self.stride)
        self.seqlin = nn.Sequential( 
            # in [b,n,patch_seq_len,embed_dim] 
            # out [b,n,out_seq_len,b,n,out_dim]
            nn.Linear(self.unpatch_seq_len,self.out_len),
            Permution(0,2,1),
            nn.Linear(self.unpatch_channel,self.out_channel),
        )
        self.activation = activation_fn()
    
    def forward(self, x:Tensor): 
        batch_size,node_num,t_len,in_channel = x.shape
        xt = x.reshape([-1,t_len,in_channel]).contiguous()
        xt = xt.permute(0,2,1) # b*n c t
        xt = self.padding(xt)
        xt = self.tconv(xt)
        if round(((self.kernal-1)*self.stride)/2)-((self.kernal-1)*self.stride) == 0:
            xt = xt[...,round(((self.kernal-1)*self.stride+(self.kernal-self.stride))/2):]
        else:
            xt = xt[...,round(((self.kernal-1)*self.stride+(self.kernal-self.stride))/2): \
                        round(((self.kernal-1)*self.stride+(self.kernal-self.stride))/2)-((self.kernal-1)*self.stride+(self.kernal-self.stride))]
        xt = self.activation(xt)
        xt = self.seqlin(xt)
        xt = xt.reshape([batch_size,node_num,-1,self.out_channel]).contiguous()
        
        return xt



"""
    for ref ------------------------------------------------------------------------------------------------------------------------------------
"""
class patching_TST(nn.Module):
    # Patching
    # in: [bs x nvars x seq_len]
    # out: [bs x nvars x patch_len x patch_num]
    def __init__(self,patch_len,padding_patch,context_window, stride=2):
        super(patching_TST, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1
    
    def forward(self, z:torch.Tensor):                                                                  
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)         
        return z,self.patch_num                                                     # z: [bs x nvars x patch_len x patch_num]


"""
    error zero loss!
"""
class patching_STconv_prev(nn.Module):
    """
    test version , adding k time seq filter and a graph filter
    Input/Output shape:
        input: [batch_size (bs) x node_num x time_seq_len x input_channel] 
        output: [batch_size (bs) x node_num x patch_num x embed_dim(out_channel*kernel_size)]
    """
    def __init__(self, in_channel, embed_dim, in_seq_len, kernel_sizes:list=[2,3,6,12], stride=1, activation_fn=nn.Tanh):
        super(patching_STconv_prev, self).__init__()
        self.kernel_num = len(kernel_sizes)
        assert (
            embed_dim % (self.kernel_num+1) == 0
        ), "Embedding dim needs to be divisible by kernel_size"
        self.kernel_size = kernel_sizes
        self.in_channel = in_channel
        self.out_channel = embed_dim // (self.kernel_num+1)
        self.embed_dim  = embed_dim
        self.in_seq_len = in_seq_len
        self.out_seq_len = math.ceil(in_seq_len / stride)
        self.activation = activation_fn()

        # pad seq for unified patch_num / len(shape)<=3
        self.paddings = nn.ModuleList([
            # nn.ReplicationPad1d((round((ks - 1) / 2), ks-round((ks - 1) / 2))) 
            nn.ReplicationPad1d((round((ks - 1) / 2), (ks-1)-round((ks - 1) / 2))) 
            for ks in kernel_sizes
        ])
        self.tconvs = nn.ModuleList([
            nn.Conv1d(in_channels=self.in_channel,out_channels=self.out_channel,kernel_size=ks,stride=stride) 
            for ks in kernel_sizes
        ])
        self.sconv = GCNConv(in_channels=self.in_channel,out_channels=self.out_channel)
        # align time-seq
        self.slin = nn.Linear(in_features=in_seq_len,out_features=self.out_seq_len)
            # nn.Conv1d(in_channels=self.in_channel,out_channels=self.out_channel,kernel_size=ks,stride=stride) 

    def forward(self, x:Tensor,dense_adj_mx): 
        edge_index,_ = dense_to_sparse(torch.from_numpy(dense_adj_mx))
        batch_size,node_num,t_len,in_channel = x.shape
        xt = x.view([-1,t_len,in_channel])
        xt = xt.permute(0,2,1) # b*n c t
        xs = x.permute(0,2,1,3) # b t n c
        out = list()
        for i in range(self.kernel_num):
            xti = self.paddings[i](xt)
            print(f'kernel {i}-th padded:',xti.shape)
            xti = self.tconvs[i](xti) # b nn ed pn
            xti = xti.permute(0,2,1)
            out.append(xti)
        out = torch.cat(out,dim=-1)

        out = out.reshape([batch_size,node_num,-1,self.out_channel*self.kernel_num]).contiguous()
        
        xs = self.sconv(xs,edge_index) # b t n c
        xs = xs.permute(0,2,3,1) # b n c t
        xs = self.slin(xs)
        xs = xs.permute(0,1,3,2)
        out = torch.cat([out,xs],dim=-1)
        out = self.activation(out)
        return out


class MixhopConv_prev(nn.Module):
    def __init__(self, c_in, c_out, gdep=3, dropout=0.1, alpha=0.2):
        super(MixhopConv, self).__init__()
        self.nconv = SimpleConv(aggr='sum',combine_root='sum')
        self.mlp = nn.Linear((gdep+1)*c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, dense_adj_mx):
        ### utils.normalized_cut, utils.add_self_loops
        dense_adj_mx = dense_adj_mx + torch.eye(dense_adj_mx.size(0)).to(x.device)
        d = dense_adj_mx.sum(1)
        h = x
        out = [h]
        dense_adj_mx = dense_adj_mx / d.view(-1, 1)
        edge_index,edge_weight = dense_to_sparse(torch.from_numpy(dense_adj_mx))
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(x=h, edge_index=edge_index,edge_weight=edge_weight)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho



if __name__ == "__main__":
    # conv = depatching_conv(embed_dim=64, unpatch_channel=16, out_channel=3, hid_seq_len=12, out_seq_len=17,stride=3)
    # x = torch.rand([1,27,12,64])
    # # x = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
    # # x = torch.tensor([[[1.,2.,3.],[2.,3.,4.]]])
    # x = conv(x)
    # print(x.shape)


    conv = patching_STconv(in_channel = 3, embed_dim = 64, in_seq_len=12)
    x = torch.rand([1,10,12,3])
    dense_adj_mx = torch.rand([10,10]).numpy()
    x = conv(x,dense_adj_mx)
    print(x.shape)
