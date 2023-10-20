from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from math import sqrt

# from libcity.model.traffic_flow_prediction.layers.layer_utils import *
from layer_utils import *


class singlehead_attention_wo_proj(nn.Module):
    """
        Input
            Q shape:      [batch_size x out_token_num(q) x channel]
            K shape:      [batch_size x in_token_num(k)  x channel]
            V shape:      [batch_size x in_token_num(k)  x channel]
        Output
            output shape: [batch_size x out_token_num x channel]
            attn/score:   [q x k]
    """
    def __init__(self, embed_dim, scale=None, 
                 mask_flag=False, bias_flag=False, key_missing_mask_flag=False,
                 attention_dropout=0.1, output_attention=False):
        
        super(singlehead_attention_wo_proj,self).__init__()
        self.mask_flag = mask_flag
        self.bias_flag = bias_flag
        self.key_missing_mask_flag = key_missing_mask_flag
        self.embed_dim = embed_dim
        if scale == 'lsa':
            self.scale = nn.Parameter(torch.tensor(self.embed_dim ** -0.5), requires_grad=True)
        else:
            self.scale = scale if scale is not None else 1. / sqrt(embed_dim)
        self.output_attention = output_attention
        self.attn_dropout = nn.Dropout(attention_dropout)

    def forward(self, value:Tensor, key:Tensor, query:Tensor,
                attn_mask:Optional[Tensor]=None,attn_bias:Optional[Tensor]=None,key_missing_mask:Optional[Tensor]=None):
        # _, in_token_num, embed_dim = key.shape
        # _, out_token_num, embed_dim = query.shape
        if self.mask_flag:
            assert (
                attn_mask is not None
            ), "Require available mask!"
            attn_mask = ~attn_mask
        
        attention_score = torch.einsum("bqd,bkd->bqk", [query, key])
        # masking & relative position enocding
        if self.mask_flag:
            # masked_fill_ [True] means masked/unseen attention
            attention_score.masked_fill_(attn_mask, -1e10)
        if self.bias_flag:
            attention_score += attn_bias
        if self.key_missing_mask_flag and key_missing_mask is not None:
            attention_score.masked_fill_(key_missing_mask.unsqueeze(-1), -1e10)

        attention_weight = torch.softmax(attention_score * self.scale, dim=2)
        attention_weight = self.attn_dropout(attention_weight)
        out = torch.einsum("bqk,bkd->bqd", [attention_weight, value])
        # nan secure
        out = torch.where(torch.isnan(out),Tensor([0,]).to(out.device),out)
        if self.output_attention:
            return out, attention_score, attention_weight
        else: return out


class H_attention(nn.Module):
    """
        Attention without/after projection and head_split
        Input
            data:       [batch_size (bs) x this_token_num x embed_dim]
            static_hid_embeddings:    [[this_token_num x embed_dim] * layer_num]

        Output shape:
            out:   as Q
    """
    def __init__(
        self, 
        # k_token_num,
        # q_token_num,
        embed_dim,
        layer_num, # hierarchical layer number
        layers_token_num:list,
        layers_head_dim:list, # projection head for each layer
        static_hid_token_embeddings:list,   # [[this_token_num x embed_dim] * layer_num]
        hid_token_dim,
        is_first_layer=True,
        scale = None,
        attention_dropout=0.05
        ):
        super(H_attention,self).__init__()

        self.is_first_layer = is_first_layer
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.layers_token_num = layers_token_num
        self.layers_head_dim = layers_head_dim
        self.static_hid_token_embeddings = static_hid_token_embeddings
        self.hid_token_dim= hid_token_dim
        self.scale =scale
        self.attention_dropout=attention_dropout
        # self.k_token_num = k_token_num
        # self.q_token_num = q_token_num
        assert (embed_dim==sum(layers_head_dim)), "Bad embed dim and head dim."
        
        
        self.fc1 = nn.Linear(self.embed_dim,self.embed_dim)

        if self.is_first_layer:
            self.static_hid_token_embeddings.insert(0,None)
            assert (layer_num==len(layers_token_num)==len(layers_head_dim)==len(static_hid_token_embeddings)), "Layer parameter inconsistence."
            self.this_q_proj = nn.Linear(self.layers_head_dim[0],self.layers_head_dim[0])
            self.this_k_proj = nn.Linear(self.layers_head_dim[0],self.layers_head_dim[0])
            self.this_v_proj = nn.Linear(self.layers_head_dim[0],self.layers_head_dim[0])

            self.this_att = singlehead_attention_wo_proj(
                embed_dim=layers_head_dim[0], 
                scale=scale, 
                attention_dropout=attention_dropout,
                mask_flag=False, bias_flag=False, key_missing_mask_flag=False,output_attention=False)
            
        else:
            assert (layer_num==len(layers_token_num)==len(layers_head_dim)==len(static_hid_token_embeddings)), "Layer parameter inconsistence."
            self.this_q_proj_down = nn.Linear(self.hid_token_dim,self.layers_head_dim[0])
            self.this_k_proj_down = nn.Linear(self.layers_head_dim[0],self.layers_head_dim[0])
            self.this_v_proj_down = nn.Linear(self.layers_head_dim[0],self.layers_head_dim[0])
            self.this_att_down = singlehead_attention_wo_proj(
                embed_dim=layers_head_dim[0], 
                scale=scale, 
                attention_dropout=attention_dropout,
                mask_flag=False, bias_flag=False, key_missing_mask_flag=False,output_attention=False)
            

            self.this_q_proj = nn.Linear(self.layers_head_dim[0],self.layers_head_dim[0])
            self.this_k_proj = nn.Linear(self.layers_head_dim[0],self.layers_head_dim[0])
            self.this_v_proj = nn.Linear(self.layers_head_dim[0],self.layers_head_dim[0])
            self.this_att = singlehead_attention_wo_proj(
                embed_dim=layers_head_dim[0], 
                scale=scale, 
                attention_dropout=attention_dropout,
                mask_flag=False, bias_flag=False, key_missing_mask_flag=False,output_attention=False)
            
            self.this_q_proj_up = nn.Linear(self.layers_head_dim[0],self.layers_head_dim[0])
            self.this_k_proj_up = nn.Linear(self.layers_head_dim[0],self.layers_head_dim[0])
            self.this_v_proj_up = nn.Linear(self.layers_head_dim[0],self.layers_head_dim[0])
            self.this_att_up = singlehead_attention_wo_proj(
                embed_dim=layers_head_dim[0], 
                scale=scale, 
                attention_dropout=attention_dropout,
                mask_flag=False, bias_flag=False, key_missing_mask_flag=False,output_attention=False)
        
        if self.layer_num >1:
            self.next_layer = H_attention(
                embed_dim = sum(layers_head_dim[1:]),
                layer_num = self.layer_num-1, 
                layers_token_num=self.layers_token_num[1:],
                layers_head_dim =self.layers_head_dim[1:], 
                static_hid_token_embeddings = self.static_hid_token_embeddings[1:],
                hid_token_dim=self.hid_token_dim,
                is_first_layer=False,
                scale = self.scale,
                attention_dropout=self.attention_dropout
            )


        self.fc2 = nn.Linear(self.embed_dim,self.embed_dim)
    
    def forward(self,data:Tensor):
        """
        data dim = self.embed_dim
        """

        data = self.fc1(data)
        x = data[...,:self.layers_head_dim[0]]
        if self.is_first_layer:
            q = self.this_q_proj(x)
            k = self.this_k_proj(x)
            v = self.this_v_proj(x)
            x = self.this_att(value=v, key=k, query=q)

        else:
            q1 = self.this_q_proj_down(self.static_hid_token_embeddings[0].unsqueeze(0))
            k1 = self.this_k_proj_down(x)
            v1 = self.this_v_proj_down(x)
            x1 = self.this_att_down(value=v1, key=k1, query=q1)

            q2 = self.this_q_proj(x1)
            k2 = self.this_k_proj(x1)
            v2 = self.this_v_proj(x1)
            x2 = self.this_att(value=v2, key=k2, query=q2)

            q3 = self.this_q_proj_up(x)
            k3 = self.this_k_proj_up(x2)
            v3 = self.this_v_proj_up(x2)
            x = self.this_att_up(value=v3, key=k3, query=q3)
        
        if self.layer_num >1:
            data_next = data[...,self.layers_head_dim[0]:]
            x_next = self.next_layer(data_next)
            x = torch.cat([x,x_next],dim=-1)
        
        x = self.fc2(x)
        return x
        
class H_Transformer(nn.Module):
    def __init__(
        self, 
        embed_dim,
        layer_num, # hierarchical layer number
        layers_token_num:list,
        layers_head_dim:list, # projection head for each layer
        static_hid_token_embeddings:list,   # [[this_token_num x embed_dim] * layer_num]
        hid_token_dim,
        scale = None,
        attention_dropout=0.05,
        ffn_dropout = 0.05,
        norm = 'batchnorm',
        forward_expansion = 4,
        device = torch.device('cpu')
        ):
        super(H_Transformer,self).__init__()
        self.device = device
        self.norm = norm
        self.embed_dim = embed_dim
        self.forward_expansion = forward_expansion

        self.attention = H_attention(embed_dim=embed_dim,layer_num=layer_num,layers_token_num=layers_token_num,
        layers_head_dim=layers_head_dim,static_hid_token_embeddings=static_hid_token_embeddings, 
        is_first_layer=True,scale = scale,attention_dropout=attention_dropout)

        self.norm1 = Norm(self.norm,self.embed_dim)
        self.norm2 = Norm(self.norm,self.embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )
        self.dropout_layer = nn.Dropout(ffn_dropout)

    def forward(self, data):

        attention_res = self.attention(data)

        data = self.dropout_layer(self.norm1(attention_res + data))
        forward = self.feed_forward(data)
        data = self.dropout_layer(self.norm2(forward + data))
        return data



def test():
    test_mdl = H_attention(
        embed_dim=8,
        layer_num=4,
        layers_token_num=[64,8,4,1],
        layers_head_dim=[2,2,2,2] ,
        static_hid_token_embeddings=[torch.Tensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]),
                                     torch.Tensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]),
                                     torch.Tensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]),],
        hid_token_dim=4,
    )

    data = torch.rand([1,64,8])
    out = test_mdl(data)

if __name__ == '__main__':
    test()

    

    


        




