from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from math import sqrt
"""
    Untested
    Unified STattentionlayer with various masking and encoding option
"""
from libcity.model.traffic_flow_prediction.layers.pe_layers import Positional_Encoding,S_Positional_Encoding
from libcity.model.traffic_flow_prediction.layers.layer_utils import *


class _ST_Attention(nn.Module):
    def __init__(self, type, embed_dim, num_heads, scale=None, 
                 mask_flag=False, bias_flag=False, key_missing_mask_flag=False,
                 attention_dropout=0.1, output_attention=False, proj_bias=True):
        """
        Input shape:
            Q:       [batch_size (bs) x node_num x max_time_seq_len x embed_dim]
            K, V:    [batch_size (bs) x node_num x time_seq_len x embed_dim]
            mask:    [[t/n] x q_len x q_len x head_num] # dtype=torch.bool, [False] means masked/unseen attention
            bias/rencoding: [[t/n] x q_len x q_len x [head_num]]
            key_missing_mask_flag : [bs  x node_num x out_seq_len]
        
        Paramaters:
            miss_mask_flag: whether to mask missing value is ST data, refer to key_padding_mask
            scale={
                'lsa': learnable scale
                None: default
                else: given scale
            }
            attention_dropout: equals randomly attention mask
        
        Output shape:
            attention_weight/attention_score:  bnqkh or bqkth
            out:   as Q
        """
        self.mask_flag = mask_flag
        self.bias_flag = bias_flag
        self.key_missing_mask_flag = key_missing_mask_flag
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dim needs to be divisible by num_heads"
        super(_ST_Attention,self).__init__()

        if scale == 'lsa':
            self.scale = nn.Parameter(torch.tensor(self.head_dim ** -0.5), requires_grad=True)
        else:
            self.scale = scale if scale is not None else 1. / sqrt(embed_dim)
        
        self.type = type
        self.output_attention = output_attention
        self.attn_dropout = nn.Dropout(attention_dropout)

        ## from STTN shared multihead parameters? O((d/n)^2)
        # self.values_proj = nn.Linear(self.head_dim, self.head_dim, bias=proj_bias)
        # self.keys_proj = nn.Linear(self.head_dim, self.head_dim, bias=proj_bias)
        # self.queries_proj = nn.Linear(self.head_dim, self.head_dim, bias=proj_bias)
        
        # full project O(d^2)
        self.values_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)
        self.keys_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)
        self.queries_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)

        ## independent project (n * (d/n)^2 )
        # self.values_proj = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias=proj_bias) for _ in range(self.num_heads)])
        # self.keys_proj = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias=proj_bias) for _ in range(self.num_heads)])
        # self.queries_proj = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias=proj_bias) for _ in range(self.num_heads)])

        self.att_dropout_layer = nn.Dropout(attention_dropout)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, value:Tensor, key:Tensor, query:Tensor,
                attn_mask:Optional[Tensor]=None,attn_bias:Optional[Tensor]=None,key_missing_mask:Optional[Tensor]=None):
        
        batch_size, num_nodes, input_window, embed_dim = query.shape

        if self.mask_flag:
            assert (
                attn_mask is not None
            ), "Require available mask!"
            attn_mask = ~attn_mask
        
        # Projection & head split

        ## from STTN shared parameters
        # value = value.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        # key = key.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        # query = query.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        # value = self.values_proj(value)
        # key = self.keys_proj(key)
        # query = self.queries_proj(query)

        # full project
        value = self.values_proj(value)
        key = self.keys_proj(key)
        query = self.queries_proj(query)
        value = value.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        query = query.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)

        # Spatial attention
        if self.type == 'S': 
            attention_score = torch.einsum("bqthd,bkthd->bqkth", [query, key])
            # masking & relative position enocding
            if self.mask_flag:
                attention_score = attention_score.permute(0,3,1,2,4) # btqkh
                # masked_fill_ [True] means masked/unseen attention
                attention_score.masked_fill_(attn_mask, -1e10)
                attention_score = attention_score.permute(0,2,3,1,4) 

            if self.bias_flag:
                attention_score = attention_score.permute(0,3,1,2,4) # btqkh
                attention_score += attn_bias
                attention_score = attention_score.permute(0,2,3,1,4) 

            if self.key_missing_mask_flag and key_missing_mask is not None:
                #bqkth bkt -> b1kt-> b1kt1
                attention_score.masked_fill_(key_missing_mask.unsqueeze(1).unsqueeze(-1), -1e10)


            attention_weight = torch.softmax(attention_score * self.scale, dim=2)
            attention_weight = self.attn_dropout(attention_weight)
            out = torch.einsum("bqkth,bkthd->bqthd", [attention_weight, value]).reshape(
                batch_size, num_nodes, input_window, self.num_heads * self.head_dim
            )
        elif self.type == 'T':
            attention_score = torch.einsum("bnqhd,bnkhd->bnqkh", [query, key])
            # masking & relative position enocding
            if self.mask_flag:
                attention_score.masked_fill_(attn_mask, -1e10)
            if self.bias_flag:
                attention_score+=attn_bias
            if self.key_missing_mask_flag and key_missing_mask is not None:
                #bnqkh bnk -> bn1k1
                attention_score.masked_fill_(key_missing_mask.unsqueeze(2).unsqueeze(-1), -1e10)

            attention_weight = torch.softmax(attention_score * self.scale, dim=3)
            attention_weight = self.attn_dropout(attention_weight)
            out = torch.einsum("bnqkh,bnkhd->bnqhd", [attention_weight, value]).reshape(
                batch_size, num_nodes, input_window, self.num_heads * self.head_dim
            )
        # nan secure
        out = torch.where(torch.isnan(out),Tensor([0,]).to(out.device),out)
        out = self.fc_out(out)
        if self.output_attention:
            return out, attention_score, attention_weight
        else: return out



class _ST_Transfomer(nn.Module):
    def __init__(self, type, embed_dim, num_heads,norm='BatchNorm',scale=None, 
                 mask_flag=False, bias_flag=False, key_missing_mask_flag=False,
                 attention_dropout=0.1,proj_dropout=0.1,
                 ffn_forward_expansion=4,activation_fn=nn.ReLU,pre_norm=False,store_attn=False):
        super(_ST_Transfomer,self).__init__()
        self.norm = norm
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dim needs to be divisible by num_heads"

        ## add absolute position encoding! # same shape as query
        
        ### add attention module
        self.attention = _ST_Attention(type=type,embed_dim=embed_dim,num_heads=num_heads,scale=scale,
                                       mask_flag=mask_flag,bias_flag=bias_flag,key_missing_mask_flag=key_missing_mask_flag,
                                       attention_dropout=attention_dropout,output_attention=store_attn)

        ### add normalized layer/ffn
        self.drop_attn = nn.Dropout(proj_dropout)
        self.norm_attn = Norm(self.norm,self.embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ffn_forward_expansion * embed_dim),
            nn.Dropout(proj_dropout),
            activation_fn(),
            nn.Linear(ffn_forward_expansion * embed_dim, embed_dim),
        )
        self.norm_ffn = Norm(self.norm,self.embed_dim)
        self.dropout_ffn = nn.Dropout(proj_dropout)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    
        
    def forward(self, value, key ,query, attn_mask=None,attn_bias=None,key_missing_mask=None):
        if self.pre_norm:
            x = self.norm_attn(x)
        # query = x + pe
        if self.store_attn:
            x1, attention_score, attention_weight = self.attention(value=value, key=key, query=query, 
                                attn_mask=attn_mask, attn_bias=attn_bias, key_missing_mask=key_missing_mask)
        else:
            x1 = self.attention(value=value, key=key, query=query, 
                                attn_mask=attn_mask, attn_bias=attn_bias, key_missing_mask=key_missing_mask)
        x = query + self.drop_attn(x1)
        if not self.pre_norm:
            x = self.norm_attn(x)
        # self.dropout_layer(self.norm1(attention + query))
        if self.pre_norm:
            x = self.norm_ffn(x)
        x1 = self.feed_forward(x)
        ## Add & Norm
        x = x + self.dropout_ffn(x1) # Add: residual connection with residual dropout
        if not self.pre_norm:
            x = self.norm_ffn(x)
        if self.store_attn:
            return x, attention_score, attention_weight
        else:
            return x


class STBlock(nn.Module):
    """
        STencoder block with 1 Stransformer and 1 Ttransformer.
        Args:
            mode: different inner structure

    """
    def __init__(self, seq_len, node_num, embed_dim, num_heads, forward_mode=0, 
                 norm='BatchNorm',scale=None, 
                 global_nodePE=None, global_tseqPE=None,
                 smask_flag=False, sbias_flag=False, 
                 tmask_flag=False, tbias_flag=False, key_missing_mask_flag=False,
                 attention_dropout=0.1,proj_dropout=0.1,activation_fn=nn.ReLU,pre_norm=False,sstore_attn=False):
        super(STBlock,self).__init__()
        self.smask_flag = smask_flag
        self.sbias_flag = sbias_flag
        self.tmask_flag = tmask_flag
        self.tbias_flag = tbias_flag
        self.key_missing_mask_flag = key_missing_mask_flag
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_forward_expansion = 4

        self.forward_mode = forward_mode
        # self.nodePE = global_nodePE if global_nodePE is not None else S_Positional_Encoding('laplacian', False, node_num, embed_dim)
        # self.tseqPE  = global_tseqPE if global_tseqPE is not None else Positional_Encoding('sincos', False, seq_len, embed_dim)

        self.STransformer = _ST_Transfomer(type='S',embed_dim=embed_dim,num_heads=num_heads,norm=norm,scale=scale,
                                          mask_flag=smask_flag, bias_flag=sbias_flag, key_missing_mask_flag= key_missing_mask_flag,
                                          attention_dropout=attention_dropout,proj_dropout=proj_dropout,activation_fn=activation_fn,
                                          ffn_forward_expansion=self.ffn_forward_expansion,pre_norm=pre_norm,store_attn=sstore_attn)
        self.TTransformer = _ST_Transfomer(type='T',embed_dim=embed_dim,num_heads=num_heads,norm=norm,scale=scale,
                                          mask_flag=tmask_flag, bias_flag=tbias_flag, key_missing_mask_flag= key_missing_mask_flag,
                                          attention_dropout=attention_dropout,proj_dropout=proj_dropout,activation_fn=activation_fn,
                                          ffn_forward_expansion=self.ffn_forward_expansion,pre_norm=pre_norm,store_attn=False)
        
        self.norm1 = Norm(norm,self.embed_dim)
        self.norm2 = Norm(norm,self.embed_dim)
        self.dropout_layer = nn.Dropout(proj_dropout)
        self.sstore_attn = sstore_attn
    
    def forward(self, x, dense_adj_mx, npe=None, tpe=None,sattn_mask=None,sattn_bias=None,tattn_mask=None,tattn_bias=None): # bntc
        # npe = npe if npe is not None else self.nodePE(dense_adj_mx).reshape(1,-1,1,self.embed_dim).contiguous()
        # tpe = tpe if tpe is not None else self.tseqPE().reshape(1,1,-1,self.embed_dim).contiguous()
        npe = npe if npe is not None else 0
        tpe = tpe if tpe is not None else 0
        if self.forward_mode==0:
            x1 = self.norm1(
                self.TTransformer(value=x, key=x ,query=x+npe,
                                  attn_mask=tattn_mask,attn_bias=tattn_bias) + x)
            if self.sstore_attn:
                xtemp,attention_score, attention_weight = self.STransformer(value=x1, key=x1 ,query=x1+tpe,
                                  attn_mask=sattn_mask,attn_bias=sattn_bias)
            else:
                xtemp = self.STransformer(value=x1, key=x1 ,query=x1+tpe,
                                  attn_mask=sattn_mask,attn_bias=sattn_bias)
            out = self.dropout_layer(self.norm2(xtemp + x1))
        if self.sstore_attn:
            return out,attention_score, attention_weight
        else:
            return out


def test(
    train_loader,eval_loader,feats_dict,
    in_seq_len,out_seq_len
    ):
    from patch_layers import patching_conv
    class testmodel(nn.Module):
        def __init__(
            self,
            in_seq_len, out_seq_len, node_num, in_channel, out_channel, embed_dim=16, num_heads=4,
            conv_stride = 1, norm_type = 'BatchNorm', device = torch.device('cpu')
            ):
            super(testmodel,self).__init__()
            self.patchencoder = patching_conv(in_channel=in_channel, embed_dim=embed_dim, in_seq_len=in_seq_len, stride=conv_stride)
            self.hid_seq_len = self.patchencoder.out_seq_len
            self.nodePEencoder = S_Positional_Encoding('laplacian', False, node_num, embed_dim,device = device)
            self.tseqPEencoder = Positional_Encoding('sincos', True, self.hid_seq_len, embed_dim,device = device)
            self.STencoder = STBlock(seq_len=self.hid_seq_len, node_num=node_num, embed_dim=embed_dim, num_heads=num_heads, forward_mode=0, 
                                    norm=norm_type,scale=None, global_nodePE=self.nodePEencoder, global_tseqPE=self.tseqPEencoder,
                                    mask_flag=False, bias_flag=False, key_missing_mask_flag=False,
                                    attention_dropout=0.1,proj_dropout=0.1,activation_fn=nn.ReLU,pre_norm=False,sstore_attn=False)
            self.linchanneldecoder = nn.Linear(embed_dim,out_channel)
            self.linseqdecoder = nn.Linear(self.hid_seq_len,out_seq_len)
            
        
        def forward(self,x,dense_adj_mx):
            # batch node_num in_t_seq in_channel
            print(x.shape)
            x = self.patchencoder(x)
            # batch node_num hid_t_seq embed_dim
            print(x.shape)
            x = self.STencoder(x,dense_adj_mx)
            print(x.shape)
            x = self.linchanneldecoder(x)
            x = x.permute(0,1,3,2)
            print(x.shape)
            x = self.linseqdecoder(x)
            x = x.permute(0,1,3,2)
            print(x.shape)
            return x

    DEVICE = torch.device('cuda')
    adj_mx = feats_dict['adj_mx']
    in_channel = feats_dict['output_dim']
    out_channel = feats_dict['feature_dim']
    node_num = feats_dict['num_nodes']

    mymodel = testmodel(in_seq_len, out_seq_len, node_num, in_channel, out_channel,device = DEVICE)
    mymodel.to(DEVICE)
    myoptimizer = torch.optim.Adam(mymodel.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # show statics
    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in mymodel.state_dict():
        print(param_tensor, '\t', mymodel.state_dict()[param_tensor].size())
        total_param += np.prod(mymodel.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)
    print('Optimizer\'s state_dict:')  
    for var_name in myoptimizer.state_dict():
        print(var_name, '\t', myoptimizer.state_dict()[var_name])

    # test 
    test_cnt = 1
    for batch in train_loader:
        batch.to_tensor(DEVICE)
        # print(batch.data['X'].shape)
        # print(batch.data['y'].shape)
        x0 = batch.data['X'].permute(0,2,1,3).contiguous()
        x1 = mymodel(x0,adj_mx)

        test_cnt-=1
        if test_cnt<=0:
            break