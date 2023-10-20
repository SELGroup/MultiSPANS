import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model.traffic_flow_prediction.layers.pe_layers import *
from libcity.model.traffic_flow_prediction.layers.STTransformer_layers import *
from libcity.model.traffic_flow_prediction.layers.patch_layers import *
from libcity.model.traffic_flow_prediction.layers.mask_layers import *

class STTransformer(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = data_feature.get('adj_mx')
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        outfeat_dim = config.get('outfeat_dim',None)
        self.output_dim = outfeat_dim if outfeat_dim is not None else self.data_feature.get('output_dim', 1)
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        # self.ext_dim = self.data_feature.get("ext_dim", 0)
        # self.num_batches = self.data_feature.get('num_batches', 1)
        self.load_external = config.get('load_external', False)
        if self.load_external:
            self.feature_dim -= 8
        self._logger = getLogger()
        
        self.device = config.get('device', torch.device('cpu'))
        self.embed_dim = config.get('embed_dim', 64)
        self.skip_conv_flag = config.get('skip_conv_flag', True)
        self.residual_conv_flag = config.get('residual_conv_flag', True)
        self.skip_dim = config.get('skip_dim', self.embed_dim)
        self.num_layers = config.get('num_layers', 3)
        self.num_heads = config.get('num_heads', 8)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)

        self.gconv_hop_num = config.get('gconv_hop_num',3)
        self.gconv_alpha = config.get('gconv_alpha',0)

        self.conv_kernels = config.get('conv_kernels',[2,3,6,12])
        self.conv_stride = config.get('conv_stride',1)
        self.conv_if_gc = config.get('conv_if_gc',False)

        self.norm_type = config.get('norm_type','BatchNorm')
        
        self.att_scale = config.get('att_scale',None)
        self.att_dropout = config.get('att_dropout',0.1)
        self.ffn_dropout = config.get('ffn_dropout',0.1)
        self.Spe_type = config.get('Satt_pe_type','laplacian')
        self.Spe_learnable = config.get('Spe_learnable',False)
        self.Tpe_type = config.get('Tatt_pe_type','sincos')
        self.Tpe_learnable = config.get('Tpe_learnable',False)
        self.Smask_flag = config.get('Smask_flag',True)
        self.block_forward_mode = config.get('block_forward_mode',0)
        self.sstore_attn = config.get('sstore_attn',False)
        # static parameters 
        self.activition_fn = nn.ReLU

        if self.skip_conv_flag is False:
            self.skip_dim = self.embed_dim
        """
            3/28: 需要skip connection
                  需要depatch解码器/ST解码器
                  加入multi-mask机制
        """

        self.patchencoder = patching_STconv(
            in_channel=self.feature_dim , embed_dim=self.embed_dim, 
            in_seq_len=self.input_window, 
            gdep = self.gconv_hop_num, alpha = self.gconv_alpha,
            kernel_sizes=self.conv_kernels,stride=self.conv_stride,device=self.device
            )  if self.conv_if_gc else patching_conv(
            in_channel=self.feature_dim , embed_dim=self.embed_dim, 
            in_seq_len=self.input_window, kernel_sizes=self.conv_kernels,stride=self.conv_stride
            )
        self.hid_seq_len = self.patchencoder.out_seq_len
        if self.Smask_flag:
            self.infomask = Infomap_Multi_Mask_Generator(self.num_nodes,self.adj_mx)
            self.graphmask = Graph_Mask_Generator(self.num_nodes,self.adj_mx)
        self.externalPEencoder = External_Encoding(d_model=self.embed_dim, device=self.device)
        self.nodePEencoder = S_Positional_Encoding(
            pe_type=self.Spe_type, learn_pe=self.Spe_learnable, node_num=self.num_nodes, 
            d_model=self.embed_dim,device = self.device)
        self.tseqPEencoder = Positional_Encoding(
            pe_type=self.Tpe_type, learn_pe=self.Tpe_learnable, q_len=self.hid_seq_len, 
            d_model=self.embed_dim,device = self.device)
        self.STencoders = nn.ModuleList(
            [STBlock(
                seq_len=self.hid_seq_len,node_num=self.num_nodes,embed_dim=self.embed_dim,num_heads=self.num_heads,
                forward_mode=self.block_forward_mode,norm=self.norm_type,scale=self.att_scale,
                global_nodePE=self.nodePEencoder,global_tseqPE=self.tseqPEencoder,smask_flag=self.Smask_flag,sbias_flag=False,
                tmask_flag=False,tbias_flag=False,key_missing_mask_flag=False,
                attention_dropout=self.att_dropout,proj_dropout=self.ffn_dropout,activation_fn=self.activition_fn,
                pre_norm=False,sstore_attn=self.sstore_attn
            ) for _ in range(self.num_layers)]
        )
        
        if self.skip_conv_flag:
            self.skip_convs = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
                ) for _ in range(self.num_layers+1)
            ])

        if self.residual_conv_flag:
            self.residual_convs = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=1,
                ) for _ in range(self.num_layers)
            ])

        self.lineardecoder = depatching_conv(embed_dim=self.skip_dim, unpatch_channel=self.skip_dim//2, out_channel=self.output_dim, 
                                            hid_seq_len = self.hid_seq_len, out_seq_len=self.output_window)

        # self.lineardecoder = nn.Sequential( 
        #     # in [b,n,patch_seq_len,embed_dim] 
        #     # out [b,n,out_seq_len,b,n,out_dim]
        #     nn.Linear(self.skip_dim,self.output_dim),
        #     Permution(0,1,3,2),
        #     nn.Linear(self.hid_seq_len,self.output_window),
        #     Permution(0,1,3,2)
        # )

        self.droput_layer = nn.Dropout(p=self.ffn_dropout)

    def forward(self, batch):
        # dense_adj_mx = self.adj_mx
        # multimask = self.multimask
        # npe = self.nodePEencoder(dense_adj_mx).reshape(1,-1,1,self.embed_dim).contiguous()
        # tpe = self.tseqPEencoder().reshape(1,1,-1,self.embed_dim).contiguous()
        # x = batch['X'].permute(0,2,1,3).contiguous()
        # if self.conv_if_gc:
        #     x = self.patchencoder(x,dense_adj_mx)
        # else: x = self.patchencoder(x) # [b,n,patch_seq_len,embed_dim]
        # skips = []
        # for block in self.STencoders:
        #     x = block(x,dense_adj_mx, npe, tpe, sattn_mask=multimask)  # [b,n,patch_seq_len,embed_dim]
        #     skips.append(x)
        # # out = torch.sum(torch.stack(skips))
        # x = self.droput_layer(x)
        # out = self.lineardecoder(x).permute(0,2,1,3).contiguous()
        # return out
        dense_adj_mx = self.adj_mx
        if self.Smask_flag:
            multimask = get_static_multihead_mask(self.num_heads,[self.infomask,self.graphmask],device=self.device)
        else : multimask = None
        npe = self.nodePEencoder(dense_adj_mx).reshape(1,-1,1,self.embed_dim).contiguous()
        tpe = self.tseqPEencoder().reshape(1,1,-1,self.embed_dim).contiguous()
        x = batch['X'].permute(0,2,1,3).contiguous() # btnc -> bntc
        if self.load_external:
            x, epe = self.externalPEencoder(x)
            npe, tpe = npe+epe, tpe+epe
        if self.conv_if_gc:
            x = self.patchencoder(x,dense_adj_mx)
        else: x = self.patchencoder(x) # [b,n,patch_seq_len,embed_dim]

        skip = self.skip_convs[-1](x.permute(0,3,2,1)) if self.skip_conv_flag else x
        if self.sstore_attn:
            for i,block in enumerate(self.STencoders):
                h,attention_score, attention_weight = block(x,dense_adj_mx, npe, tpe, sattn_mask=multimask)  # [b,n,patch_seq_len,embed_dim]
                skip = skip+self.skip_convs[i](h.permute(0,3,2,1)) if self.skip_conv_flag else skip+h
                x = self.residual_convs[i](x.permute(0,3,2,1)).permute(0,3,2,1)+h if self.residual_conv_flag else x+h
                if self.training is not True:
                    import time
                    t = time.localtime()
                    torch.save({'attention_score':attention_score, 'attention_weight':attention_weight},"./attn_save/{}_att.pt".format(time.strftime("%d_%H_%M_%S",t)))
        
        else:
            for i,block in enumerate(self.STencoders):
                h = block(x,dense_adj_mx, npe, tpe, sattn_mask=multimask)  # [b,n,patch_seq_len,embed_dim]
                skip = skip+self.skip_convs[i](h.permute(0,3,2,1)) if self.skip_conv_flag else skip+h
                x = self.residual_convs[i](x.permute(0,3,2,1)).permute(0,3,2,1)+h if self.residual_conv_flag else x+h
        skip = skip.permute(0,3,2,1) if self.skip_conv_flag else skip
        # out = torch.sum(torch.stack(skips))
        skip = self.droput_layer(skip)
        out = self.lineardecoder(skip).permute(0,2,1,3).contiguous()
        return out
       
    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)