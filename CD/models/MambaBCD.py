import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from changedetection.models.Mamba_backbone import Backbone_VSSM
from classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from changedetection.models.ChangeDecoder import ChangeDecoder,Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count


class STMambaBCD(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(STMambaBCD, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        self.batch_norm_layers = nn.ModuleList([nn.BatchNorm2d(channels) for channels in self.encoder.dims])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
 

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)
        # self.decoder=Decoder(encoder_dims=self.encoder.dims)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)   
        #print(x1.shape)  torch.Size([4, 128, 64, 64])
        post_features = self.encoder(post_data)
        # out_sum = []  # 存储每一层的结果
        # out_diff = []  # 存储每一层的结果
        # for i in range(len(pre_features)):.
        #     # 对应特征相加并应用批归一化和ReLU操作
        #     xi = pre_features[i]
        #     yi = post_features[i]

        #     out_i = F.relu(self.batch_norm_layers[i](xi + yi))
        #     #print(out_i.shape)
            
        #     # 对应特征相减并应用批归一化和ReLU操作
        #     out_j = F.relu(self.batch_norm_layers[i](xi - yi))
            
            # # 对 out_i 应用平均池化操作
            # avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            # out_i_pool,out_j_pool = avg_pool(out_i),avg_pool(out_j)
            
            # # 计算 out_i 减去平均池化结果的差值
            # out_i_d = out_i - out_i_pool
            # # 计算 out_j 减去平均池化结果的差值
            # out_j_d = out_j - out_j_pool
            # outsumi=torch.cat((out_i, out_i_d), dim=1)
            # outdiffj=torch.cat((out_j, out_j_d), dim=1)
            # 将结果存储在列表中
            # out_sum.append((out_i))
            # out_diff.append((out_j))
        # Decoder processing - passing encoder outputs to the decoder
        output = self.decoder(pre_features,post_features)
        output = self.main_clf(output)
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        return output
