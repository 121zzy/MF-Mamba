import torch
import torch.nn as nn
import torch.nn.functional as F
from classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute

class ChannelAttention_1(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention_1, self).__init__()
        self.conv = nn.Conv2d(in_planes*2, in_planes, 3, 1, 1)
        self.bn = nn.BatchNorm2d(in_planes)
        self.SiLU = nn.SiLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出最后两维1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.SiLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x) 
        x = self.bn(x)  
        x = self.SiLU(x) 
        res = x
        avg_out = self.fc(self.avg_pool(x)) 
        max_out = self.fc(self.max_pool(x)) 
        out = avg_out + max_out
        result = x * self.sigmoid(out) + res   
        return result
    
class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv =nn.Conv2d(in_channels*2,in_channels,kernel_size=1)
        self.conv3= ChannelAttention_1(in_channels)

    def forward(self, feat1, feat2):
        # 求和操作
        sum_feat = feat1 + feat2
        # 求差操作
        diff_feat = torch.abs(feat1 - feat2)
        # 对求和结果应用卷积和批归一化
        # f1=self.conv1(sum_feat)
        # f2=self.bn(f1)
        # f3=self.relu(f3)
        sum_feat = self.relu(self.bn(self.conv1(sum_feat)))
        # 对求差结果应用卷积和批归一化
        diff_feat = self.relu(self.bn(self.conv2(diff_feat)))
        fusion=self.conv(torch.cat((sum_feat,diff_feat),dim=1))
        fusion = self.conv3(fusion)
        return fusion
    
class ChangeDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(ChangeDecoder, self).__init__()
        # self.fusion4 = FeatureFusion(in_channels=encoder_dims[-1] )
        # self.max_pool= nn.Conv2d(in_channels=encoder_dims[-2], out_channels=encoder_dims[-1], kernel_size=3, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(128, out_channels=encoder_dims[-2], kernel_size=1)  # 添加卷积层改变通道数量
        # self.conv2 = nn.Conv2d(128, out_channels=encoder_dims[-3], kernel_size=1)  # 添加卷积层改变通道数量
        # self.conv1 = nn.Conv2d(128, out_channels=encoder_dims[-4], kernel_size=1)  # 添加卷积层改变通道数量
        # # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_41 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_42 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),

        )
        self.st_block_43 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        # self.fusion3 = FeatureFusion(in_channels=encoder_dims[-2])
        self.st_block_31 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_32 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_33 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        # self.fusion2 = FeatureFusion(in_channels=encoder_dims[-3])
        self.st_block_21 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_22 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_23 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        # self.fusion1 = FeatureFusion(in_channels=encoder_dims[-4])
        self.st_block_11 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_12 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_13 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        # Fuse layer  
        self.fuse_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())

        # Smooth layer
        self.smooth_layer_3 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_2 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_1 = ResBlock(in_channels=128, out_channels=128, stride=1) 
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_features, post_features):

        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        # F4 =self.fusion4(pre_feat_4, post_feat_4)
        # F3 =self.fusion3(pre_feat_3, post_feat_3)
        # F3 =self.max_pool(F3)
        '''
            Stage I
        '''
        p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))
        B, C, H, W = pre_feat_4.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_42 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_42[:, :, :, ::2] = pre_feat_4  # Odd columns
        ct_tensor_42[:, :, :, 1::2] = post_feat_4 # Even columns
        p42 = self.st_block_42(ct_tensor_42)

        ct_tensor_43 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_43[:, :, :, 0:W] = pre_feat_4
        ct_tensor_43[:, :, :, W:] = post_feat_4
        p43 = self.st_block_43(ct_tensor_43)

        # p41 = self.st_block_41(torch.cat([F4, F3], dim=1))
        # B, C, H, W = F4.size()
        # # Create an empty tensor of the correct shape (B, C, H, 2*W)
        # ct_tensor_42 = torch.empty(B, C, H, 2*W).cuda()
        # # Fill in odd columns with A and even columns with B
        # ct_tensor_42[:, :, :, ::2] = F4  # Odd columns
        # ct_tensor_42[:, :, :, 1::2] = F3  # Even columns
        # p42 = self.st_block_42(ct_tensor_42)

        # ct_tensor_43 = torch.empty(B, C, H, 2*W).cuda()
        # ct_tensor_43[:, :, :, 0:W] = F4
        # ct_tensor_43[:, :, :, W:] = F3
        # p43 = self.st_block_43(ct_tensor_43)

        p4 = self.fuse_layer_4(torch.cat([p41, p42[:, :, :, ::2], p42[:, :, :, 1::2], p43[:, :, :, 0:W], p43[:, :, :, W:]], dim=1))
       

        '''
            Stage II
        '''
        # f3 =self.fusion3(pre_feat_3, post_feat_3)
        # out4 = F.interpolate(self.conv3(p4), scale_factor=2, mode='nearest')

        p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))
        B, C, H, W = pre_feat_3.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_32 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_32[:, :, :, ::2] = pre_feat_3 # Odd columns
        ct_tensor_32[:, :, :, 1::2] = post_feat_3 # Even columns
        p32 = self.st_block_32(ct_tensor_32)

        ct_tensor_33 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_33[:, :, :, 0:W] = pre_feat_3
        ct_tensor_33[:, :, :, W:] = post_feat_3
        p33 = self.st_block_33(ct_tensor_33)

        # p31 = self.st_block_31(torch.cat([f3, out4], dim=1))
        # B, C, H, W = f3.size()
        # # Create an empty tensor of the correct shape (B, C, H, 2*W)
        # ct_tensor_32 = torch.empty(B, C, H, 2*W).cuda()
        # # Fill in odd columns with A and even columns with B
        # ct_tensor_32[:, :, :, ::2] = f3 # Odd columns
        # ct_tensor_32[:, :, :, 1::2] = out4 # Even columns
        # p32 = self.st_block_32(ct_tensor_32)

        # ct_tensor_33 = torch.empty(B, C, H, 2*W).cuda()
        # ct_tensor_33[:, :, :, 0:W] = f3
        # ct_tensor_33[:, :, :, W:] = out4
        # p33 = self.st_block_33(ct_tensor_33)

        p3 = self.fuse_layer_3(torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1))
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)
       
        '''
            Stage III
        '''
        # F2=self.fusion2(pre_feat_2, post_feat_2)
        # out3 = F.interpolate(self.conv2(p3), scale_factor=2, mode='nearest')

        p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
        B, C, H, W = pre_feat_2.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_22 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_22[:, :, :, ::2] = pre_feat_2 # Odd columns
        ct_tensor_22[:, :, :, 1::2] =post_feat_2  # Even columns
        p22 = self.st_block_22(ct_tensor_22)

        ct_tensor_23 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_23[:, :, :, 0:W] = pre_feat_2
        ct_tensor_23[:, :, :, W:] = post_feat_2
        p23 = self.st_block_23(ct_tensor_23)

        # p21 = self.st_block_21(torch.cat([F2, out3], dim=1))
        # B, C, H, W = F2.size()
        # # Create an empty tensor of the correct shape (B, C, H, 2*W)
        # ct_tensor_22 = torch.empty(B, C, H, 2*W).cuda()
        # # Fill in odd columns with A and even columns with B
        # ct_tensor_22[:, :, :, ::2] = F2 # Odd columns
        # ct_tensor_22[:, :, :, 1::2] =out3  # Even columns
        # p22 = self.st_block_22(ct_tensor_22)

        # ct_tensor_23 = torch.empty(B, C, H, 2*W).cuda()
        # ct_tensor_23[:, :, :, 0:W] = F2
        # ct_tensor_23[:, :, :, W:] = out3
        # p23 = self.st_block_23(ct_tensor_23)

        p2 = self.fuse_layer_2(torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1))
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)
       
        '''
            Stage IV
        '''
        # F1 =self.fusion1(pre_feat_1, post_feat_1)
        # out2 = F.interpolate(self.conv1(p2), scale_factor=2, mode='nearest')
        p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
        B, C, H, W = pre_feat_1.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_12 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_12[:, :, :, ::2] = pre_feat_1  # Odd columns
        ct_tensor_12[:, :, :, 1::2] = post_feat_1  # Even columns
        p12 = self.st_block_12(ct_tensor_12)

        ct_tensor_13 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_13[:, :, :, 0:W] = pre_feat_1
        ct_tensor_13[:, :, :, W:] = post_feat_1
        p13 = self.st_block_13(ct_tensor_13)
        # p11 = self.st_block_11(torch.cat([F1, out2], dim=1))
        # B, C, H, W = F1.size()
        # # Create an empty tensor of the correct shape (B, C, H, 2*W)
        # ct_tensor_12 = torch.empty(B, C, H, 2*W).cuda()
        # # Fill in odd columns with A and even columns with B
        # ct_tensor_12[:, :, :, ::2] = F1  # Odd columns
        # ct_tensor_12[:, :, :, 1::2] = out2  # Even columns
        # p12 = self.st_block_12(ct_tensor_12)

        # ct_tensor_13 = torch.empty(B, C, H, 2*W).cuda()
        # ct_tensor_13[:, :, :, 0:W] = F1
        # ct_tensor_13[:, :, :, W:] = out2
        # p13 = self.st_block_13(ct_tensor_13)

        p1 = self.fuse_layer_1(torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1))

        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)

        return p1

   
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    """Basic block in decoder."""

    def __init__(self, encoder_dims):
        super().__init__()
        self.fusion4 = ChannelAttention_1(in_planes=encoder_dims[-1] )
        self.conv4 = nn.Conv2d(in_channels=encoder_dims[-1], out_channels=encoder_dims[-2], kernel_size=1) 
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.fusion3 = ChannelAttention_1(in_planes=encoder_dims[-2] )
        self.conv3 = nn.Conv2d(in_channels=encoder_dims[-1], out_channels=encoder_dims[-3], kernel_size=1) 
        self.fusion2 = ChannelAttention_1(in_planes=encoder_dims[-3] )
        self.conv2 = nn.Conv2d(in_channels=encoder_dims[-2], out_channels=encoder_dims[-4], kernel_size=1) 
        self.fusion1 = ChannelAttention_1(in_planes=encoder_dims[-4] )
        self.conv1 = nn.Conv2d(in_channels=encoder_dims[-3], out_channels=encoder_dims[-4], kernel_size=1) 
        self.fuse4 = nn.Sequential(nn.Conv2d(in_channels=encoder_dims[-1]*2 , out_channels=encoder_dims[-1],
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(encoder_dims[-1]),
                                  nn.ReLU(inplace=True),
                                  )

        self.fuse3 = nn.Sequential(nn.Conv2d(in_channels=encoder_dims[-1] , out_channels=encoder_dims[-2],
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(encoder_dims[-2]),
                                  nn.ReLU(inplace=True),
                                  )
        self.fuse2 = nn.Sequential(nn.Conv2d(in_channels=encoder_dims[-2] , out_channels=encoder_dims[-3],
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(encoder_dims[-3]),
                                  nn.ReLU(inplace=True),
                                  )
        self.fuse1 = nn.Sequential(nn.Conv2d(in_channels=encoder_dims[-3] , out_channels=encoder_dims[-4],
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(encoder_dims[-4]),
                                  nn.ReLU(inplace=True),
                                  )
        self.upsample_x4 = nn.Sequential(
            nn.Conv2d(in_channels=encoder_dims[-4], out_channels=encoder_dims[-4]//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(encoder_dims[-4]//2),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=encoder_dims[-4]//2, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out_change = nn.Conv2d(8, 2, kernel_size=7, stride=1, padding=3)

    def forward(self, pre_features, post_features):
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        # F4 =self.fusion4(pre_feat_4, post_feat_4)
        # F4 =self.up(self.conv4(F4))  #>>[4,1024,8,8]>>[4,512,8,8]>>[4,512,16,16]
        # F3 =self.fusion3(pre_feat_3, post_feat_3)  #[4,512,16,16]
        F4 = self.fusion4(torch.cat([pre_feat_4, post_feat_4], dim=1)) #>>[4,2048,8,8]
        M4_1 = self.fuse4(torch.cat([pre_feat_4, F4], dim=1))
        M4_2 = self.fuse4(torch.cat([F4,post_feat_4], dim=1))
        OUT4 = self.conv4(self.fuse4(torch.cat([M4_1,M4_2], dim=1)))  #>512,8
        # output3 = self.fuse3(output3)  #>>[4,256,16,16]

        # F2 =self.fusion2(pre_feat_2, post_feat_2) #[4,256,32,32]
        # output3 =self.up(output3)   #[4,256,16,16]>>[4,256,32,32]
        F3 = self.fusion3(torch.cat([pre_feat_3, post_feat_3], dim=1))
        M3_1 = self.fuse3(torch.cat([pre_feat_3, F3], dim=1))
        M3_2 = self.fuse3(torch.cat([F3,post_feat_3], dim=1))
        OUT3 = self.fuse3(torch.cat([M3_1,M3_2], dim=1))   #>512,16
        OUT4 = self.up(OUT4)   #>512,16
        output3 = self.conv3(torch.cat([OUT4,OUT3], dim=1))  #256,16

        F2 = self.fusion2(torch.cat([pre_feat_2, post_feat_2], dim=1))
        M2_1 = self.fuse2(torch.cat([pre_feat_2, F2], dim=1))
        M2_2 = self.fuse2(torch.cat([F2,post_feat_2], dim=1))
        OUT2 = self.fuse2(torch.cat([M2_1,M2_2], dim=1))   #256,32
        output3 = self.up(output3)  
        output2 = self.conv2(torch.cat([output3,OUT2], dim=1))  #128,32

        F1 = self.fusion1(torch.cat([pre_feat_1, post_feat_1], dim=1))
        M1_1 = self.fuse1(torch.cat([pre_feat_1, F1], dim=1))
        M1_2 = self.fuse1(torch.cat([F1,post_feat_1], dim=1))
        OUT1 = self.fuse1(torch.cat([M1_1,M1_2], dim=1))
        output2 = self.up(output2)
        output1 = self.conv1(torch.cat([output2,OUT1], dim=1))

        # F1 =self.fusion1(pre_feat_1, post_feat_1) #[4,128,64,64]
        # output2 =self.up(output2)  #>>[4,128,64,64]>>[4,128,64,64]
        # output1 = torch.cat([output2, F1], dim=1) #[4,256,64,64]
        # output1 = self.fuse1(output1)   #[4,128,64,64]
        out=self.upsample_x4(output1)
        output=self.conv_out_change(out)

        return output
