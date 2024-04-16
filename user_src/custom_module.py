import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmcv.cnn import ConvModule

class PolicyNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_feature_c=2048,
                 dimension=1
                 ):
        super().__init__()

        self.out_feature_c = out_feature_c
        self.temperature = 5.0
        self.decay_ratio = 0.965
        self.d = dimension
        self.layers = []

        avgpool_rgb = nn.AdaptiveAvgPool2d((1, 1))
        self.add_module('avgpool_rgb', avgpool_rgb)
        self.layers.append(avgpool_rgb)

        avgpool_ir = nn.AdaptiveAvgPool2d((1, 1))
        self.add_module('avgpool_ir', avgpool_ir)
        self.layers.append(avgpool_ir)
        
        joint_net = nn.Sequential(
            nn.Linear(in_channels, self.out_feature_c), nn.ReLU(True),
            nn.Linear(self.out_feature_c, self.out_feature_c), nn.ReLU(True)
        )
        self.add_module('joint_net', joint_net)
        self.layers.append(joint_net)

        fc_rgb = nn.Linear(self.out_feature_c, 2)
        self.add_module('fc_rgb', fc_rgb)
        self.layers.append(fc_rgb)

        fc_ir = nn.Linear(self.out_feature_c, 2)
        self.add_module('fc_ir', fc_ir)
        self.layers.append(fc_ir)
        
    def wrapper_gumbel_softmax(self, logits):
        """
        :param logits: NxM, N is batch size, M is number of possible choices
        :return: Nx1: the selected index
        """
        distributions = F.gumbel_softmax(logits, tau=self.temperature, hard=True) # If hard=True, returns a one-hot vector
        decisions = distributions[:, -1]
        return decisions
    
    def forward(self, x_rgb, x_ir):
        x_rgb = self.layers[0](x_rgb)
        x_rgb = x_rgb.view(x_rgb.size(0), x_rgb.size(1), -1)

        x_ir = self.layers[1](x_ir)
        x_ir = x_ir.view(x_ir.size(0), x_rgb.size(1), -1)

        x = torch.cat([x_rgb, x_ir], dim=self.d) # 将将x列表中的所有张量沿着列连接起来（通道数相加）
        x = x.view(x.size(0), -1) # 将张量的形状进行reshape，便于后续全连接层的处理
        x = self.layers[2](x) # 全连接层

        logits = [self.layers[3](x), self.layers[4](x)]
        logits = torch.cat(logits, dim=0)
        # print("Current temperature: {}".format(self.temperature), flush=True)
        decisions = self.wrapper_gumbel_softmax(logits)
        decisions = decisions.view(2, -1) # Modality x Batchsize
        
        return decisions
    
    def set_temperature(self, temperature):
        self.temperature = temperature

    def decay_temperature(self, decay_ratio=None):
        dr = decay_ratio if decay_ratio else self.decay_ratio
        if dr:
            self.temperature *= dr 

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * e)  # hidden channels
        self.conv1 = ConvModule(in_channels=in_channels,
                              out_channels=hidden_channels,
                              kernel_size=1,
                              norm_cfg=dict(type='BN', requires_grad=True),
                              act_cfg=dict(type='SiLU', inplace=True))
        self.conv2 = ConvModule(in_channels=hidden_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=1,
                              norm_cfg=dict(type='BN', requires_grad=True),
                              act_cfg=dict(type='SiLU', inplace=True))
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * e)  # hidden channels
        self.conv1 = ConvModule(in_channels=in_channels,
                              out_channels=hidden_channels,
                              kernel_size=1,
                              norm_cfg=dict(type='BN', requires_grad=True),
                              act_cfg=dict(type='SiLU', inplace=True))
        self.conv2 = ConvModule(in_channels=in_channels,
                              out_channels=hidden_channels,
                              kernel_size=1,
                              norm_cfg=dict(type='BN', requires_grad=True),
                              act_cfg=dict(type='SiLU', inplace=True))
        self.conv3 = ConvModule(2 * hidden_channels, out_channels, 1)  # optional act=FReLU(out_channels)
        self.m = nn.Sequential(*(Bottleneck(hidden_channels, hidden_channels, shortcut, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), 1))

class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, in_channels, out_channels, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        hidden_channels = int(out_channels * e)  # hidden channels
        self.conv1 = ConvModule(in_channels=in_channels,
                              out_channels=hidden_channels,
                              kernel_size=[1, 3],
                              stride=[1, 1],
                              norm_cfg=dict(type='BN', requires_grad=True),
                              act_cfg=dict(type='SiLU', inplace=True))
        self.conv2 = ConvModule(in_channels=hidden_channels,
                              out_channels=out_channels,
                              kernel_size=[3, 1],
                              stride=[1, 1],
                              norm_cfg=dict(type='BN', requires_grad=True),
                              act_cfg=dict(type='SiLU', inplace=True))
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

@MODELS.register_module()
class FusionLayer(nn.Module):
    def __init__(self,
                 num_outs = 3,
                 out_channels = 256,
                 fusion_pattern='C3'):
        super().__init__()
        self.num_outs = num_outs
        self.fusion_layers = []
        for i in range(self.num_outs):
            if fusion_pattern == 'Conv':
                fusion_layer = ConvModule(
                    in_channels=int(out_channels * 2),
                    out_channels=out_channels,
                    kernel_size=1)
            elif fusion_pattern == 'ConvConv':
                fusion_layer = CrossConv(
                    in_channels=int(out_channels * 2),
                    out_channels=out_channels,
                    shortcut=False,
                    e=1.5)
            elif fusion_pattern == 'Cross':
                fusion_layer = CrossConv(
                    in_channels=int(out_channels * 2),
                    out_channels=out_channels,
                    shortcut=True,
                    e=1.0)
            elif fusion_pattern == 'C3':
                fusion_layer = C3(
                    in_channels=int(out_channels * 2),
                    out_channels=out_channels,
                    n=1,
                    shortcut=True,
                    e=0.5)
            else:
                raise NotImplementedError
            fusion_layer_name = f'layer{i + 1}_fusion'
            self.add_module(fusion_layer_name, fusion_layer)
            self.fusion_layers.append(fusion_layer)
        
    def forward(self, x, index):
        x = self.fusion_layers[index](x)
        return x
    

