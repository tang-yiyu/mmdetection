import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.models.backbones.resnet import ResNet
from mmcv.cnn import VGG, ConvModule
from mmengine.model import BaseModule



class PolcyNet(nn.Module):
    def __init__(self,
                 in_channels=2560,
                 dimension=1
                 ):
        super().__init__()

        self.out_feature_c = 2048
        self.temperature = 5.0
        self.d = dimension
        self.layers = []
        
        joint_net = nn.Sequential(
            nn.Linear(in_channels, self.out_feature_c), nn.ReLU(True),
            nn.Linear(self.out_feature_c, self.out_feature_c), nn.ReLU(True)
        )
        self.add_module('joint_net', joint_net)
        self.layers.append('joint_net')

        fc_rgb = nn.Linear(self.out_feature_c, 2)
        self.add_module('fc_rgb', fc_rgb)
        self.layers.append('fc_rgb')

        fc_ir = nn.Linear(self.out_feature_c, 2)
        self.add_module('fc_ir', fc_ir)
        self.layers.append('fc_ir')
        
    def wrapper_gumbel_softmax(self, logits):
        """
        :param logits: NxM, N is batch size, M is number of possible choices
        :return: Nx1: the selected index
        """
        distributions = F.gumbel_softmax(logits, tau=self.temperature, hard=True) # If hard=True, returns a one-hot vector
        decisions = distributions[:, -1]
        return decisions
    
    def forward(self, x_rgb, x_ir):
        x = torch.cat([x_rgb, x_ir], dim=self.d) # 将将x列表中的所有张量沿着列连接起来（通道数相加）
        x = x.view(x.size(0), -1) # 将张量的形状进行reshape，便于后续全连接层的处理
        x = self.layers['joint_net'](x) # 全连接层

        logits = [self.layers['fc_rgb'](x), self.layers['fc_ir'](x)]
        logits = torch.cat(logits, dim=0)
        # print("Current temperature: {}".format(self.temperature), flush=True)
        decisions = self.wrapper_gumbel_softmax(logits)
        decisions = decisions.view(2, -1) # Modality x Batchsize
        
        return decisions


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
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 base_channels=64,
                 channel_weight=4,
                 fusion_pattern='C3'):
        super().__init__()
        self.num_stages = num_stages
        self.base_channels = base_channels
        self.channel_weight = channel_weight
        self.fusion_layers = []
        for i in range(self.num_stages):
            channels = self.base_channels * 2**i  if i < 4 else 512
            if i in out_indices:
                if fusion_pattern == 'Conv':
                    fusion_layer = ConvModule(in_channels=int(channels * self.channel_weight * 2),
                              out_channels=int(channels * self.channel_weight),
                              kernel_size=1)
                elif fusion_pattern == 'ConvConv':
                    fusion_layer = CrossConv(
                        in_channels=int(channels * self.channel_weight * 2),
                        out_channels=int(channels * self.channel_weight),
                        shortcut=False,
                        e=1.5)
                elif fusion_pattern == 'Cross':
                    fusion_layer = CrossConv(
                        in_channels=int(channels * self.channel_weight * 2),
                        out_channels=int(channels * self.channel_weight),
                        shortcut=True,
                        e=1.0)
                elif fusion_pattern == 'C3':
                    fusion_layer = C3(
                        in_channels=int(channels * self.channel_weight * 2),
                        out_channels=int(channels * self.channel_weight),
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
    

@MODELS.register_module()
class VGG16(VGG, BaseModule):
    """
    Notice: If with_bn=False, checkpoint='torchvision://vgg16'. If with_bn=True, checkpoint='torchvision://vgg16_bn'.
    Reference website: https://mmclassification.readthedocs.io/en/latest/model_zoo.html#imagenet
    """
    def __init__(self,
                 depth: int,
                 with_bn: bool = False,
                 num_classes: int = -1,
                 num_stages: int = 5,
                 dilations: Sequence[int] = (1, 1, 1, 1, 1),
                 out_indices: Sequence[int] = (0, 1, 2, 3, 4),
                 frozen_stages: int = -1,
                 bn_eval: bool = True,
                 bn_frozen: bool = False,
                 ceil_mode: bool = False,
                 with_last_pool: bool = True,
                 pretrained=None,
                 init_cfg=None,
                 ):
        super(VGG16, self).__init__(depth=depth,
                                    with_bn=with_bn,
                                    num_classes=num_classes,
                                    num_stages=num_stages,
                                    dilations=dilations,
                                    out_indices=out_indices,
                                    frozen_stages=frozen_stages,
                                    bn_eval=bn_eval,
                                    bn_frozen=bn_frozen,
                                    ceil_mode=ceil_mode,
                                    with_last_pool=with_last_pool)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'

        if init_cfg is not None:
            self.init_cfg = init_cfg
        elif isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(type='Constant', val=1, layer='BatchNorm2d'),
                dict(type='Normal', std=0.01, layer='Linear'),
            ]
        else:
            raise TypeError('pretrained must be a str or None')
    
    def init_weights(self, pretrained=None):
        super(VGG, self).init_weights()


