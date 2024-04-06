import warnings
import torch
import torch.nn as nn
from torch import Tensor

from typing import Tuple

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.two_stage import TwoStageDetector

from user_src.custom_models import FusionLayer, PolcyNet

from mmdet.models.backbones.mobilenet_v2 import MobileNetV2

        
@MODELS.register_module()
class AdaptiveTwoStageDetector(TwoStageDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """
    def __init__(self, backbone: ConfigType, **kwargs):
                 
        super().__init__(backbone=backbone, **kwargs)
        
        del self.backbone
        self.backbone_rgb = MODELS.build(backbone)
        self.backbone_ir = MODELS.build(backbone)

        self.feature_rgb = MobileNetV2(out_indices=(8))
        self.feature_ir = MobileNetV2(out_indices=(8))

        self.policy_net = PolcyNet(in_channels=2048)

        if backbone['type'] == 'ResNet':
            self.fusion_layers = FusionLayer(num_stages=4, out_indices=(0, 1, 2, 3), base_channels=64, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), channel_weight=4)
        elif backbone['type'] == 'VGG16':
            self.fusion_layers = FusionLayer(num_stages=5, out_indices=(1, 2, 3, 4), base_channels=64, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), channel_weight=1)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        batch_inputs_rgb = batch_inputs[:, :3, :, :]
        batch_inputs_ir = batch_inputs[:, 3:, :, :]

        features_rgb = self.feature_rgb(batch_inputs_rgb)
        features_ir = self.feature_ir(batch_inputs_ir)

        decisions = self.policy_net(features_rgb, features_ir)

        batch_inputs_rgb = self.judge(batch_inputs_rgb, decisions[0])
        batch_inputs_ir = self.judge(batch_inputs_ir, decisions[1])

        x_rgb = self.backbone_rgb(batch_inputs_rgb)
        x_ir = self.backbone_ir(batch_inputs_ir)

        if len(x_rgb) != len(x_ir):
            raise ValueError('The length of rgb feature and ir feature should be the same.')

        x = []
        for i in range(len(x_rgb)):
            out = torch.cat((x_rgb[i], x_ir[i]), dim=1)
            out = self.fusion_layers(out, i)
            x.append(out)
        x = tuple(x)

        if self.with_neck:
            x = self.neck(x)
        return x
    
    def judge(input, decision):
        """
        Determine whether to discard an input path.
        """
        for i in range(input.shape[0]):
            input[i, :, :, :] = decision[i] * input[i, :, :, :].clone()
        return input


@MODELS.register_module()
class AdaptiveFasterRCNN(AdaptiveTwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)