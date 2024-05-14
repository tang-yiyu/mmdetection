import copy
import warnings
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.base import BaseDetector

from user_src.custom_module import  PolicyNet, ChannelAttention, SpatialAttention

@MODELS.register_module()
class AdaptiveModel(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 feature_layers: ConfigType,
                 fusion_layers: ConfigType,
                 loss_policy:ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.backbone_rgb = MODELS.build(backbone)
        self.backbone_ir = MODELS.build(backbone)

        self.feature_rgb = MODELS.build(feature_layers)
        self.feature_ir = MODELS.build(feature_layers)

        self.policy_net = PolicyNet(in_channels=2560)

        self.fusion_layers = MODELS.build(fusion_layers)

        # self.attention_layers = []
        # for i in range(fusion_layers.num_outs):
        #     attention_layer = nn.Sequential(ChannelAttention(2*fusion_layers.out_channels[i]),
        #                                     SpatialAttention(),
        #                                     nn.Dropout(0.1))
        #     attention_layer_name = f'attention_layer{i}'
        #     self.add_module(attention_layer_name, attention_layer)
        #     self.attention_layers.append(attention_layer)

        if neck is not None:
            self.neck_rgb = MODELS.build(neck)
            self.neck_ir = MODELS.build(neck)
            self.neck_fuse = MODELS.build(neck)

        self.policy_layers = []
        for i in range(neck.num_outs):
            policy_layer = PolicyNet(in_channels=512, out_feature_c = 256)
            policy_layer_name = f'policy_layer{i}'
            self.add_module(policy_layer_name, policy_layer)
            self.policy_layers.append(policy_layer)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.loss_policy = MODELS.build(loss_policy)

        self.update_policy_net = True
        self.update_main_net = True

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                               bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck_rgb') and hasattr(self, 'neck_ir') and self.neck_rgb and self.neck_ir is not None

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def judge(self, decision: Tensor, input: Tensor) -> Tensor:
        """
        Determine whether to discard an input path.
        """
        output = torch.zeros_like(input)
        for i in range(input.shape[0]):
            output[i, :, :, :] = decision[i] * input[i, :, :, :]
        return output

    def compute_policy_loss(self, decisions_set, losses):
        policy_losses = self.loss_policy(decisions_set, losses)
        return dict(
            loss_policy=policy_losses)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            x_fuse: tuple[Tensor]: Multi-level features that may have
                different resolutions.
            x_rgb: tuple[Tensor]: Multi-level features that may have
                different resolutions.
            x_ir: tuple[Tensor]: Multi-level features that may have
                different resolutions.
            decisions: Tensor: exp: torch.tensor([[0., 1.], [1., 1.]]) for batch_size = 2
                or torch.tensor([[0., 1., 0., 1.], [1., 1., 0., 1.]]) for batch_size = 4
                decisions[0] represents RGB stream, decisions[1] represents IR stream.
        """
        batch_inputs_rgb = batch_inputs[:, :3, :, :]
        batch_inputs_ir = batch_inputs[:, 3:, :, :]

        features_rgb = self.feature_rgb(batch_inputs_rgb)
        features_ir = self.feature_ir(batch_inputs_ir)
        selections = self.policy_net(features_rgb[0], features_ir[0])
        batch_inputs_rgb = self.judge(selections[0], batch_inputs_rgb)
        batch_inputs_ir = self.judge(selections[1], batch_inputs_ir)
        # selection = torch.tensor([[0., 1.], [1., 1.]], device=batch_inputs_rgb.device)

        x_rgb = self.backbone_rgb(batch_inputs_rgb)
        x_ir = self.backbone_ir(batch_inputs_ir)

        if len(x_rgb) != len(x_ir):
            raise ValueError('The length of rgb feature and ir feature should be the same.')
        
        selections_set = []
        selections_set.append(selections)
        x_fuse = []
        for i in range(len(x_rgb)):
            out = torch.cat((x_rgb[i], x_ir[i]), dim=1)
            # out = self.attention_layers[i](out)
            out = self.fusion_layers(out, i)
            x_fuse.append(out)
        x_fuse = tuple(x_fuse)

        if self.with_neck:
            x_rgb = self.neck_rgb(x_rgb)
            x_ir = self.neck_ir(x_ir)
            x_fuse = self.neck_fuse(x_fuse)
        
        return x_rgb, x_ir, x_fuse, selections_set

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x_rgb, x_ir, x_fuse, _ = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x_fuse, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        x = []
        for i in range(len(x_rgb)):
            decisions = self.policy_layers[i](x_rgb[i], x_ir[i])
            out_rgb = self.judge(decisions[0], x_rgb[i])
            out_ir = self.judge(decisions[1], x_ir[i])
            out = out_rgb + out_ir
            x.append(out)
        x = tuple(x)
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.
           
        Use for train process.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x_rgb, x_ir, x_fuse, selection_set = self.extract_feat(batch_inputs)
        
        decisions_set = []
        for selections in selection_set:
            decisions_set.append(selections)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x_fuse, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        x = []
        for i in range(len(x_rgb)):
            decisions = self.policy_layers[i](x_rgb[i], x_ir[i])
            decisions_set.append(decisions)
            out_rgb = self.judge(decisions[0], x_rgb[i])
            out_ir = self.judge(decisions[1], x_ir[i])
            out = out_rgb + out_ir
            x.append(out)
        x = tuple(x)

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        if self.update_policy_net == True:
            policy_losses = self.compute_policy_loss(decisions_set, losses)
            losses.update(policy_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Use for val and test procedures which do not need to compute loss.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x_rgb, x_ir, x_fuse, _ = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x_fuse, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        x = []
        for i in range(len(x_rgb)):
            decisions = self.policy_layers[i](x_rgb[i], x_ir[i])
            out_rgb = self.judge(decisions[0], x_rgb[i])
            out_ir = self.judge(decisions[1], x_ir[i])
            out = out_rgb + out_ir
            x.append(out)
        x = tuple(x)

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
 