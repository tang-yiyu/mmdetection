import warnings
from typing import Sequence

from mmdet.registry import MODELS
from mmcv.cnn import VGG
from mmengine.model import BaseModule

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