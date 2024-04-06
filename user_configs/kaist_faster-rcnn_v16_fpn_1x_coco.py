_base_ = './kaist_faster-rcnn_r50_fpn_1x_coco.py'

work_dir = './work_dirs/kaist_bmvc_faster-rcnn_v16_5_fpn_1x_coco/'

model = dict(
    fusion_layers=dict(
        _delete_=True,
        type='FusionLayer',
        num_stages=5,
        out_indices=(0, 1, 2, 3, 4),
        base_channels=64,
        channel_weight=1,
        fusion_pattern='Cross'),
    backbone=dict(
        _delete_=True,
        depth=16,
        frozen_stages=1,
        with_bn=True,
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://vgg16_bn'),
        num_stages=5,
        out_indices=(
            0,
            1,
            2,
            3,
            4,
        ),
        ceil_mode=True,
        type='VGG16'),
    neck=dict(
        in_channels=[
            64,
            128,
            256,
            512,
            512,
        ]))
test_evaluator = dict(
    outfile_prefix=work_dir)