_base_ = '../configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

custom_imports = dict(
    imports=['user_src.train_two_channel_pipeline', 'user_src.two_channel_data_preprocessor', 'user_src.two_channel_faster_rcnn', 'user_src.custom_hooks', 'user_src.custom_evaluator', 'user_src.custom_models'], allow_failed_imports=False)

image_size=(640, 640)
dataset_type = 'CocoDataset'
classes = ('person')
data_root='data/kaist_onlyperson/'
work_dir = './work_dirs/kaist_onlyperson_faster-rcnn_r50_fpn_1x_coco/'

default_hooks = dict(
    checkpoint=dict(interval=1, save_best='coco/bbox_mAP_50', rule='greater', type='CheckpointHook'),
    visualization=dict(score_thr=0.001, type='TwoChannelDetVisualizationHook'))

model = dict(
    fusion_layers=dict(
        type='FusionLayer',
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        base_channels=64,
        channel_weight=4,
        fusion_pattern='C3'),
    data_preprocessor=dict(
        type='TwoChannelDetDataPreprocessor'),
    neck=dict(
        num_outs=3,
        end_level=2),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1)),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.41
            ],
            scales=[
                8,
                11.3137,
            ],
            strides=[
                4,
                8,
                16,
            ],
            type='AnchorGenerator')),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.001)),
    type='TwoChannelFasterRCNN')

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone_rgb': dict(lr_mult=0.1, decay_mult=1.0),
            'backbone_ir': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=1)

train_pipeline = [
    dict(backend_args=None, type='LoadTwoChannelImageFromFiles'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.5, 2.0),
        resize_type='ResizeTwoChannel',
        keep_ratio=True),
    # dict(keep_ratio=True, scale=image_size, type='ResizeTwoChannel'),
    dict(
        type='RandomCropTwoChannel',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='RandomErasingTwoChannel', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(prob=0.5, type='RandomFlipTwoChannel'),
    dict(type='PackDetInputsTwoChannel'),
]

train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/train.json',
        backend_args=None,
        data_prefix=dict(img='train_rgb/'),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        # filter_cfg=dict(min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes),
        type='CocoDataset')
    )

val_evaluator = dict(
    ann_file='data/kaist_onlyperson/annotations/val.json')

val_pipeline = [
    dict(backend_args=None, type='LoadTwoChannelImageFromFiles'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='ResizeTwoChannel'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputsTwoChannel'),
]

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/val.json',
        backend_args=None,
        data_prefix=dict(img='val_rgb/'),
        data_root=data_root,
        pipeline=val_pipeline,
        test_mode=True,
        metainfo=dict(classes=classes),
        type='CocoDataset')
    )

test_evaluator = dict(
    ann_file='data/kaist_onlyperson/annotations/test.json',
    outfile_prefix=work_dir,
    type='CocoMetricMod')

test_pipeline = [
    dict(backend_args=None, type='LoadTwoChannelImageFromFiles'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='ResizeTwoChannel'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputsTwoChannel'),
]

test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='test_rgb/'),
        data_root=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=dict(classes=classes),
        type='CocoDataset')
    )
