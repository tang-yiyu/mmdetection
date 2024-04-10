_base_ = '../configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

custom_imports = dict(
    imports=['user_src.train_two_channel_pipeline', 'user_src.two_channel_data_preprocessor', 'user_src.two_channel_faster_rcnn', 'user_src.custom_hooks', 'user_src.custom_evaluator', 'user_src.custom_models'], allow_failed_imports=False)

image_size=(640, 640)
dataset_type = 'CocoDataset'
classes = ('person')
data_root='data/kaist_onlyperson/'
work_dir = './work_dirs/kaist_onlyperson_C3_faster-rcnn_r50_fpn_1x_coco2/'

default_hooks = dict(
    checkpoint=dict(interval=1, save_best='coco/bbox_mAP_50', rule='greater', type='CheckpointHook'),
    visualization=dict(type='TwoStreamDetVisualizationHook'))

model = dict(
    fusion_layers=dict(
        type='FusionLayer',
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        base_channels=64,
        channel_weight=4,
        fusion_pattern='C3'),
    data_preprocessor=dict(
        type='TwoStreamDetDataPreprocessor'),
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
    type='TwoStreamFasterRCNN')

train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=1)

train_pipeline = [
    dict(backend_args=None, type='LoadTwoStreamImageFromFiles'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.5, 2.0),
        resize_type='ResizeTwoStream',
        keep_ratio=True),
    # dict(keep_ratio=True, scale=image_size, type='ResizeTwoStream'),
    dict(
        type='RandomCropTwoStream',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='RandomErasingTwoStream', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(prob=0.5, type='RandomFlipTwoStream'),
    dict(type='PackDetInputsTwoStream'),
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
    dict(backend_args=None, type='LoadTwoStreamImageFromFiles'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='ResizeTwoStream'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputsTwoStream'),
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
    dict(backend_args=None, type='LoadTwoStreamImageFromFiles'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='ResizeTwoStream'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputsTwoStream'),
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
