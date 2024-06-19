auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
classes = 'person'
custom_hooks = [
    dict(type='AdjustModeHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'user_src.train_two_stream_pipeline',
        'user_src.two_stream_data_preprocessor',
        'user_src.adaptive_faster_rcnn',
        'user_src.custom_hooks',
        'user_src.custom_evaluator',
        'user_src.custom_module',
        'user_src.custom_loss',
    ])
data_root = 'data/VTUAV/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        rule='greater',
        save_best='coco/bbox_mAP_50',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True,
        score_thr=0.5,
        test_out_dir='vis',
        type='TwoStreamDetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_size = (
    960,
    540,
)
launcher = 'none'
load_from = 'work_dirs/adaptive_model_vtuav_selection9_faster-rcnn_r50_fpn_1x_coco/best_coco_bbox_mAP_50_epoch_9.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='TwoStreamDetDataPreprocessor'),
    fusion_layers=dict(
        fusion_pattern='AC3',
        num_outs=4,
        out_channels=[
            256,
            512,
            1024,
            2048,
        ],
        type='FusionLayer'),
    loss_policy=dict(loss_weight=0.001, type='PolicyLoss'),
    neck=dict(
        end_level=2,
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=3,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='SmoothL1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.41,
                1.0,
            ],
            scales=[
                2,
                8,
            ],
            strides=[
                4,
                8,
                16,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='AdaptiveModel')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=40,
        gamma=0.1,
        milestones=[
            8,
            17,
            28,
            36,
        ],
        type='MultiStepLR'),
]
randomness = dict(seed=1539225152)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='test_rgb/'),
        data_root='data/VTUAV/',
        metainfo=dict(classes='person'),
        pipeline=[
            dict(backend_args=None, type='LoadTwoStreamImageFromFiles'),
            dict(keep_ratio=True, scale=(
                960,
                540,
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
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/VTUAV/annotations/val.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    outfile_prefix=
    './work_dirs/adaptive_model_vtuav_selection9_faster-rcnn_r50_fpn_1x_coco/',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadTwoStreamImageFromFiles'),
    dict(keep_ratio=True, scale=(
        960,
        540,
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
train_cfg = dict(max_epochs=40, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/train.json',
        backend_args=None,
        data_prefix=dict(img='train_rgb/'),
        data_root='data/VTUAV/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(classes='person'),
        pipeline=[
            dict(backend_args=None, type='LoadTwoStreamImageFromFiles'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                n_patches=(
                    1,
                    5,
                ),
                ratio=(
                    0.02,
                    0.2,
                ),
                squared=False,
                type='RandomErasingTwoStream'),
            dict(keep_ratio=True, scale=(
                960,
                540,
            ), type='ResizeTwoStream'),
            dict(type='PackDetInputsTwoStream'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadTwoStreamImageFromFiles'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        n_patches=(
            1,
            5,
        ),
        ratio=(
            0.02,
            0.2,
        ),
        squared=False,
        type='RandomErasingTwoStream'),
    dict(keep_ratio=True, scale=(
        960,
        540,
    ), type='ResizeTwoStream'),
    dict(type='PackDetInputsTwoStream'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/val.json',
        backend_args=None,
        data_prefix=dict(img='val_rgb/'),
        data_root='data/VTUAV/',
        metainfo=dict(classes='person'),
        pipeline=[
            dict(backend_args=None, type='LoadTwoStreamImageFromFiles'),
            dict(keep_ratio=True, scale=(
                960,
                540,
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
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/VTUAV/annotations/val.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
val_pipeline = [
    dict(backend_args=None, type='LoadTwoStreamImageFromFiles'),
    dict(keep_ratio=True, scale=(
        960,
        540,
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
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/adaptive_model_vtuav_selection9_faster-rcnn_r50_fpn_1x_coco/'
