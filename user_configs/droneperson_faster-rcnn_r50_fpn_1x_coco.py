_base_ = '../configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

custom_imports = dict(
    imports=['user_src.train_two_stream_pipeline', 
             'user_src.two_stream_data_preprocessor', 
             'user_src.two_stream_faster_rcnn', 
             'user_src.custom_hooks', 
             'user_src.custom_evaluator', 
             'user_src.custom_module',
             'user_src.custom_visualization',], 
    allow_failed_imports=False)

image_size=(640, 512)
dataset_type = 'CocoDataset'
classes = ('person', 'rider', 'crowd')
data_root = 'data/DronePerson/'
work_dir = './work_dirs/droneperson_baseline_new2_faster-rcnn_r50_fpn_1x_coco/'
# randomness = dict(seed=608238547)
# randomness = dict(seed=0)

default_hooks = dict(
    checkpoint=dict(interval=1, save_best='coco/bbox_mAP_50', rule='greater', type='CheckpointHook'),
    visualization=dict(score_thr=0.5, type='TwoStreamDetVisualizationHook'))

model = dict(
    data_preprocessor=dict(
        type='TwoStreamDetDataPreprocessor'),
    neck=dict(
        num_outs=3,
        end_level=2),
    fusion_layers=dict(
        type='FusionLayer',
        num_outs=4,
        out_channels=[256, 512, 1024, 2048],
        # out_channels=[256, 256, 256, 256],
        fusion_pattern='AC3'),
    roi_head=dict(
        bbox_roi_extractor=dict(featmap_strides=[4, 8, 16]),
        bbox_head=dict(
            loss_bbox=dict(loss_weight=1.0, type='SmoothL1Loss'),
            num_classes=3)),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.41,
                1.0,
            ],
            scales=[
                1,
                2,
                4,
                8,
            ],
            strides=[
                4,
                8,
                16,
            ],
            type='AnchorGenerator'),
        loss_bbox=dict(loss_weight=1.0, type='SmoothL1Loss')),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.3, type='nms'),
            score_thr=0.05)),
    type='TwoStreamFasterRCNN')

# optim_wrapper = dict(
#     optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
#     type='OptimWrapper')

train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=1)

train_pipeline = [
    dict(backend_args=None, type='LoadTwoStreamImageFromFiles'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(keep_ratio=True, scale=image_size, type='ResizeTwoStream'),
    # dict(
    #     type='RandomCropTwoStream',
    #     crop_type='absolute_range',
    #     crop_size=(384, 480),
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    dict(type='RandomErasingTwoStream', n_patches=(1, 5), ratio=(0.02, 0.2), squared=False),
    dict(keep_ratio=True, scale=image_size, type='ResizeTwoStream'),
    # dict(
    #     type='RandomResize',
    #     scale=image_size,
    #     ratio_range=(0.5, 2.0),
    #     resize_type='ResizeTwoStream',
    #     keep_ratio=True),
    # dict(prob=0.5, type='RandomFlipTwoStream'),
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
    ann_file='data/DronePerson/annotations/val.json',
    classwise=True,
    type='DronePerson')

val_pipeline = [
    dict(backend_args=None, type='LoadTwoStreamImageFromFiles'),
    dict(keep_ratio=True, scale=image_size, type='ResizeTwoStream'),
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
    ann_file='data/DronePerson/annotations/val.json',
    outfile_prefix=work_dir,
    classwise=True,
    type='DronePerson')

test_pipeline = [
    dict(backend_args=None, type='LoadTwoStreamImageFromFiles'),
    dict(keep_ratio=True, scale=image_size, type='ResizeTwoStream'),
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
    batch_size=2,
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

visualizer = dict(
    name='visualizer',
    type='TwoStreamDetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
