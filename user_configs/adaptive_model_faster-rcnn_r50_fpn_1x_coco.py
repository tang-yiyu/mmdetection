_base_ = './kaist_faster-rcnn_r50_fpn_1x_coco.py'

custom_imports = dict(
    imports=['user_src.train_two_stream_pipeline', 
             'user_src.two_stream_data_preprocessor', 
             'user_src.adaptive_faster_rcnn', 
             'user_src.custom_hooks', 
             'user_src.custom_evaluator', 
             'user_src.custom_module',
             'user_src.custom_loss'], 
    allow_failed_imports=False)

custom_hooks = [
    dict(type='AdjustModeHook')
]

work_dir = './work_dirs/adaptive_model_loss001_kaist_onlyperson/'
randomness = dict(seed=1907893643) # onlyperson
# randomness = dict(seed=1531503327) # BMVC_Mod

train_dataloader = dict(
    batch_size=2,)

test_evaluator = dict(
    outfile_prefix=work_dir)

model = dict(
    feature_layers=dict(
        type='MobileNetV2',
        out_indices=(7,),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2'),
    ),
    loss_policy=dict(loss_weight=0.01, type='PolicyLoss'),
    type='AdaptiveModel')

optim_wrapper = dict(
    # optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2),
    type='OptimWrapper')

param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=40,
        gamma=0.1,
        milestones=[
            5,
            11,
            17,
            23,
            29,
            35,
        ],
        type='MultiStepLR'),
]

train_cfg = dict(max_epochs=40, type='EpochBasedTrainLoop', val_interval=1)