_base_ = './kaist_faster-rcnn_r50_fpn_1x_coco.py'

custom_imports = dict(
    imports=['user_src.train_two_stream_pipeline', 
             'user_src.two_stream_data_preprocessor', 
             'user_src.adaptive_faster_rcnn', 
             'user_src.custom_hooks', 
             'user_src.custom_evaluator', 
             'user_src.custom_module'], 
    allow_failed_imports=False)

custom_hooks = [
    dict(type='AdjustModeHook')
]

work_dir = './work_dirs/adaptive_model_try/'

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
    type='AdaptiveModel')