# CUDA_VISIBLE_DEVICES=0 python tools/train.py user_configs/kaist_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=1 python tools/train.py user_configs/kaist_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=2 python tools/train.py user_configs/kaist_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
CUDA_VISIBLE_DEVICES=0 python tools/train.py user_configs/adaptive_model_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
