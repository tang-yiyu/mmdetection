# CUDA_VISIBLE_DEVICES=0 python tools/train.py user_configs/kaist_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=1 python tools/train.py user_configs/kaist_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=2 python tools/train.py user_configs/kaist_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=0 python tools/train.py user_configs/adaptive_model_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
CONFIG_FILE=user_configs/adaptive_model_faster-rcnn_r50_fpn_1x_coco.py
GPU_NUM=2
GPUS_PER_NODE=2
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    --auto-scale-lr