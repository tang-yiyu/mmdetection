# CUDA_VISIBLE_DEVICES=0 python tools/train.py user_configs/kaist_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=1 python tools/train.py user_configs/kaist_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=3 python tools/train.py user_configs/kaist_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=0 python tools/train.py user_configs/adaptive_model_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=2 python tools/train.py user_configs/droneperson_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=0 python tools/train.py user_configs/adaptive_model_droneperson_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=1 python tools/train.py user_configs/vtuav_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=1 python tools/train.py user_configs/adaptive_model_vtuav_faster-rcnn_r50_fpn_1x_coco.py --auto-scale-lr


# CUDA_VISIBLE_DEVICES=0 python tools/train.py compare_method/diffusiondet_twostream_vtuav.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=0 python tools/train.py user_configs/adaptive_model_droneperson_gfl.py --auto-scale-lr

# Faster R-CNN
# CUDA_VISIBLE_DEVICES=0 python tools/train.py compare_method/faster-rcnn_droneperson_rgb.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=0 python tools/train.py compare_method/faster-rcnn_droneperson_ir.py --auto-scale-lr

# ATSS
# CUDA_VISIBLE_DEVICES=1 python tools/train.py compare_method/ATSS_droneperson_rgb.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=1 python tools/train.py compare_method/ATSS_droneperson_ir.py --auto-scale-lr

# FCOS
# CUDA_VISIBLE_DEVICES=2 python tools/train.py compare_method/FCOS_droneperson_rgb.py --auto-scale-lr
# CUDA_VISIBLE_DEVICES=2 python tools/train.py compare_method/FCOS_droneperson_ir.py --auto-scale-lr

# GFL
# CUDA_VISIBLE_DEVICES=3 python tools/train.py compare_method/GFL_droneperson_rgb.py --auto-scale-lr
CUDA_VISIBLE_DEVICES=3 python tools/train.py compare_method/GFL_droneperson_ir.py --auto-scale-lr
