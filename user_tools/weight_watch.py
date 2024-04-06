import torch

pthfile = r'/data1/tyy/mmdetection/work_dirs/kaist_nounpaired_faster-rcnn_r50_fpn_1x_coco/epoch_1.pth'
model = torch.load(pthfile)
# output_file = "output.txt"
# with open(output_file, "w") as f:
#     f.write('key:\n')
#     for k in model["state_dict"].keys():
#         f.write(k)
#     f.write('value:\n')
#     for k in model["state_dict"]:
#         f.write(k)
#         f.write(str(model["state_dict"][k]))
#         f.write("\n")
rgb_ir = []
for k in model["state_dict"]:
    if k.startswith('backbone_rgb.layer3.1.bn3.weight'):
        rgb_ir.append(model["state_dict"][k])
        print(k, model["state_dict"][k])
    if k.startswith('backbone_ir.layer3.1.bn3.weight'):
        rgb_ir.append(model["state_dict"][k])
        print(k, model["state_dict"][k])
# print(rgb_ir[0])
# print(rgb_ir[1])
if all(torch.eq(rgb_ir[0], rgb_ir[1])):
    print('rgb == ir')
else:
    print('rgb != ir')
    
    
