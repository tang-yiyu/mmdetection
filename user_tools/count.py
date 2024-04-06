import json  
  
# 读取COCO数据集的JSON文件  
json_file_path = '/data1/tyy/mmdetection/data/BMVC/annotations/test.json'  
  
with open(json_file_path, 'r') as file:  
    coco_data = json.load(file)  
  
# 获取图像信息列表  
images_list = coco_data['images']  
  
# 计算图像数量  
num_images = len(images_list)  
  
# 打印图像数量  
print(f"The COCO dataset contains {num_images} images.")  