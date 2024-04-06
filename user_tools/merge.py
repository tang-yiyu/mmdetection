import os  
import shutil  
  
source_folder1 = '/data1/tyy/Sanitized/KAIST_visible/images/train/'  
source_folder2 = '/data1/tyy/Sanitized/KAIST_visible/images/test/'  

source_folders = [source_folder1, source_folder2]
 
destination_folder = '/data1/tyy/Sanitized/BMVC_coco/images'

for source_folder in source_folders:
    for root, dirs, files in os.walk(source_folder):  
        for file in files:  
            source_file = os.path.join(root, file)  
            shutil.copy(source_file, destination_folder)
            print(f"Copied {source_file} to {destination_folder}")


source_folder1 = '/data1/tyy/Sanitized/KAIST_visible/labels/train/'  
source_folder2 = '/data1/tyy/Sanitized/KAIST_visible/labels/test/'  

source_folders = [source_folder1, source_folder2]
 
destination_folder = '/data1/tyy/Sanitized/BMVC_coco/labels'

for source_folder in source_folders:
    for root, dirs, files in os.walk(source_folder):  
        for file in files:  
            source_file = os.path.join(root, file)  
            shutil.copy(source_file, destination_folder)
            print(f"Copied {source_file} to {destination_folder}")
   