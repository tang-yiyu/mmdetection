import os  
  
# train 
train_folder_path = '/data1/tyy/Sanitized/KAIST_visible/images/train/'  
output_file_train = '/data1/tyy/Sanitized/BMVC_coco/train.txt'  
file_names = os.listdir(train_folder_path)
file_names.sort()
with open(output_file_train, 'w') as f:
    for file_name in file_names:  
        f.write(file_name + '\n')
        print(file_name, ' Done!')

# val
val_folder_path = '/data1/tyy/Sanitized/KAIST_visible/images/test/'   
output_file_val = '/data1/tyy/Sanitized/BMVC_coco/val.txt'  
file_names = os.listdir(val_folder_path)
file_names.sort()
with open(output_file_val, 'w') as f:
    for file_name in file_names:  
        f.write(file_name + '\n')
        print(file_name, ' Done!')


# test
test_folder_path = '/data1/tyy/Sanitized/KAIST_visible/images/test/'   
output_file_test = '/data1/tyy/Sanitized/BMVC_coco/test.txt'  
file_names = os.listdir(test_folder_path)
file_names.sort()
with open(output_file_test, 'w') as f:
    for file_name in file_names:  
        f.write(file_name + '\n')
        print(file_name, ' Done!')