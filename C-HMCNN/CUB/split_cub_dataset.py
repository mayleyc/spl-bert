from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
from random import shuffle
import os
import shutil
from shutil import copyfile

# Change the path to your dataset folder:
base_folder = 'CUB_200_2011/'

# These path should be fine
images_txt_path = base_folder+ 'images.txt'
train_test_split_path =  base_folder+ 'train_test_split.txt'
images_path =  base_folder+ 'images/'

# Here declare where you want to place the train/test folders
# You don't need to create them!
test_folder = 'CUB/test/'
val_folder = 'CUB/val/'
train_folder = 'CUB/train/'


def ignore_files(dir,files): return [f for f in files if os.path.isfile(os.path.join(dir,f))]

shutil.copytree(images_path,test_folder,ignore=ignore_files)
shutil.copytree(images_path,val_folder,ignore=ignore_files)
shutil.copytree(images_path,train_folder,ignore=ignore_files)

with open(images_txt_path) as f:
  images_lines = f.readlines()

with open(train_test_split_path) as f:
  split_lines = f.readlines()

train_images = 0
test_val_images = []

for image_line,split_line in zip(images_lines,split_lines):

  image_line = (image_line.strip()).split(' ')
  split_line = (split_line.strip()).split(' ')

  image = plt.imread(images_path + image_line[1])

  # Use only RGB images, avoid grayscale
  if len(image.shape) == 3:

    # If train image
    if(int(split_line[1]) == 1):
      copyfile(images_path+image_line[1],train_folder+image_line[1])
      train_images += 1 
      
      #if test or val
    else:
      
      test_val_images.append(image_line[1])
    
shuffle(test_val_images)
val_images_list = test_val_images[:1158]
test_images_list = test_val_images[1158:]

for image_name in val_images_list:
  copyfile(images_path+image_name,val_folder+image_name)

for image_name in test_images_list:
  copyfile(images_path+image_name,test_folder+image_name)

print(train_images, len(test_images_list), len(val_images_list))
assert train_images == 5990
assert len(val_images_list) == 1158 #10% for validation
assert len(test_images_list) == 4632

print('Dataset successfully splitted!')
