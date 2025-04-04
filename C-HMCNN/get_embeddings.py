import os
import datetime
import json
from time import perf_counter
import copy
import shutil
import glob
import pickle
from random import randint
from pathlib import Path

import numpy as np

from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import os
os.environ['TORCH_HOME'] = '/mnt/cimec-storage6/users/nguyenanhthu.tran/torch'

# misc
from common import *

#Input: CUB raw dataset with each folder as class name
#Output: the same folder tree in a different location, each folder containing an .npy embedding file
#using ResNet50 pretrained on ImageNet

#CUB modified to take string labels
class CUB_Dataset_Modified(Dataset):
    def __init__(self, image_paths, labels, class_to_name, transform = None, to_eval = True):
        """
        Args:
            image_paths (list): List of image file paths.
            transform (callable, optional): transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_name = class_to_name
        self.transform = transform
        self.to_eval = to_eval
    def __len__(self):
        """
        Returns dataset size.
        """

        return len(self.image_paths)

    def process_image(self, img_path):
        """
        Load, transform and pad an image.
        """
        # load image
        image_pil = Image.open(img_path).convert("RGB")  # load image

        if self.transform:
            image = self.transform(image_pil)  # 3, h, w
        else:
            transform = T.Compose(
            [
                T.Resize((224, 224)),  # images will be morphed
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
            image = transform(image_pil)  # 3, H, W
        # Get current height and width
        _, H, W = image.shape
        '''
        # Compute padding
       
        pad_w = max(400 - W, 0)  # Only pads if W < 1333
        pad_h = max(400 - H, 0)    # Only pads if H < 800

        # Compute symmetric padding
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Apply padding correctly: (left, top, right, bottom)
        padded_image = F.pad(image, [pad_left, pad_top, pad_right, pad_bottom])  # Padding with 0s
'''
        return image_pil, image, image.shape  #image = tensor
    
    def __getitem__(self, idx):
        """
        Get and process a sample given index `idx`.
        """
        img_path = self.image_paths[idx]
        label_set = self.labels[idx]
        _, image, _ = self.process_image(img_path)

        class_name = self.class_to_name[int(label_set)]  # Convert index to class name
        return img_path, image, class_name  # Return class name instead of tensor

args = parse_args()

device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
print(f'Using {device} for inference')

model_name = "resnet50"
dataset_name = args.dataset

resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

#load the whole dataset
#Try out with 5 classes in CUB
# Get all class folder names
all_classes = sorted(os.listdir(images_dir))  # Sorting ensures consistency

# To use the CUB mini dataset, enter the dataset as cub_mini. For full CUB, use cub_others
if "mini" in args.dataset:
# Select all classes or only the first 5 classes (mini csv, mat and image set)
    selected_classes = all_classes[:5]
    csv_path = csv_path_mini
    mat_path = mat_path_mini
else:
    selected_classes = all_classes # 200 classes
    csv_path = csv_path_full
    mat_path = mat_path_full

# Get image paths only for selected classes
image_paths = []
for cls in selected_classes:
    class_images = glob.glob(os.path.join(images_dir, cls, "*.jpg"))
    image_paths.extend(class_images)

#get unprocessed labels (folder names)
labels_unprocessed = [os.path.basename(os.path.dirname(path)) for path in image_paths]

# Create a mapping from numeric index to class name
class_to_name = {i: class_name for i, class_name in enumerate(sorted(set(labels_unprocessed)))}

# Create a mapping from numeric index to class name (optional, if needed)
name_to_class = {class_name: i for i, class_name in class_to_name.items()}

# Convert labels_unprocessed to numeric indices for the dataset
labels_numeric = [name_to_class[label] for label in labels_unprocessed]

#Create dataset and dataloader
dataset = CUB_Dataset_Modified(image_paths, labels_numeric, class_to_name, transform = None, to_eval = True)
dataset.to_eval = torch.tensor(dataset.to_eval, dtype=torch.bool)

loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)

# Run inference and retrieve embeddings
all_labels = []
all_paths = []
all_embeddings = []

torch.cuda.empty_cache()
resnet50.eval().to(device)

print('Extracting embeddings...')
for path, data, labels in tqdm(loader):
    new_labels = list(labels)
    paths = list(path)
    all_labels += new_labels
    all_paths += paths
    emb = resnet50(data.cuda()).detach().cpu().numpy() # embeddings per batch. output: numpy array
    all_embeddings += emb.tolist() # not append(), it becomes 1 single element!

'''
for x, y in zip(embeddings, all_labels):
    all_embeddings[x] = y
'''
print('Embeddings extracted.')
print('Saving dictionary to pickle...')
#create new directory tree at /embeddings

date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
fp_dest = f"embeddings/embeddings_{model_name}_{dataset_name}_{date_string}.pickle"
if Path("./embeddings").exists():
    pass
else:
    os.makedirs("./embeddings")

# option 1: 1 file with embeddings & 1 with labels
# option 2: use a pickle to save it as a dictionary
with open(fp_dest, 'wb') as f:
    pickle.dump([all_paths, all_embeddings, all_labels], f, protocol=pickle.HIGHEST_PROTOCOL)

#
with open(fp_dest, "rb") as f:
    all_paths, all_embeddings, all_labels = pickle.load(f)
print('Load successful.')
print('Checking length of lists:')
print(len(all_paths), len(all_embeddings), len(all_labels))
print(len(all_embeddings[0]))
#for path, _, label in zip(all_paths, all_embeddings, all_labels):
print('Checking 5 random examples:')
seed = args.seed
for i in range(5):
    index = randint(0, len(all_paths)-1)
    print(all_paths[index], all_labels[index])
    

'''
counter = 0
for i, label in enumerate(labels):

    class_name = name_to_class[label]  # Use class name directly
    if class_name not in all_embeddings:
        all_embeddings[class_name] = []
    print(embeddings[i])
    all_embeddings[class_name].append(embeddings[i])
    counter += 1

for label, embs in all_embeddings.items():
    class_dir = os.path.join(fp_dest, str(label))
    os.makedirs(class_dir, exist_ok=True)  # Ensure directory exists
    print(label)
    np.save(os.path.join(class_dir, f"embeddings.npy"), np.vstack(embs))  # Save as np.vstack per class

print("Embeddings saved per class.")

'''