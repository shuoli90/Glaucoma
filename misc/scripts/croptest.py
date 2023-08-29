from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from PIL import Image
import argparse
from tqdm import tqdm
from sklearn import metrics
import random
import pandas as pd
from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation, SegformerForImageClassification
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode
from Transformers.SegFormer.segformer_multitask_serial import REFUGESegDataset_semantic
from torchvision.transforms import ToTensor

import utils

root_dir = 'data/REFUGE'
train_images = []
for item in os.listdir(os.path.join(root_dir, 'Training400', 'Glaucoma')):
    if item[-3:] == 'jpg':
        path = os.path.join(root_dir, 'Training400', 'Glaucoma', item)
        train_images.append(path)

for item in os.listdir(os.path.join(root_dir, 'Training400', 'Non-Glaucoma')):
    if item[-3:] == 'jpg':
        path = os.path.join(root_dir, 'Training400', 'Non-Glaucoma', item)
        train_images.append(path)

cases_dir = []
for item in os.walk(os.path.join(root_dir, 'Annotation-Training400/Disc_Cup_Masks/Glaucoma')):
    pth = item[0]
    cases_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] == 'bmp'])
controls_dir = []
for item in os.walk(os.path.join(root_dir, 'Annotation-Training400/Disc_Cup_Masks//Non-Glaucoma')):
    pth = item[0]
    cases_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] == 'bmp'])
trainmasks = cases_dir + controls_dir

def crop_by_mask(imagepath, maskpath)
    img = Image.open(train_images[0]).convert("RGBA")
    img.save('imgexample.png')
    background = Image.new("RGBA", img.size, (0,0,0,0))
    imgPath = train_images[0].split('/')
    mask_path = os.path.join(root_dir, 
                            'Annotation-Training400', 
                            'Disc_Cup_Masks', 
                            imgPath[-2], 
                            imgPath[-1][:-4]+'.bmp')
    mask = Image.open(mask_path).convert('RGBA')
    mask.save('maskexample.png')
    mask = np.array(Image.open(mask_path)) != 255
    nonzeros_y, nonzeros_x = np.nonzero(mask)
    mask_x_min = np.min(nonzeros_x) - 2
    mask_x_max = np.max(nonzeros_x) - 2
    mask_y_min = np.min(nonzeros_y) + 2
    mask_y_max = np.max(nonzeros_y) + 2

    # new_img = Image.composite(img, background, mask)
    img_crop = img.crop((mask_x_min,mask_y_min, mask_x_max, mask_y_max))
    img_crop.save('cropexample.png')

    return img_crop 
