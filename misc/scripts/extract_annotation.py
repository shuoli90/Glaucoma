from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageChops, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import ToTensor
import random
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
import re
import matplotlib.pyplot as plt

# image1 = torch.mean(ToTensor()(Image.open(img_path_1))* 255, dim=2)
# image2 = torch.mean(ToTensor()(Image.open(img_path_2))* 255, dim=2)
# annotated1 = torch.mean(ToTensor()(Image.open(annotated1))* 255, dim=2)
# annotated2 = torch.mean(ToTensor()(Image.open(annotated2))* 255, dim=2)
# sub_1 = torch.sub(image1.int(), annotated1.int())
# sub_2 = torch.sub(image2.int(), annotated2.int())
# sub_ann = torch.sub(annotated1.int(), annotated2.int())
# print(torch.nonzero(sub_1).shape)
# print(torch.nonzero(sub_2).shape)
# print(torch.nonzero(sub_ann).shape)
# breakpoint()

def main():
    img_path_1 = './data/POAAGG/cases/00001_C_20070702/POAAGG_17492_3.PNG'
    img_path_2 = './data/POAAGG/cases/00001_C_20070702/POAAGG_17492_7.PNG'
    annotated1 = './data/POAAGG/cases/00001_C_20070702/00001_C_20070702_OD_CPO_20200730_P3_P4.jpg'
    annotated2 = './data/POAAGG/cases/00001_C_20070702/00001_C_20070702_OD_ES_20200806_P3_P4.jpg'

    # Load the two images
    png_image = Image.open(img_path_1).convert('RGB')
    jpg_image = Image.open(annotated1).convert('RGB')

    # Check if images are of the same size
    if png_image.size != jpg_image.size:
        print("Error: Images are not the same size.")
        return

    # Calculate the difference between the two images
    diff_image = ImageChops.subtract(png_image, jpg_image)
    diff_array = np.array(diff_image)
    
    plt.hist(diff_array.ravel(), bins=256, range=(0, 256))
    plt.xlabel('Pixel values')
    plt.ylabel('Frequency')
    plt.title('Histogram of difference image values')
    plt.savefig('histogram.png')

    # Save the difference image as a new PNG file
    diff_image.save('difference.jpg')

if __name__ == '__main__':
    main()
