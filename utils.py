from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from torchvision.transforms import ToTensor
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class GluDataset(Dataset):
    """Glaucoma dataset."""

    def __init__(self, images, labels, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        label = self.labels[idx]
        # print(img_path)
        image = Image.open(img_path)
        image_tmp = ToTensor()(image)
        if image_tmp.shape[0] != 3:
            return None
        # sample = {'image': image, 'label': label}

        if self.transform:
            image = self.transform(image)
        
        return image, label