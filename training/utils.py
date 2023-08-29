from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from PIL import ImageFile
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

def crop_by_mask(img, mask_path):
    mask_array = np.array(Image.open(mask_path)) != 255
    nonzeros_y, nonzeros_x = np.nonzero(mask_array)
    mask_x_min = np.min(nonzeros_x) - 2
    mask_x_max = np.max(nonzeros_x) - 2
    mask_y_min = np.min(nonzeros_y) + 2
    mask_y_max = np.max(nonzeros_y) + 2
    x_center = (mask_x_min + mask_x_max) / 2.0
    y_center = (mask_y_min + mask_y_max) / 2.0
    mask_x_min = x_center - 150
    mask_x_max = x_center + 150
    mask_y_min = y_center - 150
    mask_y_max = y_center + 150
    img_crop = img.crop((mask_x_min,mask_y_min, mask_x_max, mask_y_max))
    return img_crop 

def crop_by_center(img, center):
    x_center = center[0]
    y_center = center[1]
    mask_x_min = x_center - 150
    mask_x_max = x_center + 150
    mask_y_min = y_center - 150
    mask_y_max = y_center + 150
    img_crop = img.crop((mask_x_min,mask_y_min, mask_x_max, mask_y_max))
    # print(x_center, y_center)
    return img_crop 

def mask_background(img, mask_path, macular_center):
    img_tmp = np.asarray(img)
    mask = np.zeros_like(img_tmp)
    mask_array = np.array(Image.open(mask_path)) != 255
    nonzeros_y, nonzeros_x = np.nonzero(mask_array)
    mask_x_min = np.min(nonzeros_x) - 2
    mask_x_max = np.max(nonzeros_x) - 2
    mask_y_min = np.min(nonzeros_y) + 2
    mask_y_max = np.max(nonzeros_y) + 2
    x_center = (mask_x_min + mask_x_max) / 2.0
    y_center = (mask_y_min + mask_y_max) / 2.0
    mask_x_min = int(x_center - 150)
    mask_x_max = int(x_center + 150)
    mask_y_min = int(y_center - 150)
    mask_y_max = int(y_center + 150)
    mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max, :] = 1

    x_center = macular_center[0]
    y_center = macular_center[1]
    macula_x_min = int(x_center - 150)
    macula_x_max = int(x_center + 150)
    macula_y_min = int(y_center - 150)
    macula_y_max = int(y_center + 150)
    mask[macula_y_min:macula_y_max, macula_x_min:macula_x_max, :] = 1

    return img_tmp * mask

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
INPUT_IMAGE_HEIGHT = 300
INPUT_IMAGE_WIDTH = 300

class GluDataset(Dataset):
    """Glaucoma dataset."""

    def __init__(self, images,  transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images
        self.transform = transform
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, left = self.images[idx]
        patient_id = int((img_path.split('/')[3]).split('_')[0])
        visit_id = img_path.split('/')[-2]
        # print(img_path)
        folder = img_path.split('/')[-3]
        if folder not in ['cases', 'controls']:
            folder = img_path.split('/')[-4]
        assert(folder in ['cases', 'controls'])
        if folder == 'cases':
            label = 1
        elif folder == 'controls':
            label = 0
        else:
            print('Error!')
        # try:
        image = Image.open(img_path)
        image_tmp = ToTensor()(image)
        if image_tmp.shape[0] != 3:
            return None

        if self.transform:
            image = self.transform(image)
        
        return image, label, left, patient_id, visit_id
        # except:
        #     return None

class GluDataset_original(Dataset):
    """Glaucoma dataset."""

    def __init__(self, images,  transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images
        self.transform = transform
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        patient_id = int((img_path.split('/')[3]).split('_')[0])
        visit_id = img_path.split('/')[-2]
        # print(img_path)
        folder = img_path.split('/')[-3]
        if folder not in ['cases', 'controls']:
            folder = img_path.split('/')[-4]
        assert(folder in ['cases', 'controls'])
        if folder == 'cases':
            label = 1
        elif folder == 'controls':
            label = 0
        else:
            print('Error!')
        # try:
        image = Image.open(img_path)
        image_tmp = ToTensor()(image)
        if image_tmp.shape[0] != 3:
            return None

        if self.transform:
            image = self.transform(image)
        
        return image, label, patient_id, visit_id

class GluDataset_binary(Dataset):
    """Glaucoma dataset."""

    def __init__(self, images,  transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images
        self.transform = transform
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, selected = self.images[idx]
        patient_id = int((img_path.split('/')[3]).split('_')[0])
        visit_id = img_path.split('/')[-2]
        folder = img_path.split('/')[-3]
        if folder not in ['cases', 'controls']:
            folder = img_path.split('/')[-4]
        assert(folder in ['cases', 'controls'])
        if folder == 'cases':
            label = 1
        elif folder == 'controls':
            label = 0
        else:
            print('Error!')
        # try:
        image = Image.open(img_path)
        image_tmp = ToTensor()(image)
        if image_tmp.shape[0] != 3:
            return None

        if self.transform:
            image = self.transform(image)
        
        return image, label, selected, patient_id, visit_id
        # except:
        #     return None

class REFUGEClsDataset(Dataset):
    """Glaucoma dataset."""

    def __init__(self, images, annotation=None, nottrain=False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images
        self.transform = transform
        self.annotation = None

        if nottrain:
            self.annotation = annotation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        if self.annotation is not None:
            img_name = img_path.split('/')[-1]
            col_name = 'Label(Glaucoma=1)' if 'Label(Glaucoma=1)' in self.annotation.columns else 'Glaucoma Label'
            label = self.annotation.loc[self.annotation.ImgName==img_name, col_name].values[0]
            assert (label in [0, 1])
        else:
            folder = img_path.split('/')[-2]
            if folder not in ['Glaucoma', 'Non-Glaucoma']:
                folder = img_path.split('/')[-4]
            assert(folder in ['Glaucoma', 'Non-Glaucoma'])
            if folder == 'Glaucoma':
                label = 1
            elif folder == 'Non-Glaucoma':
                label = 0
            else:
                print('Error!')

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        
        return image, label, left
        # except:
        #     return None

class REFUGESegDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms, annotation=None, WIDTH=224, HEIGHT=224, mode='zoomin'):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.annotation = annotation
        self.mode = mode
        self.transforms = transforms
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)
    def __getitem__(self, idx):
        # grab the image path from the current index
        img_path = self.imagePaths[idx]
        img_name = img_path.split('/')[-1]
        if (self.annotation is not None) and (('Label(Glaucoma=1)' in self.annotation.columns) or 'Glaucoma Label' in self.annotation.columns):
            col_name = 'Label(Glaucoma=1)' if 'Label(Glaucoma=1)' in self.annotation.columns else 'Glaucoma Label'
            label = self.annotation.loc[self.annotation.ImgName==img_name, col_name].values[0]
            assert (label in [0, 1])
        else:
            folder = img_path.split('/')[-2]
            if folder not in ['Glaucoma', 'Non-Glaucoma']:
                folder = img_path.split('/')[-4]
            assert(folder in ['Glaucoma', 'Non-Glaucoma'])
            if folder == 'Glaucoma':
                label = 1
            elif folder == 'Non-Glaucoma':
                label = 0
            else:
                print('Error!')

        image = Image.open(img_path)
        Height, Width = image.size
        filename = list(img_path.split('/')[-1])
        filename[-4:] = '.bmp'
        filename = "".join(filename)
        maskPath = [s for s in self.maskPaths if s.endswith(filename)][0]
        mask = Image.open(maskPath).convert('RGB')
        if self.mode in ['macula', 'background']:
            X, Y = self.annotation.loc[self.annotation.ImgName == img_name, 'Fovea_X'].values[0], self.annotation.loc[self.annotation.ImgName==img_name, 'Fovea_Y'].values[0]

        if self.mode == 'macula':
            image1 = crop_by_mask(image, maskPath)
            image2 = crop_by_center(image, (X, Y))
        elif self.mode == 'background':
            image1 = mask_background(image, maskPath, (X, Y))
            image1 = Image.fromarray(image1)
        else:
            image1 = crop_by_mask(image, maskPath)
        # image1.save('tmp.jpg')
        # breakpoint()       
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image1 = self.transforms(image)
            if self.mode == 'macula':
                image2 = self.transforms(image2)
            mask = self.transforms(mask)
        seg_true = torch.zeros_like(mask)
        seg_true[0, :, :] = mask[0, :, :] == 0
        seg_true[1, :, :] = mask[1, :, :] == 0.5020
        seg_true[2, :, :] = mask[2, :, :] == 1.0

        # return a tuple of the image and its mask
        # return (image, mask, label)
        if self.mode == 'macula':
            image1 = torch.vstack([image1, image2])
        return (image1, seg_true, label)

class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)
    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                 for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)
    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        # return the list containing the intermediate outputs
        return blockOutputs

class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
                 for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                 for i in range(len(channels) - 1)])
    def forward(self, x, encFeatures):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x
    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures

class UNet(Module):
    def __init__(self, encChannels=(3, 16, 32, 64),
         decChannels=(64, 32, 16),
         nbClasses=1, retainDim=True,
         outSize=(INPUT_IMAGE_HEIGHT,  INPUT_IMAGE_WIDTH)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize
    
    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0],
            encFeatures[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        map = self.head(decFeatures)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
        # return the segmentation map
        return map