from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch
import os
import sys
sys.path.append("/home/lishuo1/glu")
from PIL import Image
import argparse
from transformers import SegformerFeatureExtractor
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode

from datasets import load_metric

metric = load_metric("mean_iou")

import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import random
import utils
import pandas as pd
import sklearn

class REFUGESegDataset_semantic(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms, feature_extractor, WIDTH=224, HEIGHT=224, annotation=None, nottrain=False):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        # self.locations = locations
        self.transforms = transforms
        self.feature_extractor = feature_extractor
        self.annotation = annotation
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)
    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]

        if self.annotation is not None:
            img_name = imagePath.split('/')[-1]
            col_name = 'Label(Glaucoma=1)' if 'Label(Glaucoma=1)' in self.annotation.columns else 'Glaucoma Label'
            label = self.annotation.loc[self.annotation.ImgName==img_name, col_name].values[0]
            assert (label in [0, 1])
        else:
            folder = imagePath.split('/')[-2]
            if folder not in ['Glaucoma', 'Non-Glaucoma']:
                folder = imagePath.split('/')[-4]
            assert(folder in ['Glaucoma', 'Non-Glaucoma'])
            if folder == 'Glaucoma':
                label = 1
            elif folder == 'Non-Glaucoma':
                label = 0
            else:
                print('Error!')

        image = Image.open(imagePath)
        Height, Width = image.size
        filename = list(imagePath.split('/')[-1])
        filename[-4:] = '.bmp'
        filename = "".join(filename)
        maskPath = [s for s in self.maskPaths if s.endswith(filename)][0]
        mask = Image.open(maskPath)
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)
        mask = ((1-mask) * 255).int()
        # print(mask.unique())
        mask[mask == 255] = 2
        mask[mask == 126] = 1
        
         # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(image, mask, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        encoded_inputs['classes'] = label
        # return a tuple of the image and its mask
        return encoded_inputs

def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':

    INPUT_IMAGE_HEIGHT = 224
    INPUT_IMAGE_WIDTH = 224
    THRESHOLD = 0.5

    parser = argparse.ArgumentParser(
                    prog = 'Segmentation finetuning',
                    description = 'finetuning segmentation using REFUGE data',
                    epilog = 'Text at the bottom of help')
    parser.add_argument('--root', type=str, default='/home/lishuo1/glu/data/REFUGE')           # positional argument
    parser.add_argument('--arch', type=str, default='unet')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    root_dir = args.root

    feature_extractor = SegformerFeatureExtractor(reduce_labels=False)

    NPUT_IMAGE_HEIGHT = 224
    INPUT_IMAGE_WIDTH = 224
    input_size = 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    transforms = transforms.Compose([transforms.Resize((INPUT_IMAGE_HEIGHT,
            INPUT_IMAGE_WIDTH), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor()])

    # create the train and test datasets

    from torch.utils.data import DataLoader


    from transformers import SegformerForSemanticSegmentation
    import json
    from huggingface_hub import cached_download, hf_hub_url


    id2label = {1: 'type I', 2: 'type II', 0: 'background'}
    label2id = {'type I': 1, 'type II': 2, 'background': 0}

    # define model
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                            num_labels=3, 
                                                            id2label=id2label, 
                                                            label2id=label2id,
    )

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
    # move model to GPU
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)
    accuracy = 0

    if args.train:

        cases_dir = []
        for item in os.walk(os.path.join(root_dir, 'Training400/Glaucoma')):
            pth = item[0]
            cases_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] == 'jpg'])
        controls_dir = []
        for item in os.walk(os.path.join(root_dir, 'Training400/Non-Glaucoma')):
            pth = item[0]
            # cases_dir.extend([os.path.join(pth, img) for img in item[2]])
            controls_dir.extend([os.path.join(pth, img) for img in item[2]])
        trainimages = cases_dir * 9 + controls_dir
        trainimages.sort()

        cases_dir = []
        for item in os.walk(os.path.join(root_dir, 'Annotation-Training400/Disc_Cup_Masks/Glaucoma')):
            pth = item[0]
            cases_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] == 'bmp'])
        controls_dir = []
        for item in os.walk(os.path.join(root_dir, 'Annotation-Training400/Disc_Cup_Masks//Non-Glaucoma')):
            pth = item[0]
            # cases_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] == 'bmp'])
            controls_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] == 'bmp'])
        trainmasks = cases_dir * 9 + controls_dir
        trainmasks.sort()

        cases_dir = []
        for item in os.walk(os.path.join(root_dir, 'REFUGE-Validation400')):
            pth = item[0]
            cases_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] == 'jpg'])
        valimages = cases_dir
        valimages.sort()

        cases_dir = []
        for item in os.walk(os.path.join(root_dir, 'REFUGE-Validation400-GT/Disc_Cup_Masks')):
            pth = item[0]
            cases_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] == 'bmp'])
        valmasks = cases_dir
        valmasks.sort()

        val_annotation = pd.read_excel(os.path.join(root_dir, 'REFUGE-Validation400-GT', 'Fovea_locations.xlsx'))
        image_tmp = []
        mask_tmp = []
        for valimage, valmask in zip(valimages, valmasks):
            image = valimage.split('/')[-1]
            label = val_annotation[val_annotation['ImgName'] == image]['Glaucoma Label'].values
            if np.any(label):
                image_tmp += [valimage] * 8
                mask_tmp += [valmask] * 8
        valimages += image_tmp
        valmasks += mask_tmp

        trainDS = REFUGESegDataset_semantic(imagePaths=trainimages, maskPaths=trainmasks,
            transforms=transforms, feature_extractor=feature_extractor)
        valDS = REFUGESegDataset_semantic(imagePaths=valimages, maskPaths=valmasks,
            transforms=transforms, feature_extractor=feature_extractor, annotation=val_annotation)

        print(f"[INFO] found {len(trainDS)} examples in the training set...")
        print(f"[INFO] found {len(valDS)} examples in the val set...")
        
        train_dataloader = DataLoader(trainDS, batch_size=4, num_workers=4, shuffle=True)
        valid_dataloader = DataLoader(valDS, batch_size=4, num_workers=4)

        model.train()
        for epoch in range(args.epoch):  # loop over the dataset multiple times
            print("Epoch:", epoch)
            for idx, batch in enumerate(tqdm(train_dataloader)):
                # get the inputs;
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss, logits = outputs.loss, outputs.logits
                
                loss.backward()
                optimizer.step()

                # evaluate
                if idx % 10 == 0:
                    predicted_list = []
                    label_list = []
                    with torch.no_grad():
                        for batch in valid_dataloader:
                            # get the inputs;
                            pixel_values = batch["pixel_values"].to(device)
                            labels = batch["labels"].to(device)

                            # forward + backward + optimize
                            outputs = model(pixel_values=pixel_values, labels=labels)
                            loss, logits = outputs.loss, outputs.logits

                            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                            predicted = upsampled_logits.argmax(dim=1)

                            predicted_list.append(predicted)
                            label_list.append(labels)
                    
                        predicted = torch.vstack(predicted_list)
                        labels = torch.vstack(label_list)
                        metrics = metric._compute(predictions=predicted.detach().cpu().numpy(), 
                                                    references=labels.detach().cpu().numpy(),
                                                    num_labels=len(id2label), 
                                                ignore_index=0,
                                                reduce_labels=False, # we've already reduced the labels before)
                        )

                        print("Loss:", loss.item())
                        print("Mean_iou:", metrics["mean_iou"])
                        print("Mean validation accuracy:", metrics["mean_accuracy"])

                        if metrics['mean_accuracy'] > accuracy:
                            accuracy = metrics['mean_accuracy']
                            torch.save(model.state_dict(), 'best_segformer_balance.pt')
        
    model.load_state_dict(torch.load('best_segformer_balance.pt'))
    model.eval()

    cases_dir = []
    for item in os.walk(os.path.join(root_dir, 'Test400')):
        pth = item[0]
        cases_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] == 'jpg'])
    testimages = cases_dir
    testimages.sort()

    cases_dir = []
    for item in os.walk(os.path.join(root_dir, 'REFUGE-Test-GT/Disc_Cup_Masks')):
        pth = item[0]
        cases_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] == 'bmp'])
    testmasks = cases_dir
    testmasks.sort()
    test_annotation = pd.read_excel(os.path.join(root_dir, 'REFUGE-Test-GT', 'Glaucoma_label_and_Fovea_location.xlsx'))

    test_annotation = pd.read_excel(os.path.join(root_dir, 'REFUGE-Test-GT', 'Glaucoma_label_and_Fovea_location.xlsx'))
    # val_annotation = pd.read_excel(os.path.join(root_dir, 'REFUGE-Validation400-GT', 'Fovea_locations.xlsx'))
    image_tmp = []
    mask_tmp = []
    for valimage, valmask in zip(valimages, valmasks):
        image = valimage.split('/')[-1]
        label = val_annotation[val_annotation['ImgName'] == image]['Label(Glaucoma=1)'].values
        if np.any(label):
            image_tmp += [valimage] * 8
            mask_tmp += [valmask] * 8
    valimages += image_tmp
    valmasks += mask_tmp

    testDS = REFUGESegDataset_semantic(imagePaths=testimages, maskPaths=testmasks,
        transforms=transforms, feature_extractor=feature_extractor, annotation=test_annotation)
    print(f"[INFO] found {len(testDS)} examples in the test set...")
    test_dataloader = DataLoader(testDS, batch_size=4, num_workers=4)

    print('Plot REFUGE')
    count = 50
    for idx, image_path in enumerate(tqdm(testimages)):
        if idx > count:
            break

        image = Image.open(image_path)
        image = image.resize((INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))

        encoding = feature_extractor(image, return_tensors="pt")
        pixel_values = encoding.pixel_values.to(device)

        with torch.no_grad():

            # forward + backward + optimize
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits.cpu()

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(logits,
                        size=image.size[::-1], # (height, width)
                        mode='bilinear',
                        align_corners=False)

        # Second, apply argmax on the class dimension
        seg = upsampled_logits.argmax(dim=1)[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # Convert to BGR
        color_seg = color_seg[..., ::-1]

        # Show image + mask
        img = np.array(image) * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)

        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.savefig(f'detection/test_segformer_{idx}.png')

        # map = Image.open(testmasks[idx]).resize((INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT), resample=0)
        maskPath = filename = list(image_path.split('/')[-1])
        filename[-4:] = '.bmp'
        filename = "".join(filename)
        maskPath = [s for s in testmasks if s.endswith(filename)][0]
        map = Image.open(maskPath).resize((INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT), resample=0)
        # map = Image.open(maskPath)
        map = np.array(map)
        map[map == 0] = 2
        map[map == 255] = 0
        map[map == 128] = 1
        classes_map = np.unique(map).tolist()
        unique_classes = [model.config.id2label[idx] if idx!=0 else None for idx in classes_map]

        # create coloured map
        color_seg = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8) # height, width, 3
        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[map == label, :] = color
        # Convert to BGR
        color_seg = color_seg[..., ::-1]

        # Show image + mask
        img = np.array(image) * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)

        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.savefig(f'gt/test_gt_{idx}.png')
        
        plt.close('all')