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
        
    model.load_state_dict(torch.load('./models/segformer/best_segformer.pt'))
    model.eval()

    root_dir = '/home/lishuo1/glu/data/POAAGG'
    patient_cases = []
    folders = {}
    for item in os.listdir(os.path.join(root_dir, 'cases')):
        patient = item.split('_')[0]
        d = os.path.join(root_dir, 'cases', item)
        if os.path.isdir(d):
            patient_cases.append(patient)
            if patient in folders:
                folders[patient].append(d)
            else:
                folders[patient] = [d]

    patient_controls = []
    for item in os.listdir(os.path.join(root_dir, 'controls')):
        patient = item.split('_')[0]
        d = os.path.join(root_dir, 'controls', item)
        if os.path.isdir(d):
            patient_controls.append(patient)
            if patient in folders:
                folders[patient].append(d)
            else:
                folders[patient] = [d]

    patients = patient_cases + patient_controls
    random.shuffle(patients)
    train_idx,test_idx,val_idx = torch.utils.data.random_split(patients, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
    train_patient = [patients[i] for i in train_idx.indices]
    val_patient = [patients[i] for i in val_idx.indices]
    test_patient = [patients[i] for i in test_idx.indices]

    def largest_indices(lst, K=6):
        """
        Returns the indices of the six largest distinct numbers in a list.
        """
        largest_indices = []
        largest_numbers = []
        for i in range(len(lst)):
            if len(largest_numbers) == K:
                break
            if lst[i] not in largest_numbers:
                largest_indices.append(i)
                largest_numbers.append(lst[i])
        return largest_indices

    test_images = []
    for patient in test_patient:
        test_folder = folders[patient]
        for folder in test_folder:
            areas = []
            paths = [path for path in os.listdir(folder) if not os.path.isdir(path)]
            for idx, image_path in enumerate(tqdm(paths)):
                image = Image.open(os.path.join(folder, image_path))
                image = image.resize((INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
                image = transforms(image)
                image = image.unsqueeze(0)
                image = image.to(device)
                with torch.no_grad():
                    output = model(image)
                pred = output[0].argmax(1).cpu().numpy()
                area = np.sum(pred == 1)
                areas.append(area)
            indices = largest_indices(areas, K=6)
            breakpoint()