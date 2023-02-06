from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
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
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

import utils

INPUT_IMAGE_HEIGHT = 224
INPUT_IMAGE_WIDTH = 224
THRESHOLD = 0.9

def prepare_plot(origImage, origMask, predMask, count):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    plt.savefig(f'segResults/segment{count}.png')

def make_predictions(model, imagePath, groundTruePath, device, count):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        # resize the image and make a copy of it for visualization
        image = cv2.resize(image, (128, 128))
        orig = image.copy()
        gtMask = cv2.imread(groundTruePath, 0)
        gtMask = cv2.resize(gtMask, (INPUT_IMAGE_HEIGHT,
            INPUT_IMAGE_HEIGHT))
        
        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(device)
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        # filter out the weak predictions and convert them to integers
        predMask = (predMask > THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)
        # prepare a plot for visualization
        prepare_plot(orig, gtMask, predMask, count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Segmentation finetuning',
                    description = 'finetuning segmentation using REFUGE data',
                    epilog = 'Text at the bottom of help')
    parser.add_argument('--root', type=str, default='data/REFUGE')           # positional argument
    parser.add_argument('--arch', type=str, default='unet')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    
    root = args.root
    # Create training and validation datasets
    root_dir = root
    cases_dir = []
    for item in os.walk(os.path.join(root_dir, 'Training400/Glaucoma')):
        pth = item[0]
        cases_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] != 'tif'])
    controls_dir = []
    for item in os.walk(os.path.join(root_dir, 'Training400/Non-Glaucoma')):
        pth = item[0]
        cases_dir.extend([os.path.join(pth, img) for img in item[2]])
    images = cases_dir + controls_dir

    cases_dir = []
    for item in os.walk(os.path.join(root_dir, 'Annotation-Training400/Disc_Cup_Masks/Glaucoma')):
        pth = item[0]
        cases_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] != 'tif'])
    controls_dir = []
    for item in os.walk(os.path.join(root_dir, 'Annotation-Training400/Disc_Cup_Masks//Non-Glaucoma')):
        pth = item[0]
        cases_dir.extend([os.path.join(pth, img) for img in item[2]])
    masks = cases_dir + controls_dir

    locations = []
    train = pd.read_csv(os.path.join(root, 'Annotation-Training400', 'Fovea_location.csv'))[['Fovea_X', 'Fovea_Y']]
    train_tensor = torch.tensor(train.values)

    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    TEST_SPLIT = 0.2
    split = train_test_split(images, masks, test_size=TEST_SPLIT, random_state=42)
    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    
    INPUT_IMAGE_HEIGHT = 224
    INPUT_IMAGE_WIDTH = 224

    transforms = transforms.Compose([transforms.Resize((INPUT_IMAGE_HEIGHT,
            INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()])
    
    # create the train and test datasets
    trainDS = utils.REFUGESegDataset(imagePaths=trainImages, maskPaths=trainMasks,
        transforms=transforms)
    testDS = utils.REFUGESegDataset(imagePaths=testImages, maskPaths=testMasks,
        transforms=transforms)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=args.batch, pin_memory=True,
        num_workers=os.cpu_count())
    testLoader = DataLoader(testDS, shuffle=False,
        batch_size=args.batch, pin_memory=True,
        num_workers=os.cpu_count())
    
    # # Detect if we have a GPU available
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # initialize our UNet model
    unet = utils.UNet().to(device)
    
    if args.train:
        # initialize loss function and optimizer
        lossFunc = BCEWithLogitsLoss()
        INIT_LR = 0.001
        opt = Adam(unet.parameters(), lr=INIT_LR)
        # calculate steps per epoch for training and test set
        trainSteps = len(trainDS) // args.batch
        testSteps = len(testDS) // args.batch
        # initialize a dictionary to store training history
        H = {"train_loss": [], "test_loss": []}

        # loop over epochs
        print("[INFO] training the network...")
        startTime = time.time()
        for e in tqdm(range(args.epoch)):
            # set the model in training mode
            unet.train()
            # initialize the total training and validation loss
            totalTrainLoss = 0
            totalTestLoss = 0
            # loop over the training set
            for (i, (x, y)) in enumerate(trainLoader):
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # perform a forward pass and calculate the training loss
                pred = unet(x)
                loss = lossFunc(pred, y)
                # first, zero out any previously accumulated gradients, then
                # perform backpropagation, and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far
                totalTrainLoss += loss
                # switch off autograd
                with torch.no_grad():
                    # set the model in evaluation mode
                    unet.eval()
                    # loop over the validation set
                    for (x, y) in testLoader:
                        # send the input to the device
                        (x, y) = (x.to(device), y.to(device))
                        # make the predictions and calculate the validation loss
                        pred = unet(x)
                        totalTestLoss += lossFunc(pred, y)
                # calculate the average training and validation loss
                avgTrainLoss = totalTrainLoss / trainSteps
                avgTestLoss = totalTestLoss / testSteps
                # update our training history
                H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
                H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
                # print the model training and validation information
                print("[INFO] EPOCH: {}/{}".format(e + 1, args.epoch))
                # print("Train loss: {:.6f}, Test loss: {:.4f}".format(
                #     avgTrainLoss, avgTestLoss))
                print("Train loss: {:.6f}, Test loss: {:.4f}".format(
                    np.mean(H['train_loss']), np.mean(H['test_loss'])))
            # display the total time needed to perform the training
            endTime = time.time()
            print("[INFO] total time taken to train the model: {:.2f}s".format(
                endTime - startTime))
        
        # plot the training loss
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H["train_loss"], label="train_loss")
        plt.plot(H["test_loss"], label="test_loss")
        plt.title("Training Loss on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        plt.savefig('train.png')
        # serialize the model to disk
        torch.save(unet, 'unet')

    unet = torch.load('unet').to(device)
    unet.eval()


    # load the image paths in our testing file and randomly select 10
    # image paths
    print("[INFO] loading up test image paths...")
    # imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
    # imagePaths = np.random.choice(imagePaths, size=10)
    imagePaths = images[-50:]
    maskPaths = masks[-50:]
    
    # iterate over the randomly selected test image paths
    for count, (imagePath, maskPath) in enumerate(zip(imagePaths, maskPaths)):
        # make predictions and visualize the results
        make_predictions(unet, imagePath, maskPath, device=device, count=count)

