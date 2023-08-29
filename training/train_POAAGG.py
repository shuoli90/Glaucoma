from __future__ import print_function
from __future__ import division
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

import utils

def train_model(model, dataloaders, criterion, optimizer, feature_extract, num_epochs=25, is_inception=False, model_name='resnet50'):
    since = time.time()

    val_auc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0

    m = nn.Softmax(dim=1)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            print(phase)
            label_list = []
            pred_list = []
            for idx, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
            # for batch in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        label_list.append(labels.data.detach().cpu().numpy())
                        pred_list.append(m(outputs)[:, 1].detach().cpu().numpy())

                    _, preds = torch.max(outputs, 1)

                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                
            if phase == 'val':
                label_list = np.hstack(label_list)
                pred_list = np.hstack(pred_list)

                fpr, tpr, thresholds = metrics.roc_curve(label_list, pred_list)
                auc = metrics.auc(fpr, tpr)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val':
                print('{} AUC'.format(phase, auc))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            if phase == 'val' and auc > best_auc:
                best_auc = auc
                best_model_wts = copy.deepcopy(model.state_dict())
                if feature_extract:
                    torch.save(model.state_dict(), f'best_resnet_{model_name}.pt')
                else:
                    torch.save(model.state_dict(), f'best_resnet_{model_name}_full.pt')
            if phase == 'val':
                val_auc_history.append(auc)

        print()
    
    plt.figure()
    plt.plot(val_auc_history)
    plt.title('AUC')
    if feature_extract:
        plt.savefig(f'AUC_{model_name}_POAAGG.png')
    else:
        plt.savefig(f'AUC_{model_name}_fulltrain_POAAGG.png')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_auc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# def collate_fn(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     inputs = [x[0] for x in batch]
#     labels = [x[1] for x in batch]
#     # return torch.utils.data.dataloader.default_collate(batch)
    # return torch.tensor(inputs), torch.tensor(labels)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.savefig('test.png')  # pause a bit so that plots are updated
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog = 'Resnet finetuning',
                    description = 'finetuning resnet-18/50 using Glaucoma data',
                    epilog = 'Text at the bottom of help')
    parser.add_argument('--root', type=str, default='data/POAAGG')           # positional argument
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--feature_extract', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = args.model

    # Number of classes in the dataset
    num_classes = 2

    # Batch size for training (change depending on how much memory you have)
    batch_size = args.batch

    # Number of epochs to train for
    num_epochs = args.epoch

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = args.feature_extract

    # Initialize the model for this run
    model_ft, input_size = initialize_model(args.model, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    # print(model_ft)

    # Data augmentation and normalization for training
    # Just normalization for validation
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

    # print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    root_dir = args.root

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

    patients = list(set(patient_cases + patient_controls))
    patients.sort()
    train_idx,test_idx,val_idx = torch.utils.data.random_split(patients, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
    train_patient = [patients[i] for i in train_idx.indices]
    val_patient = [patients[i] for i in val_idx.indices]
    test_patient = [patients[i] for i in test_idx.indices]
    
    train_images = []
    for patient in train_patient:
        train_folder = folders[patient]
        for folder in train_folder:
            paths = [path for path in os.listdir(folder) if not os.path.isdir(path)]
            length = len(paths)
            path1 = paths[int(length/2)]
            train_images.append(os.path.join(folder, path1))
            if length >= 2:
                path2 = os.listdir(folder)[int(length/2)-1]
                train_images.append(os.path.join(folder, path2))
    
    val_images = []
    for patient in val_patient:
        val_folder = folders[patient]
        for folder in val_folder:
            paths = [path for path in os.listdir(folder) if not os.path.isdir(path)]
            length = len(paths)
            path1 = paths[int(length/2)]
            val_images.append(os.path.join(folder, path1))
            if length >= 2:
                path2 = os.listdir(folder)[int(length/2)-1]
                val_images.append(os.path.join(folder, path2))
    
    test_images = []
    for patient in test_patient:
        test_folder = folders[patient]
        for folder in test_folder:
            paths = [path for path in os.listdir(folder) if not os.path.isdir(path)]
            length = len(paths)
            path1 = paths[int(length/2)]
            test_images.append(os.path.join(folder, path1))
            if length >= 2:
                path2 = os.listdir(folder)[int(length/2)-1]
                test_images.append(os.path.join(folder, path2))

    train_dataset = utils.GluDataset(train_images, data_transforms['train'])
    val_dataset = utils.GluDataset(val_images, data_transforms['val'])
    test_dataset = utils.GluDataset(test_images, data_transforms['val'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=args.workers)

    # # Detect if we have a GPU available
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    if args.train:
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    
        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss(weight = torch.tensor([2.0, 1.0], device=device))
        dataloaders_dict = {}
        dataloaders_dict['train'] = train_loader
        dataloaders_dict['val'] = val_loader
        # Train and evaluate
        model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, 
                                    optimizer_ft, num_epochs=num_epochs, 
                                    is_inception=(model_name=="inception"), model_name=args.model, feature_extract=feature_extract)
    
    if feature_extract:
        model_ft.load_state_dict(torch.load(f'best_resnet_{args.model}.pt'))
    else:
        model_ft.load_state_dict(torch.load(f'best_resnet_{args.model}_full.pt'))
    model_ft.eval()
    phase = 'test'
    running_corrects = 0

    label_list = []
    pred_list = []

    m = nn.Softmax(dim=1)

    for idx, (inputs, labels) in enumerate(tqdm(test_loader)):

        # for batch in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model_ft(inputs)
            # loss = criterion(outputs, labels)

            tmp, preds = torch.max(outputs, 1)

        # statistics
        # running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        label_list.append(labels.data.cpu().numpy())
        pred_list.append(m(outputs)[:, 1].cpu().numpy())
        
    # epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)

    label_list = np.hstack(label_list)
    pred_list = np.hstack(pred_list)

    fpr, tpr, thresholds = metrics.roc_curve(label_list, pred_list)
    auc = metrics.auc(fpr, tpr)

    print('{} Acc: {:.4f}'.format(phase, epoch_acc))    
    print(f'{phase} Auc: {auc}')

    print('Positive count', np.sum(label_list))
    print('Negative count', np.sum(1-label_list))

    accuracy_best = 0.0
    best_threshold = 0.0
    thresholds = np.linspace(0.0, 1.0, 20)
    for threshold in thresholds:
        preds = pred_list > threshold
        accuracy = np.mean(preds)
        if accuracy > accuracy_best:
            accuracy_best = accuracy
            best_threshold = threshold
    print('best threshold for REFUGE', best_threshold)

    
