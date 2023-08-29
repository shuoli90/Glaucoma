from transformers import ViTFeatureExtractor
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    RandomRotation,
                                    ColorJitter,
                                    Resize, 
                                    ToTensor)
from torch.utils.data import DataLoader
import torch
import os
import random
import utils
from transformers import ViTForImageClassification
from transformers import TrainingArguments, Trainer
import argparse
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from utils import REFUGESegDataset
import pandas as pd
from sklearn import metrics
from scipy.special import softmax
import re

def get_attention_map(img, transform, model, threshold=0.5, get_mask=False):
    m = nn.Softmax(dim=1)
     
    model = model.to(device)
    x = transform(img).to(device)
    x.size()

    results = model(x.unsqueeze(0), output_attentions=True)
    logits, att_mat = results.logits, results.attentions
    prediction = m(logits)[0, 1] > threshold

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1)).to(device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
    if get_mask:
        result = cv2.resize(mask / mask.max(), img.size)
    else:        
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        result = (mask * img).astype("uint8")
    
    return result, prediction

def plot_attention_map(original_img, att_map, idx, label, prediction, directory='vit_attention/POAAGG'):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Last Layer')
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)
    plt.title(f'label:{label}, prediction:{prediction}')
    plt.savefig(os.path.join(directory, f'test_{idx}.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Resnet finetuning',
                    description='finetuning resnet-18/50 using Glaucoma data',
                    epilog='Text at the bottom of help')
    parser.add_argument('--root', type=str, default='data/POAAGG')           # positional argument
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--feature_extract', action='store_false')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--plots', type=int, default=40)
    args = parser.parse_args()

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch32-384")

    normalize = Normalize(mean=feature_extractor.image_mean,
                          std=feature_extractor.image_std)
    size = [feature_extractor.size['height'], feature_extractor.size['width']]

    _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                # RandomRotation(),
                ColorJitter(),
                ToTensor(),
                normalize,
            ]
        )

    _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )
    
    def train_transforms(image):
        # examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        image = _train_transforms(image.convert("RGB"))
        return image

    def val_transforms(image):
        # examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        image = _val_transforms(image.convert("RGB"))
        return image
    
    def collate_fn(examples):
        # pixel_values = torch.stack([example["pixel_values"] for example in examples])
        # labels = torch.tensor([example["label"] for example in examples])
        examples = list(filter(lambda x: x is not None, examples))
        pixel_values = torch.stack([example[0] for example in examples])
        labels = torch.tensor([example[1] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    
    def collate_fn2(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    # # Detect if we have a GPU available
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    id2label = {0: 'Non-Glaucoma', 1:"Glaucoma"}
    label2id = {'Glaucoma':1, 'Non-Glaucoma':0}
    
    if args.train:
        model = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384',
                                                        num_labels=2,
                                                        id2label=id2label,
                                                        label2id=label2id,
                                                        ignore_mismatched_sizes=True).to(device)
    else:
        model = ViTForImageClassification.from_pretrained('./models/test-POAAGG-new',
                                                        num_labels=2,
                                                        id2label=id2label,
                                                        label2id=label2id).to(device)

    metric_name = "roc_auc"

    train_args = TrainingArguments(
        f"./models/test-POAAGG-new",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        # evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_dir='logs',
        remove_unused_columns=False,
    )

    from datasets import load_metric
    import evaluate
    import numpy as np
    roc_auc_score = evaluate.load("roc_auc")
    metric = load_metric("f1")

    # def compute_metrics(p):
    #     preds = np.argmax(p.predictions, axis=1)

    #     return metric_auc.compute(
    #         predictions=preds, references=p.label_ids)["f1"]

    def compute_metrics(p):
        pred_scores = softmax(p.predictions.astype("float32"), axis=1)
        labels = np.array(p.label_ids).astype("int32")
        fpr, tpr, thresholds = metrics.roc_curve(labels, pred_scores[:, 1])
        auc = metrics.auc(fpr, tpr)
        return dict(roc_auc=auc)

    # Create training an  d validation datasets
    # root_dir = args.root
    root_dir = 'data/POAAGG'

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
    train_idx, test_idx, val_idx = torch.utils.data.random_split(patients, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
    train_patient = [patients[i] for i in train_idx.indices]
    val_patient = [patients[i] for i in val_idx.indices]
    test_patient = [patients[i] for i in test_idx.indices]   
    phoneNumRegex = re.compile(r'P(\d)_P(\d+)')
    reg = re.compile(r'O([S|D])')
    train_images = []
    for patient in train_patient:
        train_folder = folders[patient]
        for folder in train_folder:
            paths = os.listdir(folder)
            paths.sort()
            paths = [path for path in paths if not os.path.isdir(path)]
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
            paths = os.listdir(folder)
            paths.sort()
            paths = [path for path in paths if not os.path.isdir(path)]
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
            # paths = [path for path in os.listdir(folder) if not os.path.isdir(path)]
            # length = len(paths)
            # path1 = paths[int(length/2)]
            # test_images.append(os.path.join(folder, path1))
            # if length >= 2:
            #     path2 = os.listdir(folder)[int(length/2)-1]
            #     test_images.append(os.path.join(folder, path2))
            # test_images.extend([os.path.join(folder, path) for path in paths])
            paths = [path for path in os.listdir(folder) if not os.path.isdir(path)]
            indices = []
            left_dict = {}
            for path in paths:
                try:
                    groups = reg.findall(path)
                        
                    if len(groups) > 0:
                        if groups[0] == 'S':
                            left = 1
                        else:
                            left = 0

                    groups = phoneNumRegex.search(path).groups()
                    indices.extend(groups)

                    for group in groups:
                        left_dict[int(group)] = left
                except:
                    pass
            indices = list(set(indices))
            indices = [int(i) for i in indices]
            for path in paths:
                try:
                    if int(path.split('.')[0].split('_')[-1]) in indices:
                        test_images.append([os.path.join(folder, path), left_dict[int(path.split('.')[0].split('_')[-1])]])
                except:
                    pass


    train_dataset = utils.GluDataset(train_images, train_transforms)
    val_dataset = utils.GluDataset(val_images, val_transforms)
    test_dataset = utils.GluDataset(test_images, val_transforms)

    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )
    
    if args.train:
        trainer.train()
        trainer.save_model()
        print('Training Finished!')
    # metrics = trainer.evaluate(test_dataset)
    # print('testing metrics', metrics)
    
    # for idx, img_path in enumerate(tqdm(test_images)):
    #     if idx > args.plots:
    #         break
    #     folder = img_path.split('/')[-3]
    #     if folder not in ['cases', 'controls']:
    #         folder = img_path.split('/')[-4]
    #     assert(folder in ['cases', 'controls'])
    #     if folder == 'cases':
    #         label = 1
    #     elif folder == 'controls':
    #         label = 0
    #     else:
    #         print('Error!')
    #     try:
    #         image = Image.open(img_path)
    #     except:
    #         continue
    #     result, prediction = get_attention_map(image, get_mask=True, transform=val_transforms, model=model)
    #     plot_attention_map(image, result, idx, label, prediction, directory='./results/vit_attention/POAAGG')
    
    m = nn.Softmax(dim=1)
    test_dataset = utils.GluDataset(test_images, val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn2, pin_memory=True, num_workers=args.workers)
    label_list = []
    pred_list = []

    running_corrects = 0
    lefts = []
    patient_ids = []
    pred_label_list = []
    for idx, (inputs, labels, left, patient_id) in enumerate(tqdm(test_loader)):

        # for batch in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = outputs.logits

            tmp, preds = torch.max(outputs, 1)

        # statistics
        # running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        label_list.append(labels.data.cpu().numpy())
        pred_list.append(m(outputs)[:, 1].cpu().numpy())
        lefts.append(left.cpu().numpy())
        patient_ids.append(patient_id.cpu().numpy())
        pred_label_list.append(preds.cpu().numpy())
        
    # epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)

    label_list = np.hstack(label_list)
    pred_list = np.hstack(pred_list)
    lefts = np.hstack(lefts)
    patient_ids = np.hstack(patient_ids)
    pred_label_list = np.hstack(pred_label_list)

    fpr, tpr, thresholds = metrics.roc_curve(label_list, pred_list)
    auc = metrics.auc(fpr, tpr)

    print('POAAGG Test Acc: {:.4f}'.format(epoch_acc))    
    print(f'POAAGG Test Auc: {auc}')

    print('Positive count', np.sum(label_list))
    print('Negative count', np.sum(1-label_list))
    plt.figure()
    plt.title('Receiver Operating Characteristic - POAAGG')
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('AUROC POAAGG.png')

    plt.figure()
    plt.title('True Positive Curve - POAAGG')
    plt.plot(thresholds, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('Thresholds')
    plt.savefig('TPR POAAGG.png')

    from scipy.stats import mode
    df = pd.DataFrame({'patient_id': patient_ids, 'left': lefts, 'label': label_list, 'pred': pred_list, 'pred_label': pred_label_list})
    grouped = df.groupby('patient_id')
    majority_vote = grouped['pred_label'].agg(lambda x: mode(x)[0][0])
    label = grouped['label'].agg(lambda x: mode(x)[0][0])
    print('patient accuracy', np.sum(majority_vote == label) / len(label))
    
    # root_dir = args.root
    root_dir = 'data/REFUGE'
    label_list = []
    pred_list = []

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

    val_images = []
    for item in os.listdir(os.path.join(root_dir, 'REFUGE-Validation400')):
        if item[-3:] == 'jpg':
            path = os.path.join(root_dir, 'REFUGE-Validation400', item)
            val_images.append(path)
    val_annotation = pd.read_excel(os.path.join(root_dir, 'REFUGE-Validation400-GT', 'Fovea_locations.xlsx'))

    cases_dir = []
    for item in os.walk(os.path.join(root_dir, 'REFUGE-Validation400-GT/Disc_Cup_Masks')):
        pth = item[0]
        cases_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] == 'bmp'])
    valmasks = cases_dir

    test_images = []
    for item in os.listdir(os.path.join(root_dir, 'Test400')):
        if item[-3:] == 'jpg':
            path = os.path.join(root_dir, 'Test400', item)
            test_images.append(path)
    test_annotation = pd.read_excel(os.path.join(root_dir, 'REFUGE-Test-GT', 'Glaucoma_label_and_Fovea_location.xlsx'))

    cases_dir = []
    for item in os.walk(os.path.join(root_dir, 'REFUGE-Test-GT/Disc_Cup_Masks')):
        pth = item[0]
        cases_dir.extend([os.path.join(pth, img) for img in item[2] if img[-3:] == 'bmp'])
    testmasks = cases_dir

    train_dataset = utils.REFUGESegDataset(imagePaths=train_images, maskPaths=trainmasks, transforms=train_transforms)
    val_dataset = utils.REFUGESegDataset(imagePaths=val_images, maskPaths=valmasks, transforms=val_transforms, annotation=val_annotation)
    test_dataset = utils.REFUGESegDataset(imagePaths=test_images, maskPaths=testmasks, transforms=val_transforms, annotation=test_annotation)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,  pin_memory=True, num_workers=args.workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True,  pin_memory=True, num_workers=args.workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, pin_memory=True, num_workers=args.workers)

    running_loss = 0.0
    running_corrects = 0

    for idx, (inputs, masks, labels) in enumerate(tqdm(train_dataloader)):
        # for batch in dataloaders[phase]:
        inputs = inputs.to(device)
        masks = masks.to(device)
        classes = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            logits = outputs.logits
            tmp, preds = torch.max(logits, 1)

        # statistics
        running_corrects += torch.sum(preds == classes.data)

        label_list.append(classes.data.cpu().numpy())
        pred_list.append(m(logits)[:, 1].cpu().numpy())
    
    for idx, (inputs, masks, labels) in enumerate(tqdm(val_dataloader)):
        # for batch in dataloaders[phase]:
        inputs = inputs.to(device)
        masks = masks.to(device)
        classes = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            logits = outputs.logits
            tmp, preds = torch.max(logits, 1)

        # statistics
        running_corrects += torch.sum(preds == classes.data)

        label_list.append(classes.data.cpu().numpy())
        pred_list.append(m(logits)[:, 1].cpu().numpy())


    for idx, (inputs, masks, labels) in enumerate(tqdm(test_dataloader)):
        # for batch in dataloaders[phase]:
        inputs = inputs.to(device)
        masks = masks.to(device)
        classes = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            logits = outputs.logits
            tmp, preds = torch.max(logits, 1)

        # statistics
        running_corrects += torch.sum(preds == classes.data)

        label_list.append(classes.data.cpu().numpy())
        pred_list.append(m(logits)[:, 1].cpu().numpy())
        
    # epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / 3 /len(test_dataloader.dataset)

    label_list = np.hstack(label_list)
    pred_list = np.hstack(pred_list)

    fpr, tpr, thresholds = metrics.roc_curve(label_list, pred_list)
    auc = metrics.auc(fpr, tpr)

    print('REFUGE Acc: {:.4f}'.format(epoch_acc))    
    print(f'REFUGE Auc: {auc}')

    print('Positive count', np.sum(label_list))
    print('Negative count', np.sum(1-label_list))
    plt.figure()
    plt.title('Receiver Operating Characteristic - REFUGE')
    plt.plot(fpr, tpr)
    # plt.plot([0, 1], ls="--")
    # plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('AUROC REFUGE.png')

    plt.figure()
    plt.title('True Positive Curve - REFUGE')
    plt.plot(thresholds, tpr)
    # plt.plot([0, 1], ls="--")
    # plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('Thresholds')
    plt.savefig('TPR REFUGE.png')

    for idx, img_path in enumerate(tqdm(test_images)):
        if idx > args.plots:
            break
        img_name = img_path.split('/')[-1]
        col_name = 'Label(Glaucoma=1)' if 'Label(Glaucoma=1)' in test_annotation.columns else 'Glaucoma Label'
        label = test_annotation.loc[test_annotation.ImgName==img_name, col_name].values[0]
        assert (label in [0, 1])
        try:
            image = Image.open(img_path)
        except:
            continue
        result, prediction = get_attention_map(image, get_mask=True, transform=val_transforms, model=model)
        plot_attention_map(image, result, idx, label, prediction, directory='./results/vit_attention/REFUGE')