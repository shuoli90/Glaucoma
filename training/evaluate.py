from torch.utils.data import DataLoader
import torch
import os
import utils
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import numpy as np


if __name__ == "__main__":
    args = utils.load_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    train_transforms, val_transforms, collate_fn, feature_extractor \
        = utils.load_model(args)
    if args.data == 'REFUGE':
        train_images, val_images, test_images, trainmasks, valmasks, testmasks, \
            val_annotation, test_annotation = \
            utils.load_REFUGE()
        train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = \
            utils.REFUGE_loader(args, train_images, val_images,
                                test_images, trainmasks,
                                valmasks, testmasks,
                                val_annotation, test_annotation,
                                train_transforms, val_transforms)
    elif args.data == 'POAAGG':
        train_images, val_images, test_images = \
            utils.load_POAAGG()
        train_dataset = utils.GluDataset(train_images, train_transforms)
        val_dataset = utils.GluDataset(val_images, val_transforms)
        test_dataset = utils.GluDataset(test_images, val_transforms)
    elif args.data == 'MIX':
        train_dataset, val_dataset, test_dataset = \
            utils.load_mix(args, train_transforms=train_transforms, val_transforms=val_transforms)
    elif args.data == 'COMB':
        train_dataset, val_dataset, test_dataset = \
            utils.load_combination(args, train_transforms=train_transforms, val_transforms=val_transforms)

    # check if model is vision transformer or cnn
    if args.model in ["resnet50", "resnet101"]:
        import torch.optim as optim
        import torch.nn as nn
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch,
            shuffle=True, collate_fn=utils.collate_fn_POAAGG,
            pin_memory=True, num_workers=args.workers)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch,
            shuffle=False, collate_fn=utils.collate_fn_POAAGG,
            pin_memory=True, num_workers=args.workers)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch,
            shuffle=False, collate_fn=utils.collate_fn_POAAGG,
            pin_memory=True, num_workers=args.workers)
        dataloaders = {'train': train_loader, 'val': val_loader}
        model, _ = utils.initialize_cnn_model(
            args.model, 2, 
            args.feature_extract)
        model = model.to(device)
        # Observe that all parameters are being optimized
        params_to_update = model.parameters()
        print("Params to learn:")
        if args.feature_extract:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad is True:
                    params_to_update.append(param)
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 9.0], device=device))
        if args.train:
            model, _ = utils.train_model(
                args, model, dataloaders,
                criterion, optimizer, num_epochs=args.epoch,
                model_name=args.model, is_inception=(args.model == "inception"),
                feature_extract=args.feature_extract,
                exp_name=f'{args.model}_{args.data}')
        else:
            model.load_state_dict(torch.load(f'./trained_models/{args.model}_{args.data}.pt'))
    else:
        from transformers import ViTForImageClassification
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        id2label = {0: 'Non-Glaucoma', 1:"Glaucoma"}
        label2id = {'Glaucoma':1, 'Non-Glaucoma':0}
        if args.train:
            model = ViTForImageClassification.from_pretrained(f"google/vit-{args.model}-patch32-384",
                                                            num_labels=2,
                                                            id2label=id2label,
                                                            label2id=label2id,
                                                            ignore_mismatched_sizes=True).to(device)
        else:
            model = ViTForImageClassification.from_pretrained(
                f'ViT_{args.data}',
                num_labels=2,
                id2label=id2label,
                label2id=label2id).to(device)
        if args.train:
            trainer = utils.setup_trainer(
                model, train_dataset, val_dataset,
                feature_extractor, collate_fn, f"./train_models/{args.model}_{args.data}")
            trainer.train()
            trainer.save_model()
    print('Training Finished!')

    # POAAGG
    train_images, val_images, test_images = \
        utils.load_POAAGG(seed=args.seed)
    test_dataset = utils.GluDataset(test_images, val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch,
                             shuffle=False, collate_fn=utils.collate_fn_POAAGG,
                             pin_memory=True, num_workers=args.workers)
    epoch_acc, label_list, pred_list, fpr, tpr, thresholds, auc, patient_ids, lefts, pred_label_list \
        = utils.evaluate_POAAGG(model, test_loader)
    utils.print_result('POAAGG', epoch_acc, auc, label_list, fpr, tpr, thresholds)
    from scipy.stats import mode
    df = pd.DataFrame({'patient_id': patient_ids, 'left': lefts,
                    'label': label_list, 'pred': pred_list,
                    'pred_label': pred_label_list})
    grouped = df.groupby('patient_id')
    majority_vote = grouped['pred_label'].agg(lambda x: mode(x)[0][0])
    label = grouped['label'].agg(lambda x: mode(x)[0][0])
    print('patient accuracy', np.sum(majority_vote == label) / len(label))

    # REFUGE
    train_images, val_images, test_images, trainmasks, valmasks, testmasks, \
        val_annotation, test_annotation = \
        utils.load_REFUGE()
    train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = \
        utils.REFUGE_loader(args, train_images, val_images,
                            test_images, trainmasks,
                            valmasks, testmasks,
                            val_annotation, test_annotation,
                            train_transforms, val_transforms)

    label_list = []
    pred_list = []
    running_corrects = 0
    for dataloader in [test_dataloader]:
        label_list, pred_list, running_corrects = \
            utils.evaluate_REFUGE(model, dataloader, label_list, pred_list, running_corrects)
    # epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() /len(test_dataloader.dataset)
    label_list = np.hstack(label_list)
    pred_list = np.hstack(pred_list)
    fpr, tpr, thresholds = metrics.roc_curve(label_list, pred_list)
    auc = metrics.auc(fpr, tpr)
    utils.print_result('REFUGE', epoch_acc, auc, label_list, fpr, tpr, thresholds)

    # MIX
    train_dataset, val_dataset, test_dataset = \
        utils.load_mix(args, train_transforms=train_transforms, val_transforms=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch,
                             shuffle=False, collate_fn=utils.collate_fn_POAAGG,
                             pin_memory=True, num_workers=args.workers)
    epoch_acc, label_list, pred_list, fpr, tpr, thresholds, auc, patient_ids, lefts, pred_label_list \
        = utils.evaluate_POAAGG(model, test_loader)
    utils.print_result('MIX', epoch_acc, auc, label_list, fpr, tpr, thresholds)

    # COMB
    train_dataset, val_dataset, test_dataset = \
        utils.load_combination(args, train_transforms=train_transforms, val_transforms=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch,
                             shuffle=False, collate_fn=utils.collate_fn_POAAGG,
                             pin_memory=True, num_workers=args.workers)
    epoch_acc, label_list, pred_list, fpr, tpr, thresholds, auc, patient_ids, lefts, pred_label_list \
        = utils.evaluate_POAAGG(model, test_loader)
    utils.print_result('COMB', epoch_acc, auc, label_list, fpr, tpr, thresholds)
    
