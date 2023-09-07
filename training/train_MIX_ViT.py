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
    save_path = '../models/test-REFUGE'
    model, train_transforms, val_transforms, collate_fn, feature_extractor \
        = utils.load_model(args, save_path)
    train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = \
        utils.load_mix(args, train_transforms=train_transforms, val_transforms=val_transforms)

    trainer = utils.setup_trainer(model, train_dataset, val_dataset,
                                  feature_extractor, collate_fn, save_path)
    if args.train:
        trainer.train()
        trainer.save_model()
        print('Training Finished!')

    train_images, val_images, test_images = \
        utils.load_POAAGG()
    test_dataset = utils.GluDataset(test_images, val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch,
                             shuffle=False, collate_fn=collate_fn,
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