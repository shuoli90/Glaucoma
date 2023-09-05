from transformers import ViTFeatureExtractor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
import torch
import os
import utils
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import evaluate
import numpy as np


if __name__ == "__main__":
    args = utils.load_args()
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch32-384")
    normalize = Normalize(mean=feature_extractor.image_mean,
                          std=feature_extractor.image_std)
    size = [feature_extractor.size['height'], feature_extractor.size['width']]
    train_transforms, val_transforms, collate_fn = \
        utils.load_transforms(size, normalize)
    # # Detect if we have a GPU available
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    id2label = {0: 'Non-Glaucoma', 1:"Glaucoma"}
    label2id = {'Glaucoma':1, 'Non-Glaucoma':0}
    if args.train:
        model = ViTForImageClassification.from_pretrained("google/vit-large-patch32-384",
                                                          num_labels=2,
                                                          id2label=id2label,
                                                          label2id=label2id,
                                                          ignore_mismatched_sizes=True).to(device)
    else:
        model = ViTForImageClassification.from_pretrained('../models/test-POAAGG-filter-new',
                                                          num_labels=2,
                                                          id2label=id2label,
                                                          label2id=label2id).to(device)
    train_images, val_images, test_images = \
        utils.load_POAAGG()
    train_dataset = utils.GluDataset(train_images, train_transforms)
    val_dataset = utils.GluDataset(val_images, val_transforms)
    test_dataset = utils.GluDataset(test_images, val_transforms)

    trainer = utils.setup_trainer(model, train_dataset, val_dataset,
                                  feature_extractor, collate_fn,)
    if args.train:
        trainer.train()
        trainer.save_model()
        print('Training Finished!')

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
    for dataloader in [train_dataloader, val_dataloader, test_dataloader]:
        label_list, pred_list, running_corrects = \
            utils.evaluate_REFUGE(model, dataloader, label_list, pred_list, running_corrects)
    # epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / 3 /len(test_dataloader.dataset)
    label_list = np.hstack(label_list)
    pred_list = np.hstack(pred_list)
    fpr, tpr, thresholds = metrics.roc_curve(label_list, pred_list)
    auc = metrics.auc(fpr, tpr)
    utils.print_result('REFUGE', epoch_acc, auc, label_list, fpr, tpr, thresholds)