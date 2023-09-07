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
import numpy as np
from sklearn.calibration import calibration_curve


if __name__ == "__main__":
    args = utils.load_args()
    save_path = '../models/test-POAAGG-filter-new'
    model, train_transforms, val_transforms, collate_fn, feature_extractor \
        = utils.load_model(args, save_path)
    train_images, val_images, test_images = \
        utils.load_POAAGG()
    train_dataset = utils.GluDataset(train_images, train_transforms)
    val_dataset = utils.GluDataset(val_images, val_transforms)
    test_dataset = utils.GluDataset(test_images, val_transforms)

    trainer = utils.setup_trainer(model, train_dataset, val_dataset,
                                  feature_extractor, collate_fn, save_path)
    if args.train:
        trainer.train()
        trainer.save_model()
        print('Training Finished!')

    val_loader = DataLoader(val_dataset, batch_size=args.batch,
                             shuffle=False, collate_fn=collate_fn,
                             pin_memory=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch,
                             shuffle=False, collate_fn=collate_fn,
                             pin_memory=True, num_workers=args.workers)

    if args.calibrate:
        from temperature_scaling.temperature_scaling import ModelWithTemperature
        import pickle
        if args.isotonic:
            model_temp = ModelWithTemperature(model, isotonic=True)
            filename = f'../logs/{args.model}_iso_first_informative'
            if os.path.exists(filename):
                model_temp.ir = pickle.load(open(filename, 'rb'))
            else:
                model_temp.isotonic_calibration(val_loader)
                pickle.dump(model_temp.ir, open(filename, 'wb'))
        else:
            model_temp = ModelWithTemperature(model) 
            filename = f'../logs/{args.model}_temperature_first_informative.pt'
            if os.path.exists(filename):
                temperature = torch.load(filename)
                model_temp.temperature = torch.nn.Parameter(torch.tensor(temperature, 
                                                                         device='cuda:0'))
            else:
                model_temp.set_temperature(val_loader)
                torch.save(model_temp.temperature.detach(),
                           filename)
    else:
        model_temp = ModelWithTemperature(model) 
        model_temp.temperature = torch.nn.Parameter(torch.tensor([1.0], device='cuda:0'))
    epoch_acc, label_list, pred_list, fpr, tpr, thresholds, auc, patient_ids, lefts, pred_label_list \
        = utils.evaluate_POAAGG(model_temp, test_loader)
    
    utils.print_result('POAAGG', epoch_acc, auc, label_list, fpr, tpr, thresholds)
    from scipy.stats import mode
    df = pd.DataFrame({'patient_id': patient_ids, 'left': lefts,
                    'label': label_list, 'pred': pred_list,
                    'pred_label': pred_label_list})
    grouped = df.groupby('patient_id')
    majority_vote = grouped['pred_label'].agg(lambda x: mode(x)[0][0])
    label = grouped['label'].agg(lambda x: mode(x)[0][0])
    print('patient accuracy', np.sum(majority_vote == label) / len(label))

    average = grouped['pred'].mean()
    print('visit auc', metrics.roc_auc_score(label, average))
    
    # prob_true, prob_pred = calibration_curve(label_list, pred_list, n_bins=10)
    # plt.figure(10)
    # plt.plot(prob_pred, prob_true, linewidth=1, color='blue')
    # plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='black')
    # plt.xlabel('Predicted probability')
    # plt.ylabel('True probability in each bin')
    # plt.legend(['Uncalibrated', 'Calibrated', 'Perfect'])
    # plt.grid(True)
    # plt.savefig('calibration_curve.png')

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