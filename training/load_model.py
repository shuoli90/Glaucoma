from transformers import ViTFeatureExtractor
import torch
from transformers import ViTForImageClassification
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ViT finetuning',
                    description='finetuning resnet-18/50 using Glaucoma data',
                    epilog='Text at the bottom of help')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='../models/test-POAAGG-new')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch32-384")

    # # Detect if we have a GPU available
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    id2label = {0: 'Non-Glaucoma', 1:"Glaucoma"}
    label2id = {'Glaucoma':1, 'Non-Glaucoma':0}
    

    model = ViTForImageClassification.from_pretrained(args.model,
                                                      num_labels=2,
                                                      id2label=id2label,
                                                      label2id=label2id).to(device)