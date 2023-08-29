#! /bin/bash

CUDA_VISIBLE_DEVICES=1,2 python train_POAAGG_ViT_calibrate.py --model_name test-POAAGG-filter-new > log_test-POAAGG-filter-new.txt
CUDA_VISIBLE_DEVICES=1,2 python train_POAAGG_ViT_calibrate.py --model_name test-POAAGG-full > log_test-POAAGG-full.txt
CUDA_VISIBLE_DEVICES=1,2 python train_POAAGG_ViT_calibrate.py --model_name test-POAAGG-new > log_test-POAAGG.txt