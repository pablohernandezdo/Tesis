#!/bin/bash

mkdir -p ../logs/XXL_train
mkdir -p ../logs/XXL_eval
mkdir -p ../models

trn="Train_data.hdf5"
tst="Test_data.hdf5"

# lr = 0.000001, bs = 32, Classifier_XXL
echo "Starting training, lr = 0.000001, epochs = 20, batch_size = 32"
python ../train.py --train_path $trn --test_path $tst --n_epochs 20 --lr 0.000001 --batch_size 32 --classifier XXL --model_name XXL_lr000001_bs32 > ../logs/XXL_train/lr_000001_bs32_XXL.txt

# Classifier_XXL
echo "Starting evaluation #1"
python ../eval.py --train_path $trn --test_path $tst --classifier XXL --model_name XXL_lr000001_bs32 > ../logs/XXL_eval/XXL_lr000001_bs32.txt

echo "Finished"