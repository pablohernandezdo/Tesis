#!/bin/bash

mkdir -p ../logs/XXLtrain
mkdir -p ../models

trnpath="Train_data.hdf5"
tstpath="Test_data.hdf5"

# lr = 0.00001, bs = 32, Classifier_XXL
echo "Starting experiment #1, lr = 0.00001, epochs = 20, batch_size = 32"
python ../train.py --train_path $trnpath --test_path $tstpath --n_epochs 20 --lr 0.000001 --batch_size 32 --classifier XXL --model_name XXL_lr000001_bs32 > ../logs/XXLtrain/lr_000001_bs32_XXL.txt
