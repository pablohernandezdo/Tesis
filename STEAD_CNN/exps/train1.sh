#!/bin/bash

# ClassConv_BN Classifier training

mkdir -p ../logs/CBN_train
mkdir -p ../models

#trn='MiniTrain.hdf5'
#tst='MiniTest.hdf5'

trn="Train_data.hdf5"
tst="Test_data.hdf5"

#trn="BigTrain.hdf5"
#tst="BigTest.hdf5"

# Classifier_XXL
echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn --test_path $tst --classifier CBN --model_name CBN_1epch > ../logs/CBN_train/CBN_1epch.txt
