#!/bin/bash

mkdir -p ../logs/inf

# DAS data inference

echo "Starting inference on France dataset"
python ../francia_inf.py --classifier XXL --model_name XXL_lr000001_bs32 > ../logs/inf/inf.txt

echo "Starting inference on Nevada dataset"
python ../nevada_inf.py --classifier XXL --model_name XXL_lr000001_bs32 >> ../logs/inf/inf.txt

echo "Starting inference on Belgica dataset"
python ../belgica_inf.py --classifier XXL --model_name XXL_lr000001_bs32 >> ../logs/inf/inf.txt

echo "Starting inference on Reykjanes dataset"
python ../reykjanes_inf.py --classifier XXL --model_name XXL_lr000001_bs32 >> ../logs/inf/inf.txt

echo "Finished"