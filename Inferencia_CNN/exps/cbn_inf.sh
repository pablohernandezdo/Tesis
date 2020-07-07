#!/bin/bash

mkdir -p ../logs/CBN

# DAS data inference

echo "Starting inference on France dataset"
python ../francia_inf.py --classifier CBN --model_name CBN_10epch > ../logs/CBN/inf.txt

echo "Starting inference on Nevada dataset"
python ../nevada_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt

echo "Starting inference on Belgica dataset"
python ../belgica_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt

echo "Starting inference on Reykjanes dataset"
python ../reykjanes_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt

echo "Finished"