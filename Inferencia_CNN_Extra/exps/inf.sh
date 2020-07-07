#!/bin/bash

mkdir -p ../logs/CBN

# DAS data inference

echo "Starting inference on California dataset"
python ../california_inf.py --classifier CBN --model_name CBN_10epch > ../logs/CBN/inf.txt

echo "Starting inference on Hydraulic dataset"
python ../hydraulic_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt

echo "Starting inference on Tides dataset"
python ../tides_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt

echo "Starting inference on Utah dataset"
python ../utah_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt

echo "Starting inference on Vibroseis dataset"
python ../vibroseis_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt

echo "Starting inference on Shaker dataset"
python ../shaker_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt

echo "Starting inference on Signals"
python ../noise_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt

echo "Finished"