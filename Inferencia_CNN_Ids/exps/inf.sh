#!/bin/bash

mkdir -p ../logs/CBN
mkdir -p ../Ids/california
mkdir -p ../Ids/hydraulic
mkdir -p ../Ids/tides
mkdir -p ../Ids/utah
mkdir -p ../Ids/vibroseis
mkdir -p ../Ids/shaker

# DAS data inference

echo "Starting inference on California dataset"
python ../california_inf.py --classifier CBN --model_name CBN_10epch > ../logs/CBN/inf.txt

##echo "Starting inference on Hydraulic dataset"
#python ../hydraulic_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt
#
#echo "Starting inference on Tides dataset"
#python ../tides_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt
#
#echo "Starting inference on Utah dataset"
#python ../utah_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt
#
#echo "Starting inference on Vibroseis dataset"
#python ../vibroseis_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt
#
#echo "Starting inference on Shaker dataset"
#python ../shaker_inf.py --classifier CBN --model_name CBN_10epch >> ../logs/CBN/inf.txt

echo "Finished"