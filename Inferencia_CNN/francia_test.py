import time
import torch
import argparse
import scipy.io
import numpy as np

from scipy import signal
from scipy.signal import butter, lfilter

from model import *


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='Default_model', help="Classifier model path")
    parser.add_argument("--classifier", default='C', help="Choose classifier architecture, C, CBN")
    args = parser.parse_args()

    # Select training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load specified Classifier
    if args.classifier == 'CBN':
        net = ClassConvBN()
    elif args.classifier == 'C':
        net = ClassConv()
    else:
        net = ClassConv()
        print('Bad Classifier option, running classifier C')
    net.to(device)

    # Load parameters from trained model
    net.load_state_dict(torch.load('../STEAD_CNN/models/' + args.model_name + '.pth'))
    net.eval()

    # Load Francia data file 1
    f = scipy.io.loadmat("../Data_Francia/Earthquake_1p9_Var_BP_2p5_15Hz.mat")

    # Read data
    data = f["StrainFilt"]
    # time= f["Time"]
    # distance = f["Distance_fiber"]

    # Sampling frequency
    fs = 100
    total = 0
    tr_seismic, tr_noise = 0, 0
    fil_seismic, fil_noise = 0, 0

    # For every trace in the file
    for idx,trace in enumerate(data):
        # Filter
        fil_trace = butter_bandpass_filter(trace, 0.5, 1, fs, order=5)

        # Resample
        resamp_trace = signal.resample(trace, 6000)
        resamp_fil_trace = signal.resample(fil_trace, 6000)

        max_tr = np.max(np.abs(resamp_trace))
        max_fil_tr = np.max(np.abs(resamp_fil_trace))

        if max_tr == 0:
            total += 1

    print(total)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', output='ba')
    return b, a


def butter_bandpass_filter(dat, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, dat)
    return y


if __name__ == "__main__":
    main()
