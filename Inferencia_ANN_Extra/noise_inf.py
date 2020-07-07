import h5py
import torch
import argparse
import numpy as np

from numpy import random
from scipy.signal import butter, lfilter

from model import *


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='XXL_lr000001_bs32', help="Classifier model path")
    parser.add_argument("--classifier", default='XXL', help="Choose classifier architecture, C, S, XS, XL, XXL, XXXL")
    args = parser.parse_args()

    # Select training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load specified Classifier
    if args.classifier == 'XS':
        net = Classifier_XS()
    elif args.classifier == 'S':
        net = Classifier_S()
    elif args.classifier == 'XL':
        net = Classifier_XL()
    elif args.classifier == 'XXL':
        net = Classifier_XXL()
    elif args.classifier == 'XXXL':
        net = Classifier_XXXL()
    else:
        net = Classifier()
    net.to(device)

    # Load parameters from trained model
    net.load_state_dict(torch.load('../../STEAD_ANN/models/' + args.model_name + '.pth'))
    net.eval()
    # Noise
    ns = random.random_sample((1, 6000))

    # Sine

    # Number of sample points
    N = 6000

    # sampling frequency
    fs = 100

    # sampling spacing
    T = 1.0 / fs

    t = np.linspace(0.0, N / fs, N)

    n = 100

    # Frequency spans
    fr1 = np.linspace(1, 100, n)
    fr2 = np.linspace(0.1, 10, n)

    # Prealocate
    wvs1 = []
    wvs2 = []
    wvs3 = []

    for i in np.arange(n):
        sig1 = np.sin(fr1[i] * 2.0 * np.pi * t)
        sig2 = np.sin(fr2[i] * 2.0 * np.pi * t)
        wvs1.append(sig1)
        wvs2.append(sig2)
        wvs3.append(sig1 + sig2)

    wvs1 = np.array(wvs1)
    wvs2 = np.array(wvs2)
    wvs3 = np.array(wvs3)

    wvs1_ns = wvs1 + random.random_sample(wvs1.shape)
    wvs2_ns = wvs2 + random.random_sample(wvs1.shape)
    wvs3_ns = wvs3 + random.random_sample(wvs1.shape)

    # Noise inference

    # Normalize
    ns_norm = ns / np.max(np.abs(ns))

    # Numpy to Torch
    ns_norm = torch.from_numpy(ns_norm).to(device)

    # Prediction
    out_ns = net(ns_norm.float())

    out_ns = torch.round(out_ns.data).item()

    print(f'Noise inference:\n'
          f'Prediction: {out_ns}\n')

    # Sine wave inf

    total = 0
    tr_seismic, tr_noise = 0, 0

    # For every trace in the file
    for trace in wvs1:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total += 1

        if pred_trace:
            tr_seismic += 1
        else:
            tr_noise += 1

    # Results
    print(f'Inferencia Waves1:\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n')

    total = 0
    tr_seismic, tr_noise = 0, 0

    # For every trace in the file
    for trace in wvs2:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total += 1

        if pred_trace:
            tr_seismic += 1
        else:
            tr_noise += 1

    # Results
    print(f'Inferencia Waves2:\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n')

    total = 0
    tr_seismic, tr_noise = 0, 0

    # For every trace in the file
    for trace in wvs3:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total += 1

        if pred_trace:
            tr_seismic += 1
        else:
            tr_noise += 1

    # Results
    print(f'Inferencia Waves3:\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n')

    total = 0
    tr_seismic, tr_noise = 0, 0

    # For every trace in the file
    for trace in wvs1_ns:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total += 1

        if pred_trace:
            tr_seismic += 1
        else:
            tr_noise += 1

    # Results
    print(f'Inferencia Waves1_ns:\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n')

    total = 0
    tr_seismic, tr_noise = 0, 0

    # For every trace in the file
    for trace in wvs2_ns:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total += 1

        if pred_trace:
            tr_seismic += 1
        else:
            tr_noise += 1

    # Results
    print(f'Inferencia Waves2_ns:\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n')

    total = 0
    tr_seismic, tr_noise = 0, 0

    # For every trace in the file
    for trace in wvs3_ns:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total += 1

        if pred_trace:
            tr_seismic += 1
        else:
            tr_noise += 1

    # Results
    print(f'Inferencia Waves3_ns:\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n')


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
