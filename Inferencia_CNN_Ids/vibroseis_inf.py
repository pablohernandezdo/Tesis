import torch
import segyio
import argparse
import numpy as np

import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import butter, lfilter

from model import *


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='CBN_10epch', help="Classifier model path")
    parser.add_argument("--classifier", default='CBN', help="Choose classifier architecture, C, CBN")
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
    net.load_state_dict(torch.load('../../STEAD_CNN/models/' + args.model_name + '.pth'))
    net.eval()

    # Load Nevada data file 1
    f = '../../Data_Vibroseis/PoroTomo_iDAS025_160325140047.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        data = segyio.tools.collect(segy.trace[:])

        # Sampling frequency
        fs = segy.header[0][117]

    total = 0
    tr_seismic, tr_noise = 0, 0
    fil_seismic, fil_noise = 0, 0

    seis_traces1 = []
    seis_fil_traces1 = []

    noise_traces1 = []
    noise_fil_traces1 = []

    # For every trace in the file
    for idx, trace in enumerate(data):
        # Filter
        fil_trace = butter_bandpass_filter(trace, 0.5, 1, fs, order=5)

        # Resample
        resamp_trace = signal.resample(trace, 6000)
        resamp_fil_trace = signal.resample(fil_trace, 6000)

        # Normalize
        resamp_trace = resamp_trace / np.max(np.abs(resamp_trace))
        resamp_fil_trace = resamp_fil_trace / np.max(np.abs(resamp_fil_trace))

        # Numpy to Torch
        resamp_trace = torch.from_numpy(resamp_trace).to(device).unsqueeze(0)
        resamp_fil_trace = torch.from_numpy(resamp_fil_trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(resamp_trace.float())
        out_fil_trace = net(resamp_fil_trace.float())

        pred_trace = torch.round(out_trace.data).item()
        pred_fil_trace = torch.round(out_fil_trace.data).item()

        # Count traces
        total += 1

        if pred_trace:
            tr_seismic += 1
            seis_traces1.append(idx)
        else:
            tr_noise += 1
            noise_traces1.append(idx)

        if pred_fil_trace:
            fil_seismic += 1
            seis_fil_traces1.append(idx)
        else:
            fil_noise += 1
            noise_fil_traces1.append(idx)

    seis_tr_id1 = np.random.choice(seis_traces1, 1)
    seis_fil_tr_id1 = np.random.choice(seis_fil_traces1, 1)

    noise_tr_id1 = np.random.choice(seis_traces1, 1)
    noise_fil_tr_id1 = np.random.choice(seis_fil_traces1, 1)

    plt.figure()
    plt.plot(data[seis_tr_id1])
    plt.savefig('seis_trace1.png')

    plt.clf()
    plt.plot(data[seis_fil_tr_id1])
    plt.savefig('seis_fil_trace1.png')

    plt.clf()
    plt.plot(data[noise_tr_id1])
    plt.savefig('noise_trace1.png')

    plt.clf()
    plt.plot(data[noise_fil_tr_id1])
    plt.savefig('noise_fil_trace1.png')

    # Results
    print(f'Inferencia Vibroseis:\n\n'
          f'File: PoroTomo_iDAS025_160325140047.sgy\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n'
          f'Predicted fil_seismic: {fil_seismic}, predicted fil_noise: {fil_noise}\n')

    # Load Nevada data file 2
    f = '../../Data_Vibroseis/PoroTomo_iDAS025_160325140117.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        data = segyio.tools.collect(segy.trace[:])

        # Sampling frequency
        fs = segy.header[0][117]

    total = 0
    tr_seismic, tr_noise = 0, 0
    fil_seismic, fil_noise = 0, 0

    seis_traces2 = []
    seis_fil_traces2 = []

    noise_traces2 = []
    noise_fil_traces2 = []

    # For every trace in the file
    for idx, trace in enumerate(data):
        # Filter
        fil_trace = butter_bandpass_filter(trace, 0.5, 1, fs, order=5)

        # Resample
        resamp_trace = signal.resample(trace, 6000)
        resamp_fil_trace = signal.resample(fil_trace, 6000)

        # Normalize
        resamp_trace = resamp_trace / np.max(np.abs(resamp_trace))
        resamp_fil_trace = resamp_fil_trace / np.max(np.abs(resamp_fil_trace))

        # Numpy to Torch
        resamp_trace = torch.from_numpy(resamp_trace).to(device).unsqueeze(0)
        resamp_fil_trace = torch.from_numpy(resamp_fil_trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(resamp_trace.float())
        out_fil_trace = net(resamp_fil_trace.float())

        pred_trace = torch.round(out_trace).data.item()
        pred_fil_trace = torch.round(out_fil_trace).data.item()

        # Count traces
        total += 1

        if pred_trace:
            tr_seismic += 1
            seis_traces2.append(idx)
        else:
            tr_noise += 1
            noise_traces2.append(idx)

        if pred_fil_trace:
            fil_seismic += 1
            seis_fil_traces2.append(idx)
        else:
            fil_noise += 1
            noise_fil_traces2.append(idx)

    seis_tr_id2 = np.random.choice(seis_traces2, 1)
    seis_fil_tr_id2 = np.random.choice(seis_fil_traces2, 1)

    noise_tr_id2 = np.random.choice(seis_traces2, 1)
    noise_fil_tr_id2 = np.random.choice(seis_fil_traces2, 1)

    plt.clf()
    plt.plot(data[seis_tr_id2])
    plt.savefig('seis_trace2.png')

    plt.clf()
    plt.plot(data[seis_fil_tr_id2])
    plt.savefig('seis_fil_trace2.png')

    plt.clf()
    plt.plot(data[noise_tr_id2])
    plt.savefig('noise_trace2.png')

    plt.clf()
    plt.plot(data[noise_fil_tr_id2])
    plt.savefig('noise_fil_trace2.png')

    # Results
    print(f'File: PoroTomo_iDAS025_160325140117.sgy\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n'
          f'Predicted fil_seismic: {fil_seismic}, predicted fil_noise: {fil_noise}\n')

    # Load Nevada data file 3
    f = '../../Data_Vibroseis/PoroTomo_iDAS16043_160325140048.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        data = segyio.tools.collect(segy.trace[:])

        # Sampling frequency
        fs = segy.header[0][117]

    total = 0
    tr_seismic, tr_noise = 0, 0
    fil_seismic, fil_noise = 0, 0

    seis_traces3 = []
    seis_fil_traces3 = []

    noise_traces3 = []
    noise_fil_traces3 = []

    # For every trace in the file
    for idx, trace in enumerate(data):
        # Filter
        fil_trace = butter_bandpass_filter(trace, 0.5, 1, fs, order=5)

        # Resample
        resamp_trace = signal.resample(trace, 6000)
        resamp_fil_trace = signal.resample(fil_trace, 6000)

        # Normalize
        resamp_trace = resamp_trace / np.max(np.abs(resamp_trace))
        resamp_fil_trace = resamp_fil_trace / np.max(np.abs(resamp_fil_trace))

        # Numpy to Torch
        resamp_trace = torch.from_numpy(resamp_trace).to(device).unsqueeze(0)
        resamp_fil_trace = torch.from_numpy(resamp_fil_trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(resamp_trace.float())
        out_fil_trace = net(resamp_fil_trace.float())

        pred_trace = torch.round(out_trace.data).item()
        pred_fil_trace = torch.round(out_fil_trace.data).item()

        # Count traces
        total += 1

        if pred_trace:
            tr_seismic += 1
            seis_traces3.append(idx)
        else:
            tr_noise += 1
            noise_traces3.append(idx)

        if pred_fil_trace:
            fil_seismic += 1
            seis_fil_traces3.append(idx)
        else:
            fil_noise += 1
            noise_fil_traces3.append(idx)

    seis_tr_id3 = np.random.choice(seis_traces3, 1)
    seis_fil_tr_id3 = np.random.choice(seis_fil_traces3, 1)

    noise_tr_id3 = np.random.choice(seis_traces3, 1)
    noise_fil_tr_id3 = np.random.choice(seis_fil_traces3, 1)

    plt.clf()
    plt.plot(data[seis_tr_id3])
    plt.savefig('seis_trace2.png')

    plt.clf()
    plt.plot(data[seis_fil_tr_id3])
    plt.savefig('seis_fil_trace2.png')

    plt.clf()
    plt.plot(data[noise_tr_id3])
    plt.savefig('noise_trace2.png')

    plt.clf()
    plt.plot(data[noise_fil_tr_id3])
    plt.savefig('noise_fil_trace2.png')

    # Results
    print(f'File: PoroTomo_iDAS16043_160325140048.sgy\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n'
          f'Predicted fil_seismic: {fil_seismic}, predicted fil_noise: {fil_noise}\n')

    # Load Nevada data file 4
    f = '../../Data_Vibroseis/PoroTomo_iDAS16043_160325140118.sgy'

    # For every trace in the file
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        data = segyio.tools.collect(segy.trace[:])

        # Sampling frequency
        fs = segy.header[0][117]

    total = 0
    tr_seismic, tr_noise = 0, 0
    fil_seismic, fil_noise = 0, 0

    seis_traces4 = []
    seis_fil_traces4 = []

    noise_traces4 = []
    noise_fil_traces4 = []

    # For every trace in the file
    for idx, trace in enumerate(data):
        # Filter
        fil_trace = butter_bandpass_filter(trace, 0.5, 1, fs, order=5)

        # Resample
        resamp_trace = signal.resample(trace, 6000)
        resamp_fil_trace = signal.resample(fil_trace, 6000)

        # Normalize
        resamp_trace = resamp_trace / np.max(np.abs(resamp_trace))
        resamp_fil_trace = resamp_fil_trace / np.max(np.abs(resamp_fil_trace))

        # Numpy to Torch
        resamp_trace = torch.from_numpy(resamp_trace).to(device).unsqueeze(0)
        resamp_fil_trace = torch.from_numpy(resamp_fil_trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(resamp_trace.float())
        out_fil_trace = net(resamp_fil_trace.float())

        pred_trace = torch.round(out_trace.data).item()
        pred_fil_trace = torch.round(out_fil_trace.data).item()

        # Count traces
        total += 1

        if pred_trace:
            tr_seismic += 1
            seis_traces4.append(idx)
        else:
            tr_noise += 1
            noise_traces4.append(idx)

        if pred_fil_trace:
            fil_seismic += 1
            seis_fil_traces4.append(idx)
        else:
            fil_noise += 1
            noise_fil_traces4.append(idx)

    seis_tr_id4 = np.random.choice(seis_traces4, 1)
    seis_fil_tr_id4 = np.random.choice(seis_fil_traces4, 1)

    noise_tr_id4 = np.random.choice(seis_traces4, 1)
    noise_fil_tr_id4 = np.random.choice(seis_fil_traces4, 1)

    plt.clf()
    plt.plot(data[seis_tr_id4])
    plt.savefig('seis_trace2.png')

    plt.clf()
    plt.plot(data[seis_fil_tr_id4])
    plt.savefig('seis_fil_trace2.png')

    plt.clf()
    plt.plot(data[noise_tr_id4])
    plt.savefig('noise_trace2.png')

    plt.clf()
    plt.plot(data[noise_fil_tr_id4])
    plt.savefig('noise_fil_trace2.png')


    # Results
    print(f'File: PoroTomo_iDAS16043_160325140118.sgy\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n'
          f'Predicted fil_seismic: {fil_seismic}, predicted fil_noise: {fil_noise}\n')


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
