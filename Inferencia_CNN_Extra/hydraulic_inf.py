import h5py
import torch
import argparse
import numpy as np

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

    # File name
    file = '../../Data_Hydraulic/CSULB500Pa600secP_141210183813.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        data = f['data'][()]
        fs = f['fs_f'][()]
        sp = f['spatial_samp'][()]

    # Reshape 6000 first samples
    data = data[:, :6000]

    total = 0
    tr_seismic, tr_noise = 0, 0
    fil_seismic, fil_noise = 0, 0

    # For every trace in the file
    for trace in data:
        # Filter
        fil_trace = butter_bandpass_filter(trace, 0.5, 1, fs, order=3)

        # Normalize
        resamp_trace = trace / np.max(np.abs(trace))
        resamp_fil_trace = fil_trace / np.max(np.abs(fil_trace))

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
        else:
            tr_noise += 1

        if pred_fil_trace:
            fil_seismic += 1
        else:
            fil_noise += 1

    # Results
    print(f'Inferencia Hydraulic:\n\n'
          f'File: CSULB500Pa600secP_141210183813.mat\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n'
          f'Predicted fil_seismic: {fil_seismic}, predicted fil_noise: {fil_noise}\n')

    file = '../../Data_Hydraulic/CSULB500Pa10secP_141210174309.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        data = f['data'][()]
        fs = f['fs_f'][()]
        sp = f['spatial_samp'][()]

    # Reshape 6000 first samples
    data = data[:, :6000]

    total = 0
    tr_seismic, tr_noise = 0, 0
    fil_seismic, fil_noise = 0, 0

    # For every trace in the file
    for trace in data:
        # Filter
        fil_trace = butter_bandpass_filter(trace, 0.5, 1, fs, order=3)

        # Normalize
        resamp_trace = trace / np.max(np.abs(trace))
        resamp_fil_trace = fil_trace / np.max(np.abs(fil_trace))

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
        else:
            tr_noise += 1

        if pred_fil_trace:
            fil_seismic += 1
        else:
            fil_noise += 1

    # Results
    print(f'File: CSULB500Pa10secP_141210174309.mat\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n'
          f'Predicted fil_seismic: {fil_seismic}, predicted fil_noise: {fil_noise}\n')

    file = '../../Data_Hydraulic/CSULB500Pa100secP_141210175257.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        data = f['data'][()]
        fs = f['fs_f'][()]
        sp = f['spatial_samp'][()]

    # Reshape 6000 first samples
    data = data[:, :6000]

    total = 0
    tr_seismic, tr_noise = 0, 0
    fil_seismic, fil_noise = 0, 0

    # For every trace in the file
    for trace in data:
        # Filter
        fil_trace = butter_bandpass_filter(trace, 0.5, 1, fs, order=3)

        # Normalize
        resamp_trace = trace / np.max(np.abs(trace))
        resamp_fil_trace = fil_trace / np.max(np.abs(fil_trace))

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
        else:
            tr_noise += 1

        if pred_fil_trace:
            fil_seismic += 1
        else:
            fil_noise += 1

    # Results
    print(f'File: CSULB500Pa100secP_141210175257.mat\n'
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
