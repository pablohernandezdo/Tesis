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
    net.load_state_dict(torch.load('../../STEAD_CNN/models/' + args.model_name + '.pth'))
    net.eval()

    # Load Belgica data
    f = scipy.io.loadmat("../../Data_Belgica/mat_2018_08_19_00h28m05s_Parkwind_HDAS_2Dmap_StrainData_2D.mat")

    # Read data
    data = f['Data_2D']

    fs = 10
    total = 0
    seis, ns = 0, 0

    # For every trace in the file
    for trace in data:
        # Resample
        resamp_trace = signal.resample(trace, 6000)

        # Normalize
        resamp_trace = resamp_trace / np.max(np.abs(resamp_trace))

        # Numpy to Torch
        resamp_trace = torch.from_numpy(resamp_trace).to(device).unsqueeze(0)

        # Prediction
        outputs = net(resamp_trace.float())
        predicted = torch.round(outputs.data).item()

        # Count traces
        total += 1

        if predicted:
            seis += 1

        else:
            ns += 1

    # Average 5km of measurements
    avg_data = np.mean(data[3500:4001, :], 0)

    # Filter average data
    avg_data_filtered1 = butter_bandpass_filter(avg_data, 0.5, 1, fs, order=5)
    avg_data_filtered2 = butter_bandpass_filter(avg_data, 0.2, 0.6, fs, order=5)
    avg_data_filtered3 = butter_bandpass_filter(avg_data, 0.1, 0.3, fs, order=5)

    # Resample
    avg_data = signal.resample(avg_data, 6000)
    avg_data_filtered1 = signal.resample(avg_data_filtered1, 6000)
    avg_data_filtered2 = signal.resample(avg_data_filtered2, 6000)
    avg_data_filtered3 = signal.resample(avg_data_filtered3, 6000)

    # Normalize
    avg_data = avg_data / np.max(np.abs(avg_data))
    avg_data_filtered1 = avg_data_filtered1 / np.max(np.abs(avg_data_filtered1))
    avg_data_filtered2 = avg_data_filtered2 / np.max(np.abs(avg_data_filtered2))
    avg_data_filtered3 = avg_data_filtered3 / np.max(np.abs(avg_data_filtered3))

    # Numpy to Torch
    avg_data = torch.from_numpy(avg_data).to(device).unsqueeze(0)
    avg_data_filtered1 = torch.from_numpy(avg_data_filtered1).to(device).unsqueeze(0)
    avg_data_filtered2 = torch.from_numpy(avg_data_filtered2).to(device).unsqueeze(0)
    avg_data_filtered3 = torch.from_numpy(avg_data_filtered3).to(device).unsqueeze(0)

    # Prediction
    output = net(avg_data.float())
    output_filtered1 = net(avg_data_filtered1.float())
    output_filtered2 = net(avg_data_filtered2.float())
    output_filtered3 = net(avg_data_filtered3.float())

    predicted = torch.round(output.data).item()
    predicted_filtered1 = torch.round(output_filtered1.data).item()
    predicted_filtered2 = torch.round(output_filtered2.data).item()
    predicted_filtered3 = torch.round(output_filtered3.data).item()

    # Results
    print(f'Inferencia Belgica:\n\n'
          f'Total traces: {total}\n'
          f'Total predicted seismic traces: {seis}\n'
          f'Total predicted noise traces: {ns}\n')

    print(f'Average predicted: {predicted}\n'
          f'Average filtered predicted 1: {predicted_filtered1}\n'
          f'Average filtered predicted 2: {predicted_filtered2}\n'
          f'Average filtered predicted 3: {predicted_filtered3}\n')


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
