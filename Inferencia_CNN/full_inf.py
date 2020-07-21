import re
import torch
import segyio
import argparse
import scipy.io
import numpy as np

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
    net.load_state_dict(torch.load('../STEAD_CNN/models/' + args.model_name + '.pth'))
    net.eval()

    # Load Francia dataset
    f = scipy.io.loadmat("../Data_Francia/Earthquake_1p9_Var_BP_2p5_15Hz.mat")

    # Read data
    data = f["StrainFilt"]

    # Count traces
    total_seismic = 0
    tp, fn = 0, 0

    # For every trace in the file
    for trace in data:
        if np.max(np.abs(trace)):
            # Normalize
            trace = trace / np.max(np.abs(trace))

            # Numpy to Torch
            trace = torch.from_numpy(trace).to(device).unsqueeze(0)

            # Prediction
            out_trace = net(trace.float())
            pred_trace = torch.round(out_trace.data).item()

            # Count traces
            total_seismic += 1

            if pred_trace:
                tp += 1
            else:
                fn += 1

    # Load Nevada dataset file 717
    f = '../Data_Nevada/PoroTomo_iDAS025_160321073717.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6000)

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_seismic += 1

        if pred_trace:
            tp += 1
        else:
            fn += 1

    # Load Nevada dataset file 747
    f = '../Data_Nevada/PoroTomo_iDAS025_160321073747.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6000)

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_seismic += 1

        if pred_trace:
            tp += 1
        else:
            fn += 1

    # Load Nevada data file 721
    f = '../Data_Nevada/PoroTomo_iDAS16043_160321073721.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        data = segyio.tools.collect(segy.trace[:])

        # Read data
        with segyio.open(f, ignore_geometry=True) as segy:
            segy.mmap()

            # Traces
            traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6000)

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_seismic += 1

        if pred_trace:
            tp += 1
        else:
            fn += 1

    # Load Nevada data file 751
    f = '../Data_Nevada/PoroTomo_iDAS16043_160321073751.sgy'

    # For every trace in the file
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        data = segyio.tools.collect(segy.trace[:])

        # Read data
        with segyio.open(f, ignore_geometry=True) as segy:
            segy.mmap()

            # Traces
            traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6000)

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_seismic += 1

        if pred_trace:
            tp += 1
        else:
            fn += 1

    # Load Belgica data
    f = scipy.io.loadmat("../Data_Belgica/mat_2018_08_19_00h28m05s_Parkwind_HDAS_2Dmap_StrainData_2D.mat")

    # Read data
    traces = f['Data_2D']

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6000)

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_seismic += 1

        if pred_trace:
            tp += 1
        else:
            fn += 1

    # Average 5km of measurements
    avg_data = np.mean(data[3500:4001, :], 0)

    # Numpy to Torch
    avg_data = torch.from_numpy(avg_data).to(device).unsqueeze(0)

    # Prediction
    output = net(avg_data.float())

    predicted = torch.round(output.data).item()

    if predicted:
        tp += 1
    else:
        fn += 1

    # Reykjanes telesismo fibra optica
    file_fo = '../Data_Reykjanes/Jousset_et_al_2018_003_Figure3_fo.ascii'

    # Dict for header and data
    data_fo = {
        'head': '',
        'strain': []
    }

    # Read fo file and save content to data_fo
    with open(file_fo, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data_fo['head'] = line.strip()
            else:
                val = line.strip()
                data_fo['strain'].append(float(val))

    # Resample
    data_fo['strain'] = signal.resample(data_fo['strain'], 6000)

    # Normalize
    data_fo['strain'] = data_fo['strain'] / np.max(np.abs(data_fo['strain']))

    # Numpy to Torch
    data_fo['strain'] = torch.from_numpy(data_fo['strain']).to(device).unsqueeze(0)

    # Prediction
    out = net(data_fo['strain'].float())
    predicted = torch.round(out.data).item()

    if predicted:
        tp += 1
    else:
        fn += 1

    # Registro de sismo local con DAS
    file = '../Data_Reykjanes/Jousset_et_al_2018_003_Figure5b.ascii'
    n_trazas = 2551

    data = {
        'head': '',
        'strain': np.empty((1, n_trazas))
    }

    with open(file, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data['head'] = line.strip()

            else:
                row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
                data['strain'] = np.concatenate((data['strain'], np.expand_dims(row, 0)))

    data['strain'] = data['strain'][1:]
    traces = data['strain'].transpose()

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6000)

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_seismic += 1

        if pred_trace:
            tp += 1
        else:
            fn += 1

    # Results
    print(f'Total traces: {total_seismic}\n'
          f'True positives: {tp}, False negatives: {fn}')


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
