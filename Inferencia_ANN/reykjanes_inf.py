import re
import torch
import argparse
import numpy as np

from scipy import signal
from scipy.signal import butter, lfilter

from model import *


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='models/XXL_lr000001_bs32.pth', help="Classifier model path")
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

    # Load data Fig. 3f0 y 3bb
    file_fo = '../../Data_Reykjanes/Jousset_et_al_2018_003_Figure3_fo.ascii'
    file_bb = '../../Data_Reykjanes/Jousset_et_al_2018_003_Figure3_bb.ascii'

    # Sampling frequency
    fs = 20

    # Dict for header and data
    data_fo = {
        'head': '',
        'strain': []
    }

    data_bb = {
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

    # Read bb file and save content to data_bb
    with open(file_bb, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data_bb['head'] = line.strip()
            else:
                val = line.strip()
                data_bb['strain'].append(float(val))

    # Filter
    fil_fo = butter_bandpass_filter(data_fo['strain'], 0.5, 1, fs, order=5)
    fil_bb = butter_bandpass_filter(data_bb['strain'], 0.5, 1, fs, order=5)

    # Resample
    resamp_fo = signal.resample(data_fo['strain'], 6000)
    resamp_bb = signal.resample(data_bb['strain'], 6000)
    resamp_fil_fo = signal.resample(fil_fo, 6000)
    resamp_fil_bb = signal.resample(fil_bb, 6000)

    # Normalize
    resamp_fo = resamp_fo / np.max(np.abs(resamp_fo))
    resamp_bb = resamp_bb / np.max(np.abs(resamp_bb))
    resamp_fil_fo = resamp_fil_fo / np.max(np.abs(resamp_fil_fo))
    resamp_fil_bb = resamp_fil_fo / np.max(np.abs(resamp_fil_bb))

    # Numpy to Torch
    resamp_fo = torch.from_numpy(resamp_fo).to(device).unsqueeze(0)
    resamp_bb = torch.from_numpy(resamp_bb).to(device).unsqueeze(0)
    resamp_fil_fo = torch.from_numpy(resamp_fil_fo).to(device).unsqueeze(0)
    resamp_fil_bb = torch.from_numpy(resamp_fil_bb).to(device).unsqueeze(0)

    # Prediction
    out_fo = net(resamp_fo.float())
    out_bb = net(resamp_bb.float())
    out_fil_fo = net(resamp_fil_fo.float())
    out_fil_bb = net(resamp_fil_bb.float())

    predicted_fo = torch.round(out_fo.data).item()
    predicted_bb = torch.round(out_bb.data).item()
    predicted_fil_fo = torch.round(out_fil_fo.data).item()
    predicted_fil_bb = torch.round(out_fil_bb.data).item()

    # Results
    print(f'Inferencia Reykjanes:\n\n'
          f'File fo: {file_fo.split("/")[-1]}\n'
          f'out_fo: {out_fo.data.item()}, predicted_fo: {predicted_fo}\n'
          f'out_fil_fo: {out_fil_fo.data.item()}, predicted_fil_fo: {predicted_fil_fo}\n\n'
          f'File bb: {file_bb.split("/")[-1]}\n'
          f'out_bb: {out_bb.data.item()}, predicted_bb: {predicted_bb}\n'
          f'out_fil_bb: {out_fil_bb.data.item()}, predicted_fil_bb: {predicted_fil_bb}\n')

    # Load data Fig. 5a_fo
    file = '../../Data_Reykjanes/Jousset_et_al_2018_003_Figure5a_fo.ascii'

    # Number of traces and sampling frequency
    n_trazas = 26
    fs = 200

    # Dict for header and data
    data = {
        'head': '',
        'strain': np.empty((1, n_trazas))
    }

    # Read file and save contents to data
    with open(file, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data['head'] = line.strip()

            else:
                row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
                data['strain'] = np.concatenate((data['strain'], np.expand_dims(row, 0)))

    # Remove empty trace and transpose
    data['strain'] = data['strain'][1:]
    data['strain'] = data['strain'].transpose()

    # True/False Positives/Negatives
    total = 0
    tr_seismic, tr_noise = 0, 0
    fil_seismic, fil_noise = 0, 0

    # For every trace in the file
    for trace in data['strain']:
        # Filter
        fil_trace = butter_bandpass_filter(trace, 0.5, 1, fs, order=5)

        # Resample
        resamp_trace = signal.resample(trace, 6000)
        resamp_fil_trace = signal.resample(fil_trace, 6000)

        # Normalize
        resamp_trace = resamp_trace / np.max(np.abs(resamp_trace))
        resamp_fil_trace = resamp_fil_trace / np.max(np.abs(resamp_fil_trace))

        # Numpy to torch
        resamp_trace = torch.from_numpy(resamp_trace).to(device).unsqueeze(0)
        resamp_fil_trace = torch.from_numpy(resamp_fil_trace).to(device).unsqueeze(0)

        # Predicition
        out_trace = net(resamp_trace.float())
        out_fil_trace = net(resamp_fil_trace.float())

        pred_trace = torch.round(out_trace.data).item()
        pred_fil_trace = torch.round(out_fil_trace.data).item()

        # Count traces
        if pred_trace:
            tr_seismic += 1
        else:
            tr_noise += 1

        if pred_fil_trace:
            fil_seismic += 1
        else:
            fil_noise += 1

        total += 1

    # Results
    print(f'File: {file.split("/")[-1]}\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n'
          f'Predicted fil_seismic: {fil_seismic}, predicted fil_noise: {fil_noise}\n')

    # Load data Fig. 5a_gph
    file = '../../Data_Reykjanes/Jousset_et_al_2018_003_Figure5a_gph.ascii'

    # Number of traces and sampling frequency
    n_trazas = 26
    fs = 200

    # Dict for header and data
    data = {
        'head': '',
        'strain': np.empty((1, n_trazas))
    }

    # Read file and save contents to data
    with open(file, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data['head'] = line.strip()

            else:
                row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
                data['strain'] = np.concatenate((data['strain'], np.expand_dims(row, 0)))

    # Remove empty trace and transpose
    data['strain'] = data['strain'][1:]
    data['strain'] = data['strain'].transpose()

    # True/False Positives/Negatives
    total = 0
    tr_seismic, tr_noise = 0, 0
    fil_seismic, fil_noise = 0, 0

    for trace in data['strain']:
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
        if pred_trace:
            tr_seismic += 1
        else:
            tr_noise += 1

        if pred_fil_trace:
            fil_seismic += 1
        else:
            fil_noise += 1

        total += 1

    # Results
    print(f'File: {file.split("/")[-1]}\n'
          f'Total traces: {total}\n'
          f'Predicted seismic: {tr_seismic}, predicted noise: {tr_noise}\n'
          f'Predicted fil_seismic: {fil_seismic}, predicted fil_noise: {fil_noise}')


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
