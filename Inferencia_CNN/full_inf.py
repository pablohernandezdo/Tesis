import re
import pywt
import h5py
import torch
import segyio
import argparse
import scipy.io
import numpy as np
from numpy.random import default_rng

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
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

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
    total_nseismic = 0
    tp, fp, tn, fn = 0, 0, 0, 0

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

    # Resample
    avg_data = signal.resample(avg_data, 6000)

    # Normalize
    avg_data = avg_data / np.max(np.abs(avg_data))

    # Numpy to Torch
    avg_data = torch.from_numpy(avg_data).to(device).unsqueeze(0)

    # Prediction
    output = net(avg_data.float())

    predicted = torch.round(output.data).item()

    total_nseismic += 1

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

    # Load California dataset file
    f = scipy.io.loadmat('../Data_California/FSE-06_480SecP_SingDec_StepTest (1).mat')

    # Read data
    traces = f['singdecmatrix']
    traces = traces.transpose()

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
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # File name
    file = '../Data_Hydraulic/CSULB500Pa600secP_141210183813.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        traces = f['data'][()]

    # Reshape 6000 first samples
    traces = traces[:, :6000]

    # For every trace in the file
    for trace in traces:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    file = '../Data_Hydraulic/CSULB500Pa10secP_141210174309.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        traces = f['data'][()]

    # Reshape 6000 first samples
    traces = traces[:, :6000]

    # For every trace in the file
    for trace in traces:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    file = '../Data_Hydraulic/CSULB500Pa100secP_141210175257.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        traces = f['data'][()]

    # Reshape 6000 first samples
    traces = traces[:, :6000]

    # For every trace in the file
    for trace in traces:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # File name
    file = '../Data_Tides/CSULB_T13_EarthTide_earthtide_mean_360_519.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        traces = f['clipdata'][()]

    # Data size
    sz = traces.size

    # Cut data to build traces matrix
    traces = traces[:(sz // 6000) * 6000]

    # Matrix of traces
    traces = traces.reshape((sz // 6000, -1))

    # For every trace in the file
    for trace in traces:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Load Nevada data file 1
    f = '../Data_Utah/FORGE_78-32_iDASv3-P11_UTC190419001218.sgy'

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
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Load Nevada dataset file 047
    f = '../Data_Vibroseis/PoroTomo_iDAS025_160325140047.sgy'

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
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Load Nevada dataset file 117
    f = '../Data_Vibroseis/PoroTomo_iDAS025_160325140117.sgy'

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
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Load Nevada dataset file 048
    f = '../Data_Vibroseis/PoroTomo_iDAS16043_160325140048.sgy'

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
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Load Vibroseis data file 118
    f = '../Data_Vibroseis/PoroTomo_iDAS16043_160325140118.sgy'

    # For every trace in the file
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
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Load Shaker dataset file
    f = '../Data_Shaker/large shaker NEES_130910161319 (1).sgy'

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
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Init rng
    rng = default_rng()

    # Noise
    ns = rng.normal(0, 1, 6000)

    # Sine waves

    # Number of sample points
    N = 6000

    # sampling frequency
    fs = 100

    # Time axis
    t = np.linspace(0.0, N / fs, N)

    # Number of frequency interval steps
    n = 100

    # Frequency spans
    fr1 = np.linspace(1, 50, n)
    fr2 = np.linspace(0.01, 1, n)

    # Prealocate
    wvs1 = []
    wvs2 = []
    wvs3 = []

    for f1, f2 in zip(fr1, fr2):
        sig1 = np.sin(f1 * 2.0 * np.pi * t)
        sig2 = np.sin(f2 * 2.0 * np.pi * t)
        wvs1.append(sig1)
        wvs2.append(sig2)
        wvs3.append(sig1 + sig2)

    wvs1 = np.array(wvs1)
    wvs2 = np.array(wvs2)
    wvs3 = np.array(wvs3)

    wvs1_ns = wvs1 + 0.5 * rng.normal(0, 1, wvs1.shape)
    wvs2_ns = wvs2 + 0.5 * rng.normal(0, 1, wvs2.shape)
    wvs3_ns = wvs3 + 0.5 * rng.normal(0, 1, wvs3.shape)

    # PADDED SINES

    # Number of intermediate sample points
    ni = [1000, 2000, 4000, 5000]

    # Number of points to zero-pad
    pad = [(N-n) // 2 for n in ni]

    # Time axis for smaller waves
    lts = [np.linspace(0.0, nis / fs, nis) for nis in ni]

    # All frequencies list
    all_fr = []

    # Calculate max period for smaller waves
    max_periods = [n_points / fs for n_points in ni]

    # Calculate frequencies for smaller waves
    for per in max_periods:
        freqs = []
        for i in range(1, int(per)+1):
            if per % i == 0:
                freqs.append(1/i)
        all_fr.append(freqs)

    # Preallocate waves
    wvs_pad = []

    # Generate waves and zero pad
    for idx, fr_ls in enumerate(all_fr):
        for fr in fr_ls:
            wv = np.sin(fr * 2.0 * np.pi * lts[idx])
            wv = np.pad(wv, (pad[idx], pad[idx]), 'constant')
            wvs_pad.append(wv)

    # Wavelets

    # Preallocate wavelets
    lets = []

    # Discrete wavelet families
    discrete_families = ['db', 'sym', 'coif', 'bior', 'rbio']

    # Obtain wavelet waveforms, resample and append
    for fam in discrete_families:
        for wavelet in pywt.wavelist(fam):
            wv = pywt.Wavelet(wavelet)
            if wv.orthogonal:
                [_, psi, _] = pywt.Wavelet(wavelet).wavefun(level=5)
                psi = signal.resample(psi, 6000)
                lets.append(psi)

    # Normalize
    ns_norm = ns / np.max(np.abs(ns))

    # Numpy to Torch
    ns_norm = torch.from_numpy(ns_norm).to(device).unsqueeze(0)

    # Prediction
    out = net(ns_norm.float())

    out = torch.round(out.data).item()

    total_nseismic += 1

    if out:
        fp += 1
    else:
        tn += 1

    # For every trace in wvs1
    for trace in wvs1:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # For every trace in wvs1
    for trace in wvs2:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # For every trace in wvs1
    for trace in wvs3:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # For every trace in wvs1
    for trace in wvs1_ns:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    for trace in wvs2_ns:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    for trace in wvs3_ns:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    for trace in wvs_pad:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    for trace in lets:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = torch.round(out_trace.data).item()

        # Count traces
        total_nseismic += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Evaluation metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * (precision * recall) / (precision + recall)

    # Results
    print(f'Total seismic traces: {total_seismic}\n'
          f'Total non seismic traces: {total_nseismic}\n'
          f'True positives: {tp}\n'
          f'False positives: {fp}\n'
          f'True negatives: {tn}\n'
          f'False negatives: {fn}\n\n'
          f'Evaluation metrics:\n\n'          
          f'Precision: {precision:5.3f}\n'
          f'Recall: {recall:5.3f}\n'
          f'F-score: {fscore:5.3f}\n')


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
