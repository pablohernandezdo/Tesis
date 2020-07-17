import h5py
import numpy as np
from numpy.random import default_rng

import scipy.fftpack as sfft
import scipy.signal as signal
from scipy.signal import butter, lfilter

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path


def main():
    # Create images and animations folder

    Path("Imgs").mkdir(exist_ok=True)
    Path("Imgs/500Pa10sec").mkdir(exist_ok=True)
    Path("Imgs/500Pa100sec").mkdir(exist_ok=True)
    Path("Imgs/500Pa600sec").mkdir(exist_ok=True)

    # Carga traza STEAD

    # st = '../Data_STEAD/Train_data.hdf5'
    #
    # with h5py.File(st, 'r') as h5_file:
    #     grp = h5_file['earthquake']['local']
    #     for idx, dts in enumerate(grp):
    #         st_trace = grp[dts][:, 0] / np.max(np.abs(grp[dts][:, 0]))
    #         break

    # 959 canales, largo 119_999 muestras
    # 500Pa10sec
    fi = '../Data_Hydraulic/CSULB500Pa10secP_141210174309.mat'

    with h5py.File(fi, 'r') as f:
        traces = f['data'][()]
        fs = f['fs_f'][()]

    # Number of traces to plot
    n = 4

    # Traces to plot
    trtp = []

    # Init rng
    rng = default_rng()

    # Traces to plot numbers
    trtp_ids = rng.choice(len(traces), size=n, replace=False)
    trtp_ids.sort()

    # Retrieve selected traces
    for idx, trace in enumerate(traces):
        if idx in trtp_ids:
            trtp.append(trace)

    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Figure to plot
    plt.figure()

    # For trace in traces to print
    for idx, trace in enumerate(trtp):
        yf = sfft.fftshift(sfft.fft(trace))

        plt.clf()
        plt.subplot(211)
        plt.plot(np.squeeze(t_ax), trace)
        plt.title(f'Traza 500Pa10sec y espectro #{trtp_ids[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(212)
        plt.plot(np.squeeze(xf), np.abs(yf) / np.max(np.abs(yf)))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/500Pa10sec/500Pa10sec_{trtp_ids[idx]}')

    # 959 canales, largo 2_056_235 muestras
    # 500Pa100sec
    fi = '../Data_Hydraulic/CSULB500Pa100secP_141210175257.mat'

    with h5py.File(fi, 'r') as f:
        traces = f['data'][()]
        fs = f['fs_f'][()]

    # Number of traces to plot
    n = 4

    # Traces to plot
    trtp = []

    # Traces to plot numbers
    trtp_ids = rng.choice(len(traces), size=n, replace=False)
    trtp_ids.sort()

    # Retrieve selected traces
    for idx, trace in enumerate(traces):
        if idx in trtp_ids:
            trtp.append(trace)

    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Figure to plot
    plt.figure()

    # For trace in traces to print
    for idx, trace in enumerate(trtp):
        yf = sfft.fftshift(sfft.fft(trace))

        plt.clf()
        plt.subplot(211)
        plt.plot(np.squeeze(t_ax), trace)
        plt.title(f'Traza 500Pa100sec y espectro #{trtp_ids[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(212)
        plt.plot(np.squeeze(xf), np.abs(yf) / np.max(np.abs(yf)))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/500Pa100sec/500Pa100sec_{trtp_ids[idx]}')

    # 959 canales, largo 6_002_723 muestras
    # 500Pa600sec
    fi = '../Data_Hydraulic/CSULB500Pa600secP_141210183813.mat'

    with h5py.File(fi, 'r') as f:
        traces = f['data'][()]
        fs = f['fs_f'][()]

    # Number of traces to plot
    n = 4

    # Traces to plot
    trtp = []

    # Traces to plot numbers
    trtp_ids = rng.choice(len(traces), size=n, replace=False)
    trtp_ids.sort()

    # Retrieve selected traces
    for idx, trace in enumerate(traces):
        if idx in trtp_ids:
            trtp.append(trace)

    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Figure to plot
    plt.figure()

    # For trace in traces to print
    for idx, trace in enumerate(trtp):
        yf = sfft.fftshift(sfft.fft(trace))

        plt.clf()
        plt.subplot(211)
        plt.plot(np.squeeze(t_ax), trace)
        plt.title(f'Traza 500Pa600sec y espectro #{trtp_ids[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(212)
        plt.plot(np.squeeze(xf), np.abs(yf) / np.max(np.abs(yf)))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/500Pa600sec/500Pa600sec_{trtp_ids[idx]}')

    #traces = data[:, :6000]

    # t_ax = np.arange(1, len(traces[0]) + 1) / fs
    # t_ax = np.squeeze(t_ax)
    #
    # trace1 = traces[0]
    # trace2 = traces[450]
    # trace3 = traces[900]
    #
    # trace1_fil = butter_bandpass_filter(trace1, 0.1, 10, fs, order=3)
    # trace2_fil = butter_bandpass_filter(trace2, 0.1, 10, fs, order=3)
    # trace3_fil = butter_bandpass_filter(trace3, 0.1, 10, fs, order=3)
    #
    # trace1 = trace1 / np.max(np.abs(trace1))
    # trace2 = trace2 / np.max(np.abs(trace2))
    # trace3 = trace3 / np.max(np.abs(trace3))
    #
    # trace1_fil = trace1_fil / np.max(np.abs(trace1_fil))
    # trace2_fil = trace2_fil / np.max(np.abs(trace2_fil))
    # trace3_fil = trace3_fil / np.max(np.abs(trace3_fil))
    #
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(t_ax, trace1)
    # plt.grid(True)
    # plt.ylabel('Strain [-]')
    # plt.xlabel('Tiempo [s]')
    # plt.title('Trazas DAS datos Hydraulic')
    #
    # plt.subplot(312)
    # plt.plot(t_ax, trace2)
    # plt.grid(True)
    # plt.ylabel('Strain [-]')
    # plt.xlabel('Tiempo [s]')
    #
    # plt.subplot(313)
    # plt.plot(t_ax, trace3)
    # plt.grid(True)
    # plt.ylabel('Strain [-]')
    # plt.xlabel('Tiempo [s]')
    # plt.tight_layout()
    # plt.savefig('Imgs/TrazasDAS.png')
    #
    # plt.clf()
    # plt.subplot(311)
    # plt.plot(t_ax, trace1_fil)
    # plt.grid(True)
    # plt.ylabel('Strain [-]')
    # plt.xlabel('Tiempo [s]')
    # plt.title('Trazas DAS datos HYdraulic filtrados 1 - 10 Hz')
    #
    # plt.subplot(312)
    # plt.plot(t_ax, trace2_fil)
    # plt.grid(True)
    # plt.ylabel('Strain [-]')
    # plt.xlabel('Tiempo [s]')
    #
    # plt.subplot(313)
    # plt.plot(t_ax, trace3_fil)
    # plt.grid(True)
    # plt.ylabel('Strain [-]')
    # plt.xlabel('Tiempo [s]')
    # plt.tight_layout()
    # plt.savefig('Imgs/TrazasDAS_fil.png')
    #
    # plt.clf()
    # line_st, = plt.plot(trace1, label='DAS')
    # line_das, = plt.plot(st_trace, label='STEAD')
    # plt.grid(True)
    # plt.xlabel('Muestras [-]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS HYdraulic')
    # plt.legend(handles=[line_st, line_das], loc='upper left')
    # plt.savefig('Imgs/STEADHYdraulic.png')
    #
    # plt.clf()
    # plt.subplot(211)
    # plt.plot(st_trace)
    # plt.grid(True)
    # plt.xlabel('Muestras [-]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS HYdraulic')
    # plt.subplot(212)
    # plt.plot(trace1)
    # plt.grid(True)
    # plt.xlabel('Muestras [-]')
    # plt.ylabel('Strain [-]')
    # plt.savefig('Imgs/STEADHYdraulic1.png')


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


def butter_lowpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], output='ba')
    return b, a


def butter_lowpasspass_filter(dat, lowcut, highcut, fs, order=5):
    b, a = butter_lowpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, dat)
    return y


if __name__ == '__main__':
    main()
