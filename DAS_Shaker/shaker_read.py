import h5py
import segyio
import numpy as np
import numpy.random as random

import matplotlib.pyplot as plt

import scipy.fftpack as sfft
import scipy.signal as signal
from scipy.signal import butter, lfilter

from pathlib import Path


def main():
    # Create images and animations folder

    Path("Imgs").mkdir(exist_ok=True)

    # Carga traza STEAD

    st = '../Data_STEAD/Train_data.hdf5'

    with h5py.File(st, 'r') as h5_file:
        grp = h5_file['earthquake']['local']
        for idx, dts in enumerate(grp):
            st_trace = grp[dts][:, 0] / np.max(np.abs(grp[dts][:, 0]))
            break

    # 1984 trazas de 12600 muestras
    f = '../Data_Shaker/large shaker NEES_130910161319 (1).sgy'

    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        # fs = segy.header[0][117] NO ES LA REAL

    # Sampling frequency
    fs = 200
    
    # Number of traces to plot
    n = 4

    # Traces to plot
    trtp = []

    # Traces to plot numbers
    trtp_ids = random.randint(0, high=len(traces), size=n)

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

    # Plot n random traces with their spectrum
    for idx, trace in enumerate(trtp):
        yf = sfft.fftshift(sfft.fft(trace))

        plt.clf()
        plt.subplot(211)
        plt.plot(t_ax, trace)
        plt.title(f'Traza Shaker y espectro #{trtp_ids[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(212)
        plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/Shaker_{trtp_ids[idx]}')

    # t_ax = np.arange(1, len(traces[0]) + 1) / fs
    #
    # trace1 = traces[0]
    # trace2 = traces[100]
    # trace3 = traces[200]
    #
    # trace1_resamp = signal.resample(traces[0], 6000)
    # trace2_resamp = signal.resample(traces[100], 6000)
    # trace3_resamp = signal.resample(traces[200], 6000)
    #
    # trace1_fil = butter_bandpass_filter(trace1, 0.1, 10, fs, order=3)
    # trace2_fil = butter_bandpass_filter(trace2, 0.1, 10, fs, order=3)
    # trace3_fil = butter_bandpass_filter(trace3, 0.1, 10, fs, order=3)
    #
    # trace1_resamp = trace1_resamp / np.max(np.abs(trace1_resamp))
    # trace2_resamp = trace2_resamp / np.max(np.abs(trace2_resamp))
    # trace3_resamp = trace3_resamp / np.max(np.abs(trace3_resamp))
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
    # plt.title('Trazas DAS datos Shaker')
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
    # plt.title('Trazas DAS datos Shaker filtrados 1 - 10 Hz')
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
    # line_st, = plt.plot(trace1_resamp, label='DAS')
    # line_das, = plt.plot(st_trace, label='STEAD')
    # plt.grid(True)
    # plt.xlabel('Muestras [-]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Shaker')
    # plt.legend(handles=[line_st, line_das], loc='upper left')
    # plt.savefig('Imgs/STEADShaker.png')
    #
    # plt.clf()
    # plt.subplot(211)
    # plt.plot(st_trace)
    # plt.grid(True)
    # plt.xlabel('Muestras [-]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Shaker')
    # plt.subplot(212)
    # plt.plot(trace1_resamp)
    # plt.grid(True)
    # plt.xlabel('Muestras [-]')
    # plt.ylabel('Strain [-]')
    # plt.savefig('Imgs/STEADShaker1.png')


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
