import h5py
import numpy as np
import numpy.random as random

import matplotlib.pyplot as plt

import scipy.fftpack as sfft
import scipy.signal as signal
from scipy.signal import butter, lfilter

from pathlib import Path


def main():
    # Create images folder

    Path("Imgs").mkdir(exist_ok=True)

    # STEAD dataset path
    st = '../Data_STEAD/Train_data.hdf5'

    # Number of traces to plot
    n = 4

    # Traces to plot
    trtp = []

    with h5py.File(st, 'r') as h5_file:

        # Seismic traces group
        grp = h5_file['earthquake']['local']

        # Traces to plot ids
        trtp_ids = [8359, 11211, 16256, 21276]
        # trtp_ids = random.randint(0, high=len(grp), size=n).sort()

        for idx, dts in enumerate(grp):
            if idx in trtp_ids:
                trtp.append(grp[dts][:, 0])

    # Sampling frequency
    fs = 100

    # Data len
    N = len(trtp[0])

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
        plt.plot(t_ax, trace)
        plt.title(f'Traza STEAD y espectro #{trtp_ids[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(212)
        plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/STEAD{trtp_ids[idx]}')

    # plt.figure()
    # plt.plot(st_trace)
    # plt.show()


if __name__ == '__main__':
    main()
