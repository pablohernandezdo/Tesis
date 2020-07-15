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
        trtp_ids = random.randint(0, high=len(grp), size=(1, n))

        for idx, dts in enumerate(grp):
            if idx in trtp_ids:
                trtp.append(grp[dts][:, 0])

    print(len(trtp))

    # plt.figure()
    # plt.plot(st_trace)
    # plt.show()


if __name__ == '__main__':
    main()
