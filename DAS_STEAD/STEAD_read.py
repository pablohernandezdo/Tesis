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

    # Carga traza STEAD

    st = '../Data_STEAD/Train_data.hdf5'

    #id = random.randint(0, high=10000, size=1)

    with h5py.File(st, 'r') as h5_file:
        grp = h5_file['earthquake']['local']
        print(len(grp))
        # for idx, dts in enumerate(grp):
        #     if id == idx:
        #         st_trace = grp[dts][:, 0] / np.max(np.abs(grp[dts][:, 0]))
        #         break
        #     else:
        #         continue

    # plt.figure()
    # plt.plot(st_trace)
    # plt.show()


if __name__ == '__main__':
    main()
