import h5py

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
from scipy.signal import butter, lfilter


def main():
    # Carga traza STEAD

    st = '../Data_STEAD/Train_data.hdf5'

    with h5py.File(st, 'r') as h5_file:
        grp = h5_file['earthquake']['local']
        for idx, dts in enumerate(grp):
            st_trace = grp[dts][:, 0] / np.max(grp[dts][:, 0])
            break

    # 1 registro, largo 259_094_163 muestras
    file = '../Data_Tides/CSULB_T13_EarthTide_earthtide_mean_360_519.mat'

    with h5py.File(file, 'r') as f:
        print(f.keys())
        data = f['clipdata'][()]

    data = data / np.max(np.abs(data))
    data_cut = data[:(data.size // 6000) * 6000]
    data_reshape = data_cut.reshape((data.size // 6000, -1))

    # plt.figure()
    # plt.plot(data[:600001])
    # plt.savefig('data.png')


if __name__ == '__main__':
    main()