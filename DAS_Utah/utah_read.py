import h5py
import segyio
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

    f = '../Data_Utah/FORGE_78-32_iDASv3-P11_UTC190419001218.sgy'

    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    print(traces.shape)
    print(fs)



if __name__ == '__main__':
    main()
