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

    f = '../Data_Shaker/large shaker NEES_130910161319 (1).sgy'

    # 1984 trazas de 12600 muestras

    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    t_ax = np.arange(1, len(traces[0]) + 1) / fs

    trace1 = traces[0] / np.max(traces[0])
    trace2 = traces[100] / np.max(traces[100])
    trace3 = traces[200] / np.max(traces[200])

    trace1_fil = butter_bandpass_filter(trace1, 0.1, 10, fs, order=3)
    trace2_fil = butter_bandpass_filter(trace2, 0.1, 10, fs, order=3)
    trace3_fil = butter_bandpass_filter(trace3, 0.1, 10, fs, order=3)

    trace1_fil = trace1_fil / np.max(trace1_fil)
    trace2_fil = trace2_fil / np.max(trace2_fil)
    trace3_fil = trace3_fil / np.max(trace3_fil)

    plt.figure()
    plt.subplot(211)
    plt.plot(trace1)
    plt.subplot(212)
    plt.plot(trace1_fil)
    plt.savefig('tr1.png')

    plt.clf()
    plt.subplot(211)
    plt.plot(trace2)
    plt.subplot(212)
    plt.plot(trace2_fil)
    plt.savefig('tr2.png')

    plt.clf()
    plt.subplot(211)
    plt.plot(trace3)
    plt.subplot(212)
    plt.plot(trace3_fil)
    plt.savefig('tr3.png')



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
