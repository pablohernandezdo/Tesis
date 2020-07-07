import scipy.io

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import butter, lfilter

def main():
    # 4192 canales, 42000 muestra por traza

    f = scipy.io.loadmat("mat_2018_08_19_00h28m05s_Parkwind_HDAS_2Dmap_StrainData_2D.mat")

    data = f['Data_2D']
    plt_tr = 4000
    fs = 10

    t_ax = np.arange(len(data[plt_tr])) / fs

    avg_trace = np.mean(data[3500:4001, :], 0)

    fil1 = butter_bandpass_filter(avg_trace, 0.5, 1, fs, order=5)
    fil2 = butter_bandpass_filter(avg_trace, 0.2, 0.6, 10, order=5)
    fil3 = butter_bandpass_filter(avg_trace, 0.1, 0.3, 10, order=5)
    fil4 = butter_bandpass_filter(avg_trace, 0.02, 0.08, 10, order=3)

    f, t, Zxx = signal.stft(avg_trace, fs, nperseg=1000)

    f1, t1, Zxx1 = signal.stft(avg_trace, fs, nperseg=1000)
    f2, t2, Zxx2 = signal.stft(avg_trace, fs, nperseg=1000)
    f3, t3, Zxx3 = signal.stft(avg_trace, fs, nperseg=1000)
    f4, t4, Zxx4 = signal.stft(avg_trace, fs, nperseg=1000)

    plt.figure()
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(avg_trace))
    plt.title('ASD')
    plt.ylabel('ASD')
    plt.xlabel('ASD')
    plt.colorbar()
    plt.show()

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