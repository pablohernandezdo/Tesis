import h5py
import torch
import numpy as np

import matplotlib.pyplot as plt

import scipy.signal as signal

from scipy.fft import fft
from scipy.signal import butter, lfilter


def main():
    file_path = 'MiniTrain.hdf5'
    fs = 100

    with h5py.File(file_path, 'r') as h5_file:
        grp1 = h5_file['earthquake']['local']
        grp2 = h5_file['non_earthquake']['noise']

        traces_len = len(grp1)
        noise_len = len(grp2)

        for idx, dts in enumerate(grp1):
            out = grp1[dts][:, 0] / np.max(grp1[dts][:, 0])
            #out_fil = butter_bandpass_filter(out, 1, 20, 100, order=5)
            break

        # Features

        # Spectral Centroid
        # spCen =

        _, _, Sxx = signal.spectrogram(out, fs)
        print(Sxx.shape)

        [C, CMean, CSD, CMax] = spCen(out, fs)


def maxAmp(data):
    return np.max(data)

def spCen(data, fs):
    _, _, Sxx = signal.spectrogram(data, fs)

    arr = np.transpose((np.tile(np.arange(1, Sxx.shape[0]+1), [Sxx.shape[1], 1])))
    C = np.sum(np.divide(np.multiply(arr, np.abs(Sxx)), np.sum(np.abs(Sxx))))
    # Segun yo estos 3 ultimos parametros valen kqk
    CMean = np.mean(C)
    CSD = np.std(C)
    CMax = np.max(C)
    return [C, CMean, CSD, CMax]

def rect():
    pass

def rmsA(data):
    x = np.abs(fft(data))
    return np.sqrt(np.mean(x**2))

def maxPFA(data):
    x = np.abs(fft(data))
    return np.max(x)

def maxFA(data, fs):
    x = np.abs(fft(data))
    return np.abs((np.argmax(x) - len(data) // 2) / fs)

def maxPF_FA(data, fs):
    mpfa = maxPFA(data)
    mfa = maxFA(data, fs)
    return mpfa / mfa

def ccnAb2D(data):
    pass

def Dip(data):
    pass

def ccD():
    pass

def envelopD(data):
    pass

def envelopS(data):
    pass

def xp2S(data):
    pass

def semD():
    pass

def DipRec():
    pass

def xotsu():
    pass

def semblanceD():
    pass

def skwnss():
    pass

def udD():
    pass

def ccnAb2S():
    pass

def udS():
    pass

def ypicD():
    pass

def ypicS():
    pass

def maxPF_FA():
    pass

def azmth():
    pass

def xp2D():
    pass

def xp2D():
    pass

def ccnRel2S():
    pass

def maxAmp(data):
    return np.max(data)

def ccS():
    pass

def indAngle():
    pass

def enrg(data):
    pass

def semblanceS():
    pass

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


if __name__ == '__main__':
    main()
