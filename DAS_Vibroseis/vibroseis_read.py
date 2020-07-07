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

    # f = '../Data_Vibroseis/PoroTomo_iDAS025_160325140047.sgy'
    # f = '../Data_Vibroseis/PoroTomo_iDAS025_160325140117.sgy'
    # f = '../Data_Vibroseis/PoroTomo_iDAS16043_160325140048.sgy'
    f = '../Data_Vibroseis/PoroTomo_iDAS16043_160325140118.sgy'

    # 8721 trazas de 30000 muestras

    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    t_ax = np.arange(1, len(traces[0]) + 1) / fs

    trace1 = signal.resample(traces[0], 6000)
    trace2 = signal.resample(traces[100], 6000)
    trace3 = signal.resample(traces[200], 6000)

    trace1_fil = butter_bandpass_filter(trace1, 0.1, 10, fs, order=3)
    trace2_fil = butter_bandpass_filter(trace2, 0.1, 10, fs, order=3)
    trace3_fil = butter_bandpass_filter(trace3, 0.1, 10, fs, order=3)

    trace1 = trace1 / np.max(np.abs(trace1))
    trace2 = trace2 / np.max(np.abs(trace2))
    trace3 = trace3 / np.max(np.abs(trace3))

    trace1_fil = trace1_fil / np.max(np.abs(trace1_fil))
    trace2_fil = trace2_fil / np.max(np.abs(trace2_fil))
    trace3_fil = trace3_fil / np.max(np.abs(trace3_fil))

    plt.figure()
    plt.subplot(311)
    plt.plot(t_ax, trace1)
    plt.grid(True)
    plt.ylabel('Strain [-]')
    plt.xlabel('Tiempo [s]')
    plt.title('Trazas DAS datos Nevada')

    plt.subplot(312)
    plt.plot(t_ax, trace2)
    plt.grid(True)
    plt.ylabel('Strain [-]')
    plt.xlabel('Tiempo [s]')

    plt.subplot(313)
    plt.plot(t_ax, trace2)
    plt.grid(True)
    plt.ylabel('Strain [-]')
    plt.xlabel('Tiempo [s]')
    plt.tight_layout()
    plt.savefig('Imgs/TrazasDAS.png')

    plt.clf()
    plt.subplot(311)
    plt.plot(t_ax, trace1_fil)
    plt.grid(True)
    plt.ylabel('Strain [-]')
    plt.xlabel('Tiempo [s]')
    plt.title('Trazas DAS datos Vibroseis filtrados 1 - 10 Hz')

    plt.subplot(312)
    plt.plot(t_ax, trace2_fil)
    plt.grid(True)
    plt.ylabel('Strain [-]')
    plt.xlabel('Tiempo [s]')

    plt.subplot(313)
    plt.plot(t_ax, trace3_fil)
    plt.grid(True)
    plt.ylabel('Strain [-]')
    plt.xlabel('Tiempo [s]')
    plt.tight_layout()
    plt.savefig('Imgs/TrazasDAS_fil.png')

    plt.clf()
    line_st, = plt.plot(trace1, label='DAS')
    line_das, = plt.plot(st_trace, label='STEAD')
    plt.grid(True)
    plt.xlabel('Muestras [-]')
    plt.ylabel('Strain [-]')
    plt.title('Traza STEAD y traza DAS Vibroseis')
    plt.legend(handles=[line_st, line_das], loc='upper left')
    plt.savefig('Imgs/STEADVibroseis.png')

    plt.clf()
    plt.subplot(211)
    plt.plot(st_trace)
    plt.grid(True)
    plt.xlabel('Muestras [-]')
    plt.ylabel('Strain [-]')
    plt.title('Traza STEAD y traza DAS Vibroseis')
    plt.subplot(212)
    plt.plot(trace1)
    plt.grid(True)
    plt.xlabel('Muestras [-]')
    plt.ylabel('Strain [-]')
    plt.savefig('Imgs/STEADVibroseis1.png')


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
