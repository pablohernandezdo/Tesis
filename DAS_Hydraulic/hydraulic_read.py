import h5py
import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter


def main():
    # Carga traza STEAD

    st = '../Data_STEAD/Train_data.hdf5'

    with h5py.File(st, 'r') as h5_file:
        grp = h5_file['earthquake']['local']
        for idx, dts in enumerate(grp):
            st_trace = grp[dts][:, 0] / np.max(np.abs(grp[dts][:, 0]))
            break

    # 959 canales, largo 6_002_723 muestras
    fi = '../Data_Hydraulic/CSULB500Pa600secP_141210183813.mat'

    with h5py.File(fi, 'r') as f:
        data = f['data'][()]
        fs = f['fs_f'][()]

    print('CSULB500Pa600secP_141210183813')
    print(data.shape)
    print(fs)

    fi = '../Data_Hydraulic/CSULB500Pa10secP141210174309.mat'

    with h5py.File(fi, 'r') as f:
        data = f['data'][()]
        fs = f['fs_f'][()]

    print('CSULB500Pa10secP141210174309')
    print(data.shape)
    print(fs)

    fi = '../Data_Hydraulic/CSULB500Pa100secP141210175257.mat'

    with h5py.File(fi, 'r') as f:
        data = f['data'][()]
        fs = f['fs_f'][()]

    print('CSULB500Pa100secP141210175257')
    print(data.shape)
    print(fs)

    #traces = data[:, :6000]

    # t_ax = np.arange(1, len(traces[0]) + 1) / fs
    # t_ax = np.squeeze(t_ax)
    #
    # trace1 = traces[0]
    # trace2 = traces[450]
    # trace3 = traces[900]
    #
    # trace1_fil = butter_bandpass_filter(trace1, 0.1, 10, fs, order=3)
    # trace2_fil = butter_bandpass_filter(trace2, 0.1, 10, fs, order=3)
    # trace3_fil = butter_bandpass_filter(trace3, 0.1, 10, fs, order=3)
    #
    # trace1 = trace1 / np.max(np.abs(trace1))
    # trace2 = trace2 / np.max(np.abs(trace2))
    # trace3 = trace3 / np.max(np.abs(trace3))
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
    # plt.title('Trazas DAS datos Hydraulic')
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
    # plt.title('Trazas DAS datos HYdraulic filtrados 1 - 10 Hz')
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
    # line_st, = plt.plot(trace1, label='DAS')
    # line_das, = plt.plot(st_trace, label='STEAD')
    # plt.grid(True)
    # plt.xlabel('Muestras [-]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS HYdraulic')
    # plt.legend(handles=[line_st, line_das], loc='upper left')
    # plt.savefig('Imgs/STEADHYdraulic.png')
    #
    # plt.clf()
    # plt.subplot(211)
    # plt.plot(st_trace)
    # plt.grid(True)
    # plt.xlabel('Muestras [-]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS HYdraulic')
    # plt.subplot(212)
    # plt.plot(trace1)
    # plt.grid(True)
    # plt.xlabel('Muestras [-]')
    # plt.ylabel('Strain [-]')
    # plt.savefig('Imgs/STEADHYdraulic1.png')


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
