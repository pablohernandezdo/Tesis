import h5py
import numpy as np

import scipy.io as sio
import scipy.fftpack as sfft
import scipy.signal as signal

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.signal import butter, lfilter


def main():
    # Carga traza STEAD

    st = '../Data_STEAD/Train_data.hdf5'

    with h5py.File(st, 'r') as h5_file:
        grp = h5_file['earthquake']['local']
        for idx, dts in enumerate(grp):
            st_trace = grp[dts][:, 0] / np.max(np.abs(grp[dts][:, 0]))
            break

    # Registro de 1 minuto de sismo M1.9 a 100 Km NE del cable

    f = sio.loadmat("../Data_Francia/Earthquake_1p9_Var_BP_2p5_15Hz.mat")

    traces = f["StrainFilt"]
    # time= f["Time"]
    # distance = f["Distance_fiber"]
    plt_tr = 3000
    fs = 100
    N = len(traces[0])

    fig = plt.figure()

    ims = []
    for trace in traces:
        im = plt.plot(trace, animated=True)
        ims.append(im)

    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
                                    repeat=False)
    ani.save('traces.mp4')

    fig = plt.figure()

    ims = []
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    for trace in traces:
        im = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)), animated=True)
        ims.append(im)

    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
                                    repeat=False)

    ani.save('spectrums.mp4')

    # for trace in traces:
    #     yf = sfft.fftshift(sfft.fft(trace))
    #     xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)
    #     plt.clf()
    #     plt.subplot(211)
    #     plt.plot(trace)
    #     plt.subplot(212)
    #     plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.show(block=False)
    #     plt.pause(1.5)
    #     plt.close()

    # t_ax = np.arange(len(traces[plt_tr])) / fs
    #
    # trace1 = traces[0] / np.max(traces[0])
    # trace2 = traces[2000] / np.max(traces[2000])
    # trace3 = traces[5000] / np.max(traces[5000])
    #
    # trace1_fil = butter_bandpass_filter(trace1, 0.1, 10, fs, order=3)
    # trace2_fil = butter_bandpass_filter(trace2, 0.1, 10, fs, order=3)
    # trace3_fil = butter_bandpass_filter(trace3, 0.1, 10, fs, order=3)
    #
    # trace1_fil = trace1_fil / np.max(trace1_fil)
    # trace2_fil = trace2_fil / np.max(trace2_fil)
    # trace3_fil = trace3_fil / np.max(trace3_fil)
    #
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(t_ax, trace1)
    # plt.grid(True)
    # plt.ylabel('Strain [-]')
    # plt.xlabel('Tiempo [s]')
    # plt.title('Trazas DAS datos Francia')
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
    # plt.title('Trazas DAS datos Francia filtrados')
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
    # line_st, = plt.plot(signal.resample(trace3, 6000), label='DAS')
    # line_das, = plt.plot(st_trace, label='STEAD')
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Francia')
    # plt.legend(handles=[line_st, line_das], loc='upper left')
    # plt.savefig('Imgs/STEADFrancia.png')
    #
    # plt.clf()
    # plt.subplot(211)
    # plt.plot(st_trace)
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Francia')
    # plt.subplot(212)
    # plt.plot(signal.resample(trace3_fil, 6000))
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.savefig('Imgs/STEADFrancia1.png')


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


if __name__ == "__main__":
    main()
