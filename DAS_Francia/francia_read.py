import h5py
import numpy as np

import scipy.io as sio
import scipy.fftpack as sfft
import scipy.signal as signal

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from scipy.signal import butter, lfilter


def main():
    # Create images and animations folder

    Path("Imgs").mkdir(exist_ok=True)
    Path("Animations").mkdir(exist_ok=True)

    # Load STEAD trace

    # st = '../Data_STEAD/Train_data.hdf5'
    #
    # with h5py.File(st, 'r') as h5_file:
    #     grp = h5_file['earthquake']['local']
    #     for idx, dts in enumerate(grp):
    #         st_trace = grp[dts][:, 0] / np.max(np.abs(grp[dts][:, 0]))
    #         break

    # Registro de 1 minuto de sismo M1.9 a 100 Km NE del cable
    # 6848 trazas de 6000 muestras
    f = sio.loadmat("../Data_Francia/Earthquake_1p9_Var_BP_2p5_15Hz.mat")

    traces = f["StrainFilt"]
    # time= f["Time"]
    # distance = f["Distance_fiber"]

    # sampling frequency
    fs = 100

    # traces to print
    trtp = [0, 3000, 3100, 4000]

    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Create figure for plotting
    plt.figure()

    # plot chosen traces
    for i in trtp:
        yf = sfft.fftshift(sfft.fft(traces[i]))

        plt.clf()
        plt.subplot(211)
        plt.plot(traces[i])
        plt.grid(True)
        plt.ylabel('Strain [-]')
        plt.xlabel('Tiempo [s]')
        plt.title(f'Serie de tiempo y espectro #{i} Francia')

        plt.subplot(212)
        plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.grid(True)
        plt.ylabel('Amplitud [-]')
        plt.xlabel('Frecuencia [Hz]')
        plt.tight_layout()
        plt.savefig(f'Imgs/Francia_trace_{i}.png')

    # # Create animation of whole data
    # fig_tr = plt.figure()
    # ims_tr = []
    #
    # for trace in traces:
    #     im_tr = plt.plot(t_ax, trace)
    #     plt.title('Trazas dataset Francia')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Tiempo [s]')
    #     plt.grid(True)
    #     ims_tr.append(im_tr)
    #
    # ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=50, blit=True, repeat=False)
    # ani_tr.save('Animations/Francia_traces.mp4')
    #
    # # Create animation of whole data spectrums
    # fig_sp = plt.figure()
    # ims_sp = []
    #
    # for trace in traces:
    #     yf = sfft.fftshift(sfft.fft(trace))
    #     im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.title('Espectro trazas dataset Francia')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.grid(True)
    #     ims_sp.append(im_sp)
    #
    # ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=50, blit=True, repeat=False)
    # ani_sp.save('Animations/Francia_spectrums.mp4')

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
