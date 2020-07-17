import h5py
import numpy as np
from numpy.random import default_rng

import scipy.io as sio
import scipy.fftpack as sfft
import scipy.signal as signal

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.signal import butter, lfilter

from pathlib import Path


def main():
    # Create images and animations folder

    Path("Imgs").mkdir(exist_ok=True)
    Path("Animations").mkdir(exist_ok=True)

    # # Carga traza STEAD
    #
    # st = '../Data_STEAD/Train_data.hdf5'
    #
    # with h5py.File(st, 'r') as h5_file:
    #     grp = h5_file['earthquake']['local']
    #     for idx, dts in enumerate(grp):
    #         st_trace = grp[dts][:, 0] / np.max(np.abs(grp[dts][:, 0]))
    #         break


    # Read file
    # 4192 canales, 42000 muestra por traza
    f = sio.loadmat("../Data_Belgica/mat_2018_08_19_00h28m05s_Parkwind_HDAS_2Dmap_StrainData_2D.mat")

    # Read data
    traces = f['Data_2D']
    plt_tr = 4000

    # Sampling frequency
    fs = 10

    # Number of traces to plot
    n = 4

    # Traces to plot
    trtp = []

    # Init rng
    rng = default_rng()

    # Traces to plot numbers
    trtp_ids = rng.choice(len(traces), size=n, replace=False)
    trtp_ids.sort()

    # Retrieve selected traces
    for idx, trace in enumerate(traces):
        if idx in trtp_ids:
            trtp.append(trace)

    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Figure to plot
    plt.figure()

    # Plot n random traces with their spectrum
    for idx, trace in enumerate(trtp):
        yf = sfft.fftshift(sfft.fft(trace))

        plt.clf()
        plt.subplot(211)
        plt.plot(t_ax, trace)
        plt.title(f'Traza Belgica y espectro #{trtp_ids[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(212)
        plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/Belgica_{trtp_ids[idx]}')

    # Obtain 5 km average trace
    avg_trace = np.mean(traces[3500:4001, :], 0)
    avg_fil1 = butter_bandpass_filter(avg_trace, 0.5, 1, fs, order=5)
    avg_fil2 = butter_bandpass_filter(avg_trace, 0.2, 0.6, 10, order=5)
    avg_fil3 = butter_bandpass_filter(avg_trace, 0.1, 0.3, 10, order=5)

    yf = sfft.fftshift(sfft.fft(avg_trace))
    yf_fil1 = sfft.fftshift(sfft.fft(avg_fil1))
    yf_fil2 = sfft.fftshift(sfft.fft(avg_fil2))
    yf_fil3 = sfft.fftshift(sfft.fft(avg_fil3))

    plt.clf()
    plt.subplot(211)
    plt.plot(t_ax, avg_trace)
    plt.title(f'Traza promedio Belgica y espectro')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Imgs/Belgica_avg')

    plt.clf()
    plt.subplot(211)
    plt.plot(t_ax, avg_fil1)
    plt.title(f'Traza promedio Belgica filtrada 0.5 - 1 Hz y espectro')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(xf, np.abs(yf_fil1) / np.max(np.abs(yf_fil1)))
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Imgs/Belgica_avg_fil1')

    plt.clf()
    plt.subplot(211)
    plt.plot(t_ax, avg_fil2)
    plt.title(f'Traza promedio Belgica filtrada 0.2 - 0.6 Hz y espectro')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(xf, np.abs(yf_fil2) / np.max(np.abs(yf_fil2)))
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Imgs/Belgica_avg_fil2')

    plt.clf()
    plt.subplot(211)
    plt.plot(t_ax, avg_fil3)
    plt.title(f'Traza promedio Belgica filtrada 0.1 - 0.3 Hz y espectro')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(xf, np.abs(yf_fil3) / np.max(np.abs(yf_fil3)))
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Imgs/Belgica_avg_fil3')


# t_ax = np.arange(len(traces[plt_tr])) / fs
    #
    # trace1 = traces[1000] / np.max(traces[1000])
    # trace2 = traces[2000] / np.max(traces[2000])
    # trace3 = traces[3000] / np.max(traces[3000])
    #
    # trace1_fil = butter_bandpass_filter(trace1, 0.5, 1, fs, order=5)
    # trace2_fil = butter_bandpass_filter(trace2, 0.5, 1, fs, order=5)
    # trace3_fil = butter_bandpass_filter(trace3, 0.5, 1, fs, order=5)
    #
    # trace1_fil = trace1_fil / np.max(trace1_fil)
    # trace2_fil = trace2_fil / np.max(trace2_fil)
    # trace3_fil = trace3_fil / np.max(trace3_fil)
    #
    # avg_trace = np.mean(traces[3500:4001, :], 0)
    # avg_trace = avg_trace / np.max(avg_trace)
    #
    # avg_fil1 = butter_bandpass_filter(avg_trace, 0.5, 1, fs, order=5)
    # avg_fil2 = butter_bandpass_filter(avg_trace, 0.2, 0.6, 10, order=5)
    # avg_fil3 = butter_bandpass_filter(avg_trace, 0.1, 0.3, 10, order=5)
    #
    # avg_fil1 = avg_fil1 / np.max(avg_fil1)
    # avg_fil2 = avg_fil2 / np.max(avg_fil2)
    # avg_fil3 = avg_fil3 / np.max(avg_fil3)
    #
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(t_ax, trace1)
    # plt.grid(True)
    # plt.ylabel('Strain [-]')
    # plt.xlabel('Tiempo [s]')
    # plt.title('Trazas DAS datos Belgica')
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
    # plt.title('Trazas DAS datos Belgica filtrados 0.5 - 1 Hz')
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
    # plt.plot(t_ax, avg_trace)
    # plt.grid(True)
    # plt.ylabel('Strain [-]')
    # plt.xlabel('Tiempo [s]')
    # plt.title('Traza DAS promedio datos Belgica')
    # plt.savefig('Imgs/Avgdata.png')
    #
    # plt.clf()
    # plt.plot(t_ax, avg_fil1)
    # plt.grid(True)
    # plt.ylabel('Strain [-]')
    # plt.xlabel('Tiempo [s]')
    # plt.title('Traza DAS promedio datos Belgica filtrada 0.5 - 1 Hz')
    # plt.savefig('Imgs/Avgdata_fil1.png')
    #
    # plt.clf()
    # plt.plot(t_ax, avg_fil2)
    # plt.grid(True)
    # plt.ylabel('Strain [-]')
    # plt.xlabel('Tiempo [s]')
    # plt.title('Traza DAS promedio datos Belgica filtrada 0.2 - 0.6 Hz')
    # plt.savefig('Imgs/Avgdata_fil2.png')
    #
    # plt.clf()
    # plt.plot(t_ax, avg_fil3)
    # plt.grid(True)
    # plt.ylabel('Strain [-]')
    # plt.xlabel('Tiempo [s]')
    # plt.title('Traza DAS promedio datos Belgica filtrada 0.1 - 0.3 Hz')
    # plt.savefig('Imgs/Avgdata_fil3.png')
    #
    # plt.clf()
    # line_avg, = plt.plot(t_ax, avg_trace, label='NO filtrada')
    # line_fil, = plt.plot(t_ax, avg_fil1, label='Filtrada')
    # plt.grid(True)
    # plt.xlabel('Tiempo [s]')
    # plt.ylabel('Strain [-]')
    # plt.legend(handles=[line_avg, line_fil], loc='upper right')
    # plt.title('Traza promedio datos Belgica filtrada 0.5 - 1 Hz')
    # plt.savefig('Imgs/AvgComp1.png')
    #
    # plt.clf()
    # line_avg, = plt.plot(t_ax, avg_trace, label='NO filtrada')
    # line_fil, = plt.plot(t_ax, avg_fil2, label='Filtrada')
    # plt.grid(True)
    # plt.xlabel('Tiempo [s]')
    # plt.ylabel('Strain [-]')
    # plt.legend(handles=[line_avg, line_fil], loc='upper right')
    # plt.title('Traza promedio datos Belgica filtrada 0.2 - 0.6 Hz')
    # plt.savefig('Imgs/AvgComp2.png')
    #
    # plt.clf()
    # line_avg, = plt.plot(t_ax, avg_trace, label='NO filtrada')
    # line_fil, = plt.plot(t_ax, avg_fil3, label='Filtrada')
    # plt.grid(True)
    # plt.xlabel('Tiempo [s]')
    # plt.ylabel('Strain [-]')
    # plt.legend(handles=[line_avg, line_fil], loc='upper right')
    # plt.title('Traza promedio datos Belgica filtrada 0.1 - 0.3 Hz')
    # plt.savefig('Imgs/AvgComp3.png')
    #
    # plt.clf()
    # line_st, = plt.plot(signal.resample(trace3, 6000), label='DAS')
    # line_das, = plt.plot(st_trace, label='STEAD')
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Belgica')
    # plt.legend(handles=[line_st, line_das], loc='upper left')
    # plt.savefig('Imgs/STEADBelgicaT3.png')
    #
    # plt.clf()
    # plt.subplot(211)
    # plt.plot(st_trace)
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Belgica')
    # plt.subplot(212)
    # plt.plot(signal.resample(trace3, 6000))
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.savefig('Imgs/STEADBelgicaT31.png')
    #
    # plt.clf()
    # line_st, = plt.plot(signal.resample(trace3_fil, 6000), label='DAS')
    # line_das, = plt.plot(st_trace, label='STEAD')
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Belgica filtrada')
    # plt.legend(handles=[line_st, line_das], loc='upper left')
    # plt.savefig('Imgs/STEADBelgicaT3fil.png')
    #
    # plt.clf()
    # plt.subplot(211)
    # plt.plot(st_trace)
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Belgica filtrada')
    # plt.subplot(212)
    # plt.plot(signal.resample(trace3_fil, 6000))
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.savefig('Imgs/STEADBelgicaT3fil1.png')
    #
    # plt.clf()
    # line_st, = plt.plot(signal.resample(avg_trace, 6000), label='DAS')
    # line_das, = plt.plot(st_trace, label='STEAD')
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Belgica promedio')
    # plt.legend(handles=[line_st, line_das], loc='upper left')
    # plt.savefig('Imgs/STEADBelgicaAVG.png')
    #
    # plt.clf()
    # plt.subplot(211)
    # plt.plot(st_trace)
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Belgica promedio')
    # plt.subplot(212)
    # plt.plot(signal.resample(avg_trace, 6000))
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.savefig('Imgs/STEADBelgicaAVG1.png')
    #
    # plt.clf()
    # line_st, = plt.plot(signal.resample(avg_fil1, 6000), label='DAS')
    # line_das, = plt.plot(st_trace, label='STEAD')
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Belgica promedio filtrada')
    # plt.legend(handles=[line_st, line_das], loc='upper left')
    # plt.savefig('Imgs/STEADBelgicaAVGfil.png')
    #
    # plt.clf()
    # plt.subplot(211)
    # plt.plot(st_trace)
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Belgica promedio filtrada')
    # plt.subplot(212)
    # plt.plot(signal.resample(avg_fil1, 6000))
    # plt.grid(True)
    # plt.xlabel('Muestras [s]')
    # plt.ylabel('Strain [-]')
    # plt.savefig('Imgs/STEADBelgicaAVGfil1.png')


# Filtro pasabanda butterworth
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
