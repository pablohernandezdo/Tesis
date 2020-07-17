import h5py
import segyio
import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.fftpack as sfft
import scipy.signal as signal
from scipy.signal import butter, lfilter

from pathlib import Path


def main():
    # Create images and animations folder

    Path("Imgs").mkdir(exist_ok=True)
    Path("Imgs/721").mkdir(exist_ok=True)
    Path("Imgs/751").mkdir(exist_ok=True)
    Path("Imgs/747").mkdir(exist_ok=True)
    Path("Imgs/717").mkdir(exist_ok=True)

    Path("Animations").mkdir(exist_ok=True)
    Path("Animations/721").mkdir(exist_ok=True)
    Path("Animations/751").mkdir(exist_ok=True)
    Path("Animations/747").mkdir(exist_ok=True)
    Path("Animations/717").mkdir(exist_ok=True)

    # Carga traza STEAD

    # st = '../Data_STEAD/Train_data.hdf5'
    #
    # with h5py.File(st, 'r') as h5_file:
    #     grp = h5_file['earthquake']['local']
    #     for idx, dts in enumerate(grp):
    #         st_trace = grp[dts][:, 0] / np.max(np.abs(grp[dts][:, 0]))
    #         break

    # File 721
    f = '../Data_Nevada/PoroTomo_iDAS16043_160321073721.sgy'

    # Read file
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

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
        plt.title(f'Traza Nevada y espectro #{trtp_ids[idx]} archivo 721')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(212)
        plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/721/Nevada721_{trtp_ids[idx]}')

    # # Data animation
    # fig_tr = plt.figure()
    # ims_tr = []
    #
    # for trace in traces:
    #     im_tr = plt.plot(t_ax, trace)
    #     plt.title('Trazas dataset Nevada archivo 721')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Tiempo [s]')
    #     plt.grid(True)
    #     ims_tr.append(im_tr)
    #
    # ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=50, blit=True, repeat=False)
    # ani_tr.save('Animations/721/Traces.mp4')
    #
    # # Spectrum animation
    # fig_sp = plt.figure()
    # ims_sp = []
    #
    # for trace in traces:
    #     yf = sfft.fftshift(sfft.fft(trace))
    #     im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.title('Espectro trazas dataset Nevada archivo 721')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.grid(True)
    #     ims_sp.append(im_sp)
    #
    # ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=50, blit=True, repeat=False)
    # ani_sp.save('Animations/721/Spectrums.mp4')

    # File 751
    f = '../Data_Nevada/PoroTomo_iDAS16043_160321073751.sgy'

    # Read file
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    # Number of traces to plot
    n = 4

    # Traces to plot
    trtp = []

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

    # For trace in traces to print
    for idx, trace in enumerate(trtp):
        yf = sfft.fftshift(sfft.fft(trace))

        plt.clf()
        plt.subplot(211)
        plt.plot(t_ax, trace)
        plt.title(f'Traza Nevada y espectro #{trtp_ids[idx]} archivo 751')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(212)
        plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/751/Nevada751_{trtp_ids[idx]}')

    # # Data animation
    # fig_tr = plt.figure()
    # ims_tr = []
    #
    # for trace in traces:
    #     im_tr = plt.plot(t_ax, trace)
    #     plt.title('Trazas dataset Nevada archivo 751')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Tiempo [s]')
    #     plt.grid(True)
    #     ims_tr.append(im_tr)
    #
    # ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=50, blit=True, repeat=False)
    # ani_tr.save('Animations/751/Traces.mp4')
    #
    # # Spectrum animation
    # fig_sp = plt.figure()
    # ims_sp = []
    #
    # for trace in traces:
    #     yf = sfft.fftshift(sfft.fft(trace))
    #     im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.title('Espectro trazas dataset Nevada archivo 751')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.grid(True)
    #     ims_sp.append(im_sp)
    #
    # ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=50, blit=True, repeat=False)
    # ani_sp.save('Animations/751/Spectrums.mp4')

    # File 747
    f = '../Data_Nevada/PoroTomo_iDAS025_160321073747.sgy'

    # Read file
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    # Number of traces to plot
    n = 4

    # Traces to plot
    trtp = []

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

    # For trace in traces to print
    for idx, trace in enumerate(trtp):
        yf = sfft.fftshift(sfft.fft(trace))

        plt.clf()
        plt.subplot(211)
        plt.plot(t_ax, trace)
        plt.title(f'Traza Nevada y espectro #{trtp_ids[idx]} archivo 747')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(212)
        plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/747/Nevada747_{trtp_ids[idx]}')

    # # Data animation
    # fig_tr = plt.figure()
    # ims_tr = []
    #
    # for trace in traces:
    #     im_tr = plt.plot(t_ax, trace)
    #     plt.title('Trazas dataset Nevada archivo 747')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Tiempo [s]')
    #     plt.grid(True)
    #     ims_tr.append(im_tr)
    #
    # ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=50, blit=True, repeat=False)
    # ani_tr.save('Animations/747/Traces.mp4')
    #
    # # Spectrum animation
    # fig_sp = plt.figure()
    # ims_sp = []
    #
    # for trace in traces:
    #     yf = sfft.fftshift(sfft.fft(trace))
    #     im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.title('Espectro trazas dataset Nevada archivo 747')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.grid(True)
    #     ims_sp.append(im_sp)
    #
    # ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=50, blit=True, repeat=False)
    # ani_sp.save('Animations/747/Spectrums.mp4')

    # File 717
    f = '../Data_Nevada/PoroTomo_iDAS025_160321073717.sgy'

    # Read file
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    # Number of traces to plot
    n = 4

    # Traces to plot
    trtp = []

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

    # For trace in traces to print
    for idx, trace in enumerate(trtp):
        yf = sfft.fftshift(sfft.fft(trace))

        plt.clf()
        plt.subplot(211)
        plt.plot(t_ax, trace)
        plt.title(f'Traza Nevada y espectro #{trtp_ids[idx]} archivo 717')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(212)
        plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/717/Nevada717_{trtp_ids[idx]}')

    # # Data animation
    # fig_tr = plt.figure()
    # ims_tr = []
    #
    # for trace in traces:
    #     im_tr = plt.plot(t_ax, trace)
    #     plt.title('Trazas dataset Nevada archivo 717')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Tiempo [s]')
    #     plt.grid(True)
    #     ims_tr.append(im_tr)
    #
    # ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=50, blit=True, repeat=False)
    # ani_tr.save('Animations/717/Traces.mp4')
    #
    # # Spectrum animation
    # fig_sp = plt.figure()
    # ims_sp = []
    #
    # for trace in traces:
    #     yf = sfft.fftshift(sfft.fft(trace))
    #     im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.title('Espectro trazas dataset Nevada archivo 717')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.grid(True)
    #     ims_sp.append(im_sp)
    #
    # ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=50, blit=True, repeat=False)
    # ani_sp.save('Animations/717/Spectrums.mp4')

    # t_ax = np.arange(1, len(traces[0]) + 1) / fs
    #
    # trace1 = traces[0]
    # trace2 = traces[100]
    # trace3 = traces[200]
    #
    # trace1_resamp = signal.resample(traces[0], 6000)
    # trace2_resamp = signal.resample(traces[100], 6000)
    # trace3_resamp = signal.resample(traces[200], 6000)
    #
    # trace1_fil = butter_bandpass_filter(trace1, 0.1, 10, fs, order=3)
    # trace2_fil = butter_bandpass_filter(trace2, 0.1, 10, fs, order=3)
    # trace3_fil = butter_bandpass_filter(trace3, 0.1, 10, fs, order=3)
    #
    # trace1_resamp = trace1 / np.max(np.abs(trace1_resamp))
    # trace2_resamp = trace2 / np.max(np.abs(trace2_resamp))
    # trace3_resamp = trace3 / np.max(np.abs(trace3_resamp))
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
    # plt.title('Trazas DAS datos Nevada')
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
    # plt.title('Trazas DAS datos Nevada filtrados 1 - 10 Hz')
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
    # line_st, = plt.plot(trace1_resamp, label='DAS')
    # line_das, = plt.plot(st_trace, label='STEAD')
    # plt.grid(True)
    # plt.xlabel('Muestras [-]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Nevada')
    # plt.legend(handles=[line_st, line_das], loc='upper left')
    # plt.savefig('Imgs/STEADNevada.png')
    #
    # plt.clf()
    # plt.subplot(211)
    # plt.plot(st_trace)
    # plt.grid(True)
    # plt.xlabel('Muestras [-]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza STEAD y traza DAS Nevada')
    # plt.subplot(212)
    # plt.plot(trace1_resamp)
    # plt.grid(True)
    # plt.xlabel('Muestras [-]')
    # plt.ylabel('Strain [-]')
    # plt.savefig('Imgs/STEADNevada1.png')


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
