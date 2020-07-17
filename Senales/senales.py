import pywt
import numpy as np
import numpy.random as random

from scipy import signal
import matplotlib.pyplot as plt

from pathlib import Path


def main():
    # Create images folder

    Path("Imgs").mkdir(exist_ok=True)
    Path("Imgs/Noise").mkdir(exist_ok=True)
    Path("Imgs/Sin1").mkdir(exist_ok=True)
    Path("Imgs/Sin2").mkdir(exist_ok=True)
    Path("Imgs/Sin3").mkdir(exist_ok=True)
    Path("Imgs/Sin1_ns").mkdir(exist_ok=True)
    Path("Imgs/Sin2_ns").mkdir(exist_ok=True)
    Path("Imgs/Sin3_ns").mkdir(exist_ok=True)
    Path("Imgs/Sin_pad").mkdir(exist_ok=True)
    Path("Imgs/Wavelets").mkdir(exist_ok=True)

    # Noise

    ns = random.random_sample((1, 6000))

    # Sine waves

    # Number of sample points
    N = 6000

    # sampling frequency
    fs = 100

    # Time axis
    t = np.linspace(0.0, N / fs, N)

    # Number of frequency interval steps
    n = 100

    # Frequency spans
    fr1 = np.linspace(1, 100, n)
    fr2 = np.linspace(0, 1, n)

    # Prealocate
    wvs1 = []
    wvs2 = []
    wvs3 = []

    for f1, f2 in zip(fr1, fr2):
        print(f1, f2)
        sig1 = np.sin(f1 * 2.0 * np.pi * t)
        sig2 = np.sin(f2 * 2.0 * np.pi * t)
        wvs1.append(sig1)
        wvs2.append(sig2)
        wvs3.append(sig1 + sig2)

    wvs1 = np.array(wvs1)
    wvs2 = np.array(wvs2)
    wvs3 = np.array(wvs3)

    wvs1_ns = wvs1 + 0.5 * random.random_sample(wvs1.shape)
    wvs2_ns = wvs2 + 0.5 * random.random_sample(wvs1.shape)
    wvs3_ns = wvs3 + 0.5 * random.random_sample(wvs1.shape)

    # PADDED SINES

    # Number of intermediate sample points
    ni = [1000, 2000, 4000, 5000]

    # Number of points to zero-pad
    pad = [(N-n) // 2 for n in ni]

    # Time axis for smaller waves
    lts = [np.linspace(0.0, nis / fs, nis) for nis in ni]

    # All frequencies list
    all_fr = []

    # Calculate max period for smaller waves
    max_periods = [n_points / fs for n_points in ni]

    # Calculate frequencies for smaller waves
    for per in max_periods:
        freqs = []
        for i in range(1, int(per)+1):
            if per % i == 0:
                freqs.append(1/i)
        all_fr.append(freqs)

    # Preallocate waves
    wvs = []

    # Generate waves and zero padd
    for idx, fr_ls in enumerate(all_fr):
        for fr in fr_ls:
            wv = np.sin(fr * 2.0 * np.pi * lts[idx])
            wv = np.pad(wv, (pad[idx], pad[idx]), 'constant')
            wvs.append(wv)

    # Wavelets

    # Preallocate wavelets
    lets = []

    # Discrete wavelet families
    discrete_families = ['db', 'sym', 'coif', 'bior', 'rbio']

    # Obtain wavelet waveforms, resample and append
    for fam in discrete_families:
        for wavelet in pywt.wavelist(fam):
            wv = pywt.Wavelet(wavelet)
            if wv.orthogonal:
                [_, psi, _] = pywt.Wavelet(wavelet).wavefun(level=5)
                psi = signal.resample(psi, 6000)
                lets.append(psi)

    # Plot Sine waveforms

    # Number of traces to plot
    n_trtp = 4

    # Traces to plot
    trtp_sin1 = []
    trtp_sin2 = []
    trtp_sin3 = []
    trtp_sin1_ns = []
    trtp_sin2_ns = []
    trtp_sin3_ns = []

    # Traces to plot numbers
    trtp_ids_sin1 = random.randint(0, high=len(wvs1), size=n_trtp)
    trtp_ids_sin2 = random.randint(0, high=len(wvs2), size=n_trtp)
    trtp_ids_sin3 = random.randint(0, high=len(wvs3), size=n_trtp)
    trtp_ids_sin1_ns = random.randint(0, high=len(wvs1_ns), size=n_trtp)
    trtp_ids_sin2_ns = random.randint(0, high=len(wvs1_ns), size=n_trtp)
    trtp_ids_sin3_ns = random.randint(0, high=len(wvs1_ns), size=n_trtp)

    # Retrieve selected traces
    for idx, trace in enumerate(wvs1):
        if idx in trtp_ids_sin1:
            trtp_sin1.append(trace)

    for idx, trace in enumerate(wvs2):
        if idx in trtp_ids_sin2:
            trtp_sin2.append(trace)

    for idx, trace in enumerate(wvs3):
        if idx in trtp_ids_sin3:
            trtp_sin3.append(trace)

    for idx, trace in enumerate(wvs1_ns):
        if idx in trtp_ids_sin1_ns:
            trtp_sin1_ns.append(trace)

    for idx, trace in enumerate(wvs2_ns):
        if idx in trtp_ids_sin2_ns:
            trtp_sin2_ns.append(trace)

    for idx, trace in enumerate(wvs3_ns):
        if idx in trtp_ids_sin3_ns:
            trtp_sin3_ns.append(trace)

    # Figure to plot
    plt.figure()

    # Plot n random Sin 1 traces
    for idx, trace in enumerate(trtp_sin1):
        plt.clf()
        plt.plot(t, trace)
        plt.title(f'Traza sinusoides 1 #{trtp_ids_sin1[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.savefig(f'Imgs/Sin1/Sin1_{trtp_ids_sin1[idx]}')

    # Plot n random Sin 2 traces
    for idx, trace in enumerate(trtp_sin2):
        plt.clf()
        plt.plot(t, trace)
        plt.title(f'Traza sinusoides 2 #{trtp_ids_sin2[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.savefig(f'Imgs/Sin2/Sin2_{trtp_ids_sin2[idx]}')

    # Plot n random Sin 3 traces
    for idx, trace in enumerate(trtp_sin3):
        plt.clf()
        plt.plot(t, trace)
        plt.title(f'Traza sinusoides 3 #{trtp_ids_sin3[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.savefig(f'Imgs/Sin3/Sin3_{trtp_ids_sin3[idx]}')

    # Plot n random Sin 1 + noise traces
    for idx, trace in enumerate(trtp_sin1_ns):
        plt.clf()
        plt.plot(t, trace)
        plt.title(f'Traza sinusoides 1 + noise #{trtp_ids_sin1_ns[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.savefig(f'Imgs/Sin1_ns/Sin1_ns_{trtp_ids_sin1_ns[idx]}')

    # Plot n random Sin 2 + noise traces
    for idx, trace in enumerate(trtp_sin2_ns):
        plt.clf()
        plt.plot(t, trace)
        plt.title(f'Traza sinusoides 2 #{trtp_ids_sin2_ns[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.savefig(f'Imgs/Sin2_ns/Sin2_ns_{trtp_ids_sin2_ns[idx]}')

    # Plot n random Sin 3 + noise traces
    for idx, trace in enumerate(trtp_sin3_ns):
        plt.clf()
        plt.plot(t, trace)
        plt.title(f'Traza sinusoides 3 + noise #{trtp_ids_sin3_ns[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.savefig(f'Imgs/Sin3_ns/Sin3_ns_{trtp_ids_sin3_ns[idx]}')

    # Plot Padded Sine waveforms

    # Traces to plot
    trtp_padd = []

    # Traces to plot numbers
    trtp_ids_padd = random.randint(0, high=len(wvs), size=n_trtp)

    # Retrieve selected traces
    for idx, trace in enumerate(wvs):
        if idx in trtp_ids_padd:
            trtp_padd.append(trace)

    # Plot n random Sin 1 traces
    for idx, trace in enumerate(trtp_padd):
        plt.clf()
        plt.plot(t, trace)
        plt.title(f'Traza sinusoides padd #{trtp_ids_padd[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.savefig(f'Imgs/Sin_pad/Sin_pad_{trtp_ids_padd[idx]}')

    # Plot wavelet waveforms

    # Traces to plot
    trtp_wvlets = []

    # Traces to plot numbers
    trtp_ids_wvlets = random.randint(0, high=len(lets), size=n_trtp)

    # Retrieve selected traces
    for idx, trace in enumerate(lets):
        if idx in trtp_ids_wvlets:
            trtp_wvlets.append(trace)

    # Plot n random wavelet
    for idx, trace in enumerate(trtp_wvlets):
        plt.clf()
        plt.plot(t, trace)
        plt.title(f'Wavelet #{trtp_ids_padd[idx]}')
        plt.xlabel('Muestras [-]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.savefig(f'Imgs/Wavelets/wavelet_{trtp_ids_wvlets[idx]}')

    # Plot noise

    plt.clf()
    plt.plot(ns)
    plt.xlabel('Muestras [-]')
    plt.ylabel('Amplitud [-]')
    plt.title('Ruido blanco')
    plt.grid(True)
    plt.savefig(f'Imgs/Noise/noise.png')


if __name__ == "__main__":
    main()
