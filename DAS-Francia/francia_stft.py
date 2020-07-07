import numpy as np
import scipy.io

import matplotlib.pyplot as plt

from scipy import signal


def main():
    # Registro de 1 minuto de sismo M1.9 a 100 Km NE del cable

    f = scipy.io.loadmat("Earthquake_1p9_Var_BP_2p5_15Hz.mat")

    data = f["StrainFilt"]
    # time= f["Time"]
    # distance = f["Distance_fiber"]
    plt_tr = 3000
    fs = 100
    n_trace = 100

    # t_ax = np.arange(len(data[plt_tr])) / fs

    trace = data[n_trace]

    f, t, Zxx = signal.stft(trace, fs, nperseg=1000)

    plt.figure()
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(trace))
    plt.title('ASD')
    plt.ylabel('ASD')
    plt.xlabel('ASD')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
