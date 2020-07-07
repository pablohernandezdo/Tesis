import re
import segyio
import numpy as np

import matplotlib.pyplot as plt

from scipy import signal

def main():
    # f = 'PoroTomo_iDAS16043_160321073751.sgy'
    # f = 'PoroTomo_iDAS16043_160321073721.sgy'
    # f = 'PoroTomo_iDAS025_160321073747.sgy'
    f = 'PoroTomo_iDAS025_160321073717.sgy'

    n_trace = 10

    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    # time = np.arange(1, len(traces[0]) + 1) / fs

    trace = traces[n_trace] / np.max(traces[n_trace])

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