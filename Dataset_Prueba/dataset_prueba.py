import re
import h5py
import samplerate
import numpy as np

import scipy.signal as signal

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from scipy.signal import butter, lfilter


def main():
    # Dataset Reykjanes

    # Telesismo Fibra optica

    file_fo = '../Data_Reykjanes/Jousset_et_al_2018_003_Figure3_fo.ascii'

    fs = 20

    data_fo = {
        'head': '',
        'strain': []
    }

    with open(file_fo, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data_fo['head'] = line.strip()
            else:
                val = line.strip()
                data_fo['strain'].append(float(val))

    data_fo['strain'] = np.asarray(data_fo['strain'])
    print(data_fo['strain'].shape)

    # Resample

    ratio = 5
    converter = 'sinc_best'
    data_fo['strain'] = samplerate.resample(data_fo['strain'], ratio, converter)

    print(data_fo['strain'].shape)


if __name__ == "__main__":
    main()