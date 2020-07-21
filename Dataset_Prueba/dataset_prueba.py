import re
import h5py
import numpy as np

import scipy.io as sio
from scipy import signal

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from scipy.signal import butter, lfilter


def main():

    # Dataset sismico, no sismico

    seismic_dset = np.empty((1, 6000))
    nseismic_dset = np.empty((1, 6000))

    # Dataset Francia

    # Read file
    Francia = sio.loadmat("../Data_Francia/Earthquake_1p9_Var_BP_2p5_15Hz.mat")

    # Load data
    traces = Francia["StrainFilt"]

    for trace in traces:
        if np.max(np.abs(trace)):
            seismic_dset = np.vstack((seismic_dset, trace))

    seismic_dset = seismic_dset[1:]

    print(f'seismic dset shape: {seismic_dset.shape}')


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

    data_fo['strain'] = signal.resample(np.asarray(data_fo['strain']), 6000)

    # Dataset sísmico


if __name__ == "__main__":
    main()