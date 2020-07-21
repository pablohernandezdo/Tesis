import re
import h5py
import segyio
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

    # # Dataset Francia (Hay 606 trazas nulas)
    #
    # # Read file
    # Francia = sio.loadmat("../Data_Francia/Earthquake_1p9_Var_BP_2p5_15Hz.mat")
    #
    # # Load data
    # traces = Francia["StrainFilt"]
    #
    # # Add valid traces to dataset file
    # for trace in traces:
    #     if np.max(np.abs(trace)):
    #         seismic_dset = np.vstack((seismic_dset, trace))
    #
    # # Remove initial definition row
    # seismic_dset = seismic_dset[1:]


    # Dataset Nevada

    # File 721
    Nevada721 = '../Data_Nevada/PoroTomo_iDAS16043_160321073721.sgy'

    # Read file
    with segyio.open(Nevada721, ignore_geometry=True) as segy:
        #segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    for trace in traces:
        resamp_trace = signal.resample(trace, 3000)
        resamp_trace = np.pad(resamp_trace, (0, 3000), 'constant')
        seismic_dset = np.vstack((seismic_dset, resamp_trace))

    # File 751
    Nevada751 = '../Data_Nevada/PoroTomo_iDAS16043_160321073751.sgy'

    # Read file
    with segyio.open(Nevada751, ignore_geometry=True) as segy:
        #segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    for trace in traces:
        resamp_trace = signal.resample(trace, 3000)
        resamp_trace = np.pad(resamp_trace, (0, 3000), 'constant')
        seismic_dset = np.vstack((seismic_dset, resamp_trace))

    # File 747
    Nevada747 = '../Data_Nevada/PoroTomo_iDAS025_160321073747.sgy'

    # Read file
    with segyio.open(Nevada747, ignore_geometry=True) as segy:
        #segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    for trace in traces:
        resamp_trace = signal.resample(trace, 3000)
        resamp_trace = np.pad(resamp_trace, (0, 3000), 'constant')
        seismic_dset = np.vstack((seismic_dset, resamp_trace))

    # File 717
    Nevada717 = '../Data_Nevada/PoroTomo_iDAS025_160321073717.sgy'

    # Read file
    with segyio.open(Nevada717, ignore_geometry=True) as segy:
        #segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    for trace in traces:
        resamp_trace = signal.resample(trace, 3000)
        resamp_trace = np.pad(resamp_trace, (0, 3000), 'constant')
        seismic_dset = np.vstack((seismic_dset, resamp_trace))

    print(seismic_dset.shape)

    # # Dataset Belgica
    #
    # Belgica = sio.loadmat("../Data_Belgica/mat_2018_08_19_00h28m05s_Parkwind_HDAS_2Dmap_StrainData_2D.mat")
    #
    # # Read data
    # traces = f['Data_2D']
    # plt_tr = 4000

    # Sampling frequency

    # # Dataset Reykjanes
    #
    # # Telesismo Fibra optica
    #
    # file_fo = '../Data_Reykjanes/Jousset_et_al_2018_003_Figure3_fo.ascii'
    #
    # fs = 20
    #
    # data_fo = {
    #     'head': '',
    #     'strain': []
    # }
    #
    # with open(file_fo, 'r') as f:
    #     for idx, line in enumerate(f):
    #         if idx == 0:
    #             data_fo['head'] = line.strip()
    #         else:
    #             val = line.strip()
    #             data_fo['strain'].append(float(val))
    #
    # data_fo['strain'] = signal.resample(np.asarray(data_fo['strain']), 6000)


if __name__ == "__main__":
    main()