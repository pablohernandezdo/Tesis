import pywt
import numpy as np
import scipy.io

import matplotlib.pyplot as plt

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
    cA, cD = pywt.dwt(trace, 'db2')
    print(cA, cD)


if __name__ == "__main__":
    main()
