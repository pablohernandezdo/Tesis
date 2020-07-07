import numpy as np

import matplotlib.pyplot as plt

from numpy import random

def main():

    # Noise
    ns = random.random_sample((6000, 1))

    # Sine

    # Number of sample points
    N = 6000

    # sampling frequency
    fs = 100

    # sampling spacing
    T = 1.0 / fs

    t = np.linspace(0.0, N / fs, N)

    n = 100

    # Frequency spans
    fr1 = np.linspace(1, 100, n)
    fr2 = np.linspace(0.1, 10, n)

    # Prealocate
    wvs1 = []
    wvs2 = []
    wvs3 = []

    for i in np.arange(n):
        sig1 = np.sin(fr1[i] * 2.0 * np.pi * t)
        sig2 = np.sin(fr2[i] * 2.0 * np.pi * t)
        wvs1.append(sig1)
        wvs2.append(sig2)
        wvs3.append(sig1 + sig2)

    wvs1 = np.array(wvs1)
    wvs2 = np.array(wvs2)
    wvs3 = np.array(wvs3)

    wvs1_ns = wvs1 + random.random_sample(wvs1.shape)
    wvs2_ns = wvs2 + random.random_sample(wvs1.shape)
    wvs3_ns = wvs3 + random.random_sample(wvs1.shape)


    print(ns.shape)
    print(sig1.shape)
    print(f't len: {len(t)}')
    print(f'fr1 shape: {len(fr1)}')
    print(f'fr2 shape: {len(fr2)}')
    print(f'wvs1 shape: {wvs1.shape}')
    print(f'wvs1 el shape: {wvs1[0].shape}')
    print(f'arange shape: {len(np.arange(100))}')

    plt.figure()
    plt.subplot(311)
    plt.plot(wvs1[0])
    plt.subplot(312)
    plt.plot(wvs2[0])
    plt.subplot(313)
    plt.plot(wvs3[0])

    plt.figure()
    plt.subplot(311)
    plt.plot(wvs1_ns[0])
    plt.subplot(312)
    plt.plot(wvs2_ns[0])
    plt.subplot(313)
    plt.plot(wvs3_ns[0])
    plt.show()

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(ns)
    # plt.subplot(212)
    # plt.plot(wv)
    # plt.show()

if __name__ == "__main__":
    main()
