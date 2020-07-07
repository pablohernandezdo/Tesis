import scipy.io

import matplotlib.pyplot as plt

def main():

    # f = scipy.io.loadmat('../Data_California/FSE-11_1080SecP_SingDec_StepTest (1).mat')
    f = scipy.io.loadmat('../Data_California/FSE-06_480SecP_SingDec_StepTest (1).mat')

    data = f['singdecmatrix']

    # 196 trazas de 953432 muestras

    data = data.transpose()

    print(data[0].shape)
    print(data[:,0].shape)

    plt.figure()
    plt.plot(data[0])
    plt.savefig('data.png')

    plt.clf()
    plt.plot(data[:,0])
    plt.savefig('dataxx.png')


if __name__ == '__main__':
    main()