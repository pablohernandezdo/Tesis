import re
import h5py
import numpy as np

import matplotlib.pyplot as plt

import scipy.signal as signal
from scipy.signal import butter, lfilter


# RECREAR FIGURAS DEL PAPER DE JOUSSET

def main():
    # Carga traza STEAD

    st = '../Data_STEAD/Train_data.hdf5'

    with h5py.File(st, 'r') as h5_file:
        grp = h5_file['earthquake']['local']
        for idx, dts in enumerate(grp):
            st_trace = grp[dts][:, 0] / np.max(grp[dts][:, 0])
            break


    ########## DATOS UTILES ##########

    # Fig. 3fo and 3bb.
    # Comparacion entre registros de un telesismo por fibra optica y sismometro

    file_fo = 'Jousset_et_al_2018_003_Figure3_fo.ascii'
    file_bb = 'Jousset_et_al_2018_003_Figure3_bb.ascii'

    fs = 20

    data_fo = {
        'head': '',
        'strain': []
    }

    data_bb = {
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

    with open(file_bb, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data_bb['head'] = line.strip()
            else:
                val = line.strip()
                data_bb['strain'].append(float(val))

    t_ax = np.arange(len(data_fo['strain'])) / fs

    plt.figure()
    plt.plot(t_ax, data_fo['strain'])
    plt.grid(True)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Strain [-]')
    plt.title('Registro telesismo DAS')
    plt.savefig('Imgs/TelesismoDAS.png')

    plt.clf()
    plt.plot(t_ax, data_bb['strain'])
    plt.grid(True)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Strain [-]')
    plt.title('Registro telesismo sismómetro')
    plt.savefig('Imgs/TelesismoBBS.png')

    plt.clf()
    line_fo, = plt.plot(t_ax, data_fo['strain'], label='DAS')
    line_bb, = plt.plot(t_ax, data_bb['strain'], label='Sismómetro')
    plt.grid(True)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Strain [-]')
    plt.title('Registros telesismo DAS y sismómetro')
    plt.legend(handles=[line_fo, line_bb], loc='upper left')
    plt.savefig('Imgs/TelesismoComp.png')

    plt.clf()
    line_st, = plt.plot(signal.resample(data_fo['strain'], 6000), label='DAS')
    line_das, = plt.plot(st_trace, label='STEAD')
    plt.grid(True)
    plt.xlabel('Muestras [s]')
    plt.ylabel('Strain [-]')
    plt.title('Traza STEAD y traza DAS Reykjanes telesismo')
    plt.legend(handles=[line_st, line_das], loc='upper left')
    plt.savefig('Imgs/STEADTelesismo.png')

    plt.clf()
    plt.subplot(211)
    plt.plot(st_trace)
    plt.grid(True)
    plt.xlabel('Muestras [s]')
    plt.ylabel('Strain [-]')
    plt.title('Traza STEAD y traza DAS Reykjanes telesismo')
    plt.subplot(212)
    plt.plot(signal.resample(data_fo['strain'], 6000))
    plt.grid(True)
    plt.xlabel('Muestras [s]')
    plt.ylabel('Strain [-]')
    plt.savefig('Imgs/STEADTelesismo1.png')


    # Fig. 5a_fo
    # Registro de sismo local con DAS

    file = 'Jousset_et_al_2018_003_Figure5a_fo.ascii'
    n_trazas = 26
    plt_tr = 10
    fs = 200

    data = {
        'head': '',
        'strain': np.empty((1, n_trazas))
    }

    with open(file, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data['head'] = line.strip()

            else:
                row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
                data['strain'] = np.concatenate((data['strain'], np.expand_dims(row, 0)))

    data['strain'] = data['strain'][1:]
    data['strain'] = data['strain'] / data['strain'].max(axis=0)
    data['strain'] = data['strain'].transpose()
    data_das = data

    t_ax = np.arange(len(data['strain'][plt_tr])) / fs

    plt.clf()
    plt.plot(t_ax, data['strain'][plt_tr])
    plt.grid(True)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Strain [-]')
    plt.title('Registro sismo local DAS')
    plt.savefig('Imgs/SismolocalDAS.png')

    plt.clf()
    line_st, = plt.plot(st_trace, label='STEAD')
    line_das, = plt.plot(signal.resample(data['strain'][plt_tr], 6000), label='DAS')
    plt.grid(True)
    plt.xlabel('Muestras [s]')
    plt.ylabel('Strain [-]')
    plt.title('Traza STEAD y traza DAS Reykjanes sismo local')
    plt.legend(handles=[line_st, line_das], loc='upper left')
    plt.savefig('Imgs/STEADLocal.png')

    plt.clf()
    plt.subplot(211)
    plt.plot(st_trace)
    plt.grid(True)
    plt.xlabel('Muestras [s]')
    plt.ylabel('Strain [-]')
    plt.title('Traza STEAD y traza DAS Reykjanes sismo local')
    plt.subplot(212)
    plt.plot(signal.resample(data['strain'][plt_tr], 6000))
    plt.grid(True)
    plt.xlabel('Muestras [s]')
    plt.ylabel('Strain [-]')
    plt.savefig('Imgs/STEADLocal1.png')

    # # Fig. 5a_gph
    # # Registro de sismo local con geofono

    file = 'Jousset_et_al_2018_003_Figure5a_gph.ascii'
    n_trazas = 26
    plt_tr = 10
    fs = 200

    data = {
        'head': '',
        'strain': np.empty((1, n_trazas))
    }

    with open(file, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data['head'] = line.strip()

            else:
                row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
                data['strain'] = np.concatenate((data['strain'], np.expand_dims(row, 0)))

    data['strain'] = data['strain'][1:]
    data['strain'] = data['strain'] / data['strain'].max(axis=0)
    data['strain'] = data['strain'].transpose()

    t_ax = np.arange(len(data['strain'][plt_tr])) / fs

    plt.clf()
    plt.plot(t_ax, data['strain'][plt_tr])
    plt.grid(True)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Strain [-]')
    plt.title('Registro sismo local geófono')
    plt.savefig('Imgs/SismolocalGEO.png')

    plt.clf()
    plt.subplot(311)
    line_das, = plt.plot(t_ax, data_das['strain'][plt_tr], label='DAS')
    line_geo, = plt.plot(t_ax, data['strain'][plt_tr], label='Geófono')
    plt.grid(True)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Strain [-]')
    plt.legend(handles=[line_das, line_geo], loc='upper right')
    plt.title('Registros sismo local geofono y DAS')

    plt.subplot(312)
    line_das, = plt.plot(t_ax, data_das['strain'][15], label='DAS')
    line_geo, = plt.plot(t_ax, data['strain'][15], label='Geófono')
    plt.grid(True)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Strain [-]')
    plt.legend(handles=[line_das, line_geo], loc='upper right')

    plt.subplot(313)
    line_das, = plt.plot(t_ax, data_das['strain'][20], label='DAS')
    line_geo, = plt.plot(t_ax, data['strain'][20], label='Geófono')
    plt.grid(True)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Strain [-]')
    plt.legend(handles=[line_das, line_geo], loc='upper right')
    plt.savefig('Imgs/LocalComp.png')

    # Fig. 5b
    # Registro de sismo local con DAS

    file = 'Jousset_et_al_2018_003_Figure5b.ascii'
    n_trazas = 2551
    plt_tr = 1000
    fs = 200

    data = {
        'head': '',
        'strain': np.empty((1, n_trazas))
    }

    with open(file, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data['head'] = line.strip()

            else:
                row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
                data['strain'] = np.concatenate((data['strain'], np.expand_dims(row, 0)))

    data['strain'] = data['strain'][1:]
    data['strain'] = data['strain'] / data['strain'].max(axis=0)
    data['strain'] = data['strain'].transpose()

    t_ax = np.arange(len(data['strain'][plt_tr])) / fs

    plt.clf()
    plt.subplot(311)
    plt.plot(t_ax, data['strain'][plt_tr])
    plt.grid(True)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Strain [-]')
    plt.title('Trazas sismo local solo DAS')

    plt.subplot(312)
    plt.plot(t_ax, data['strain'][1500])
    plt.grid(True)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Strain [-]')

    plt.subplot(313)
    plt.plot(t_ax, data['strain'][2000])
    plt.grid(True)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Strain [-]')
    plt.savefig('Imgs/SismolocalDAS5b.png')

    plt.clf()
    line_st, = plt.plot(st_trace, label='STEAD')
    line_das, = plt.plot(signal.resample(data['strain'][plt_tr], 6000), label='DAS')
    plt.grid(True)
    plt.xlabel('Muestras [s]')
    plt.ylabel('Strain [-]')
    plt.title('Traza STEAD y traza DAS Reykjanes sismo local solo DAS')
    plt.legend(handles=[line_st, line_das], loc='upper left')
    plt.savefig('Imgs/STEADLocalDAS.png')

    plt.clf()
    plt.subplot(211)
    plt.plot(st_trace)
    plt.grid(True)
    plt.xlabel('Muestras [s]')
    plt.ylabel('Strain [-]')
    plt.title('Traza STEAD y traza DAS Reykjanes sismo local solo DAS')
    plt.subplot(212)
    plt.plot(signal.resample(data['strain'][plt_tr], 6000))
    plt.grid(True)
    plt.xlabel('Muestras [s]')
    plt.ylabel('Strain [-]')
    plt.savefig('Imgs/STEADLocalDAS1.png')

    ########## DEMAS FIGURAS ##########

    # Fig.2a

    # file = 'Jousset_et_al_2018_003_Figure2a.ascii'
    # n_trazas = 101
    # fs = 200
    #
    # data = {
    #     'head': '',
    #     'strain': np.empty((1, n_trazas))
    # }

    # Read all traces

    # with open(file, 'r') as f:
    #     for idx, line in enumerate(f):
    #         if idx == 0:
    #             data['head'] = line.strip()
    #
    #         else:
    #             row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
    #             data['strain'] = np.concatenate((data['strain'], np.expand_dims(row, 0)))
    #             print(f'data strain shape: {data["strain"].shape}')
    #
    # data['strain'] = data['strain'][1:]
    # data['strain'] = data['strain'].transpose()

    # Read specific trace

    # n = 90
    # tr = []
    #
    # with open(file, 'r') as f:
    #     for idx, line in enumerate(f):
    #         if idx:
    #             tr.append(float(re.sub(' +', ' ', line).strip().split(' ')[n]))
    #
    # tr_full = (np.asarray(tr).transpose()) / np.max(tr)
    # t_ax = np.arange(len(tr_full)) / fs
    #
    # plt.figure()
    # plt.plot(t_ax, tr_full)
    # plt.grid(True)
    # plt.xlabel('Tiempo [s]')
    # plt.ylabel('Strain [-]')
    # plt.title('Traza DAS fig2a #' + str(n))
    # plt.show()

    # # Fig. 2b
    #
    # file = 'Jousset_et_al_2018_003_Figure2b.ascii'
    # n_trazas = 2401
    #
    # data = {
    #     'head': '',
    #     'strain': np.empty((1, n_trazas))
    # }
    #
    # first = 1
    #
    # with open(file, 'r') as f:
    #     for line in f:
    #         if first:
    #             data['head'] = line.strip()
    #             first = 0
    #
    #         else:
    #             row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
    #             data['strain'] = np.concatenate((data['strain'], np.expand_dims(row, 0)))
    #             print(f'data strain shape: {data["strain"].shape}')
    #
    # data['strain'] = data['strain'][1:]
    #
    # plt.clf()
    # plt.imshow(data['strain'], aspect='auto')
    # plt.savefig('Fig2b.png')

    # # Fig. 7-8
    #
    # file = 'Jousset_et_al_2018_003_Figure7_8.ascii'
    # l_traza = 76600
    #
    # data = {
    #     'head': '',
    #     'strain': np.empty((1, l_traza))
    # }
    #
    # first = 1
    #
    # with open(file, 'r') as f:
    #     for line in f:
    #         if first:
    #             data['head'] = line.strip()
    #             first = 0
    #
    #         else:
    #             row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
    #             data['strain'] = np.concatenate((data['strain'], np.expand_dims(row, 0)))
    #             print(f'data strain shape: {data["strain"].shape}')
    #
    # data['strain'] = data['strain'][1:]
    #
    # plt.clf()
    # plt.imshow(data['strain'], aspect='auto')
    # plt.savefig('Fig7_8.png')


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


if __name__ == "__main__":
    main()
