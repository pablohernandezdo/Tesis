import re
import pywt
import numpy as np

def main():
    # Fig. 3fo and 3bb.
    # COmparacion entre registros de un sismo por fibra optica y sismometro

    # file_fo = 'Jousset_et_al_2018_003_Figure3_fo.ascii'
    # file_bb = 'Jousset_et_al_2018_003_Figure3_bb.ascii'
    #
    # fs = 20
    #
    # data_fo = {
    #     'head': '',
    #     'strain': []
    # }
    #
    # data_bb = {
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
    # with open(file_bb, 'r') as f:
    #     for idx, line in enumerate(f):
    #         if idx == 0:
    #             data_bb['head'] = line.strip()
    #         else:
    #             val = line.strip()
    #             data_bb['strain'].append(float(val))
    #
    # # t_ax = np.arange(len(data_fo['strain'])) / fs
    #
    # cA, cD = pywt.dwt(data_fo['strain'], 'db2')

    # Fig. 5a_fo

    # file = 'Jousset_et_al_2018_003_Figure5a_fo.ascii'
    # n_trazas = 26
    # plt_tr = 20
    # fs = 200
    #
    # data = {
    #     'head': '',
    #     'strain': np.empty((1, n_trazas))
    # }
    #
    # with open(file, 'r') as f:
    #     for idx, line in enumerate(f):
    #         if idx == 0:
    #             data['head'] = line.strip()
    #
    #         else:
    #             row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
    #             data['strain'] = np.concatenate((data['strain'], np.expand_dims(row, 0)))
    #
    # data['strain'] = data['strain'][1:]
    # data['strain'] = data['strain'] / data['strain'].max(axis=0)
    # data['strain'] = data['strain'].transpose()
    #
    # # t_ax = np.arange(len(data['strain'][plt_tr])) / fs
    #
    # trace = data['strain'][plt_tr]

    # cA, cD = pywt.dwt(trace, 'db2')

    # Fig. 5a_gph

    # file = 'Jousset_et_al_2018_003_Figure5a_gph.ascii'
    # n_trazas = 26
    # plt_tr = 10
    # fs = 200
    #
    # data = {
    #     'head': '',
    #     'strain': np.empty((1, n_trazas))
    # }
    #
    # with open(file, 'r') as f:
    #     for idx, line in enumerate(f):
    #         if idx == 0:
    #             data['head'] = line.strip()
    #
    #         else:
    #             row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
    #             data['strain'] = np.concatenate((data['strain'], np.expand_dims(row, 0)))
    #
    # data['strain'] = data['strain'][1:]
    # data['strain'] = data['strain'] / data['strain'].max(axis=0)
    # data['strain'] = data['strain'].transpose()
    #
    # # t_ax = np.arange(len(data['strain'][plt_tr])) / fs
    #
    # trace = data['strain'][plt_tr]
    # cA, cD = pywt.dwt(trace, 'db2')

    # # Fig. 5b

    # file = 'Jousset_et_al_2018_003_Figure5b.ascii'
    # n_trazas = 2551
    # plt_tr = 1550
    # fs = 200
    #
    # data = {
    #     'head': '',
    #     'strain': np.empty((1, n_trazas))
    # }
    #
    # with open(file, 'r') as f:
    #     for idx, line in enumerate(f):
    #         if idx == 0:
    #             data['head'] = line.strip()
    #
    #         else:
    #             row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
    #             data['strain'] = np.concatenate((data['strain'], np.expand_dims(row, 0)))
    #
    # data['strain'] = data['strain'][1:]
    # data['strain'] = data['strain'] / data['strain'].max(axis=0)
    # data['strain'] = data['strain'].transpose()
    #
    # # t_ax = np.arange(len(data['strain'][plt_tr])) / fs
    #
    # trace = data['strain'][plt_tr]
    # cA, cD = pywt.dwt(trace, 'db2')

    # Fig. 5b

    file = 'Jousset_et_al_2018_003_Figure5b.ascii'
    n_trazas = 2551
    plt_tr = 1550
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

    # t_ax = np.arange(len(data['strain'][plt_tr])) / fs

    trace = data['strain'][plt_tr]
    cA, cD = pywt.dwt(trace, 'db2')
    print(cA, cD)


if __name__ == "__main__":
    main()

