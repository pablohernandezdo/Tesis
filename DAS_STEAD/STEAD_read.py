import h5py
import numpy as np

import numpy.random as random

def main():
    # Carga traza STEAD

    st = '../Data_STEAD/Train_data.hdf5'

    id = random.randint(0, high=10000, size=1)

    with h5py.File(st, 'r') as h5_file:
        grp = h5_file['earthquake']['local']
        for idx, dts in enumerate(grp):
            if id == idx:
                st_trace = grp[dts][:, 0] / np.max(np.abs(grp[dts][:, 0]))
                break
            else:
                continue

    plt.figure()
    plt.plot(st_trace)
    plt.show()


if __name__ == '__main__':
    main()
