import h5py
import numpy as np

import torch
from torch.utils import data


class HDF5Dataset(data.Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as h5_file:
            grp1 = h5_file['earthquake']['local']
            grp2 = h5_file['non_earthquake']['noise']
            self.traces_len = len(grp1)
            self.noise_len = len(grp2)

    def __len__(self):
        return self.traces_len + self.noise_len

    def __getitem__(self, item):
        with h5py.File(self.file_path, 'r') as h5_file:
            if item >= self.traces_len:
                item -= self.traces_len
                grp = h5_file['non_earthquake']['noise']
                for idx, dts in enumerate(grp):
                    if idx == item:
                        out = grp[dts]['features']                          # ASI DEBERIA SER, O ALGO PARECIDO
                        return torch.from_numpy(out), torch.Tensor([0])

            else:
                grp = h5_file['earthquake']['local']
                for idx, dts in enumerate(grp):
                    if idx == item:
                        out = grp[dts]['features']                          # ASI DEBERIA SER, O ALGO PARECIDO
                        return torch.from_numpy(out), torch.Tensor([1])
