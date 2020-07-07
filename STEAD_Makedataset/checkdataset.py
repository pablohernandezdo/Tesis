import h5py
import argparse


def main():
    # Check dataset groups lenght

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='STEAD.hdf5', help="HDF5 Dataset path")
    args = parser.parse_args()

    # Open dataset
    with h5py.File(args.dataset_path, 'r') as dset:

        # Read groups
        seismic = dset['earthquake']['local']
        noise = dset['non_earthquake']['noise']

        # Print number of traces
        print(f'Number of seismic traces: {len(seismic)} \n'
              f'Number of noise traces: {len(noise)}')


if __name__ == "__main__":
    main()
