import h5py
import argparse
from numpy.random import default_rng

import tqdm


def main():
    # Create a new hdf5 dataset from STEAD.hdf5

    # Args
    parser = argparse.ArgumentParser(description='Dataset creation parameters')
    parser.add_argument('--source_file', default='STEAD.hdf5', help='Source HDF5 file path')
    parser.add_argument('--train_file', default='Train_data_v2.hdf5', help='Output train HDF5 file path')
    parser.add_argument('--val_file', default='Validation_data_v2.hdf5', help='Output validation HDF5 file path')
    parser.add_argument('--test_file', default='Test_data_v2.hdf5', help='Output test HDF5 file path')
    parser.add_argument('--train_traces', type=int, default=6, help='Number of training seismic traces to copy')
    parser.add_argument('--train_noise', type=int, default=6, help='Number of training noise traces to copy')
    parser.add_argument('--val_traces', type=int, default=2, help='Number of validation seismic traces to copy')
    parser.add_argument('--val_noise', type=int, default=2, help='Number of validation noise traces to copy')
    parser.add_argument('--test_traces', type=int, default=2, help='Number of test seismic traces to copy')
    parser.add_argument('--test_noise', type=int, default=2, help='Number of test noise traces to copy')
    parser.add_argument('--snr_db', type=float, default=0.0, help='Minimum signal to noise ratio')
    parser.add_argument('--azimuth', type=float, default=0.0, help='Back_azimuth_deg parameter')
    parser.add_argument('--source_magnitude', type=float, default=0.0, help='Minimum source magnitude')
    parser.add_argument('--source_distance_km', type=float, default=1000.0, help='Maximum source distance in km')
    args = parser.parse_args()

    # Init rng
    rng = default_rng()

    # Read the hdf5 source file
    with h5py.File(args.source_file, 'r') as source:

        # Retrieve file groups
        src_wv = source['earthquake']['local']
        src_ns = source['non_earthquake']['noise']

        # Total number of traces to copy
        seis2copy = args.train_traces + args.val_traces + args.test_traces
        ns2copy = args.train_noise + args.val_noise + args.test_noise

        # Traces to copy
        seismic_ids = rng.choice(len(src_wv), size=seis2copy, replace=False)
        noise_ids = rng.choice(len(src_ns), size=ns2copy, replace=False)

        train_seis_ids = seismic_ids[:args.train_traces]
        train_noise_ids = noise_ids[:args.train_noise]

        val_seis_ids = seismic_ids[args.train_traces:args.train_traces + args.val_traces]
        val_noise_ids = noise_ids[args.train_noise:args.train_noise + args.val_noise]

        test_seis_ids = seismic_ids[args.train_traces + args.val_traces:args.train_traces + args.val_traces+args.test_traces]
        test_noise_ids = noise_ids[args.train_noise + args.val_noise:args.train_noise + args.val_noise+args.test_noise]

        print(f'seismic_ids: {seismic_ids}\n'
              f'noise_ids: {noise_ids}\n'
              f'train_seis: {train_seis_ids}\n'
              f'train_noise: {train_noise_ids}\n'
              f'val_seis: {val_seis_ids}\n'
              f'val_noise: {val_noise_ids}\n'
              f'test_seis: {test_seis_ids}\n'
              f'test_noise: {test_noise_ids}')

        # # Create new train and test files
        # with h5py.File(args.train_file, 'w') as train_dst:
        #
        #     # Create new train file groups
        #     train_dst_wv = train_dst.create_group('earthquake/local')
        #     train_dst_ns = train_dst.create_group('non_earthquake/noise')
        #
        #     # Number of seismic and noise waves copied
        #     wv_copied = 0
        #     ns_copied = 0
        #
        #     # tqdm progress bars
        #     trn_traces_bar = tqdm.tqdm(total=args.train_traces, desc='Train traces', position=0)
        #     trn_noise_bar = tqdm.tqdm(total=args.train_noise, desc='Train noise', position=1)


if __name__ == '__main__':
    main()
