import h5py
import argparse

import tqdm


def main():
    # Create a new hdf5 dataset from STEAD.hdf5

    # Args
    parser = argparse.ArgumentParser(description='Dataset creation parameters')
    parser.add_argument('--source_file', default='STEAD.hdf5', help='Source HDF5 file path')
    parser.add_argument('--train_file', default='Train_data.hdf5', help='Output train HDF5 file path')
    parser.add_argument('--test_file', default='Test_data.hdf5', help='Output test HDF5 file path')
    parser.add_argument('--train_traces', type=int, default=25000, help='Number of training seismic traces to copy')
    parser.add_argument('--train_noise', type=int, default=25000, help='Number of training noise traces to copy')
    parser.add_argument('--test_traces', type=int, default=2500, help='Number of test seismic traces to copy')
    parser.add_argument('--test_noise', type=int, default=2500, help='Number of test noise traces to copy')
    parser.add_argument('--snr_db', type=float, default=0.0, help='Minimum signal to noise ratio')
    parser.add_argument('--azimuth', type=float, default=0.0, help='Back_azimuth_deg parameter')
    parser.add_argument('--source_magnitude', type=float, default=0.0, help='Minimum source magnitude')
    parser.add_argument('--source_distance_km', type=float, default=1000.0, help='Maximum source distance in km')
    args = parser.parse_args()

    # Read the hdf5 source file
    with h5py.File(args.source_file, 'r') as source:

        # Retrieve file groups
        src_wv = source['earthquake']['local']
        src_ns = source['non_earthquake']['noise']

        # Create new train and test files
        with h5py.File(args.train_file, 'w') as train_dst, h5py.File(args.test_file, 'w') as test_dst:

            # Create new train file groups
            train_dst_wv = train_dst.create_group('earthquake/local')
            train_dst_ns = train_dst.create_group('non_earthquake/noise')

            # Create new test file groups
            test_dst_wv = test_dst.create_group('earthquake/local')
            test_dst_ns = test_dst.create_group('non_earthquake/noise')

            # Number of seismic and noise waves copied
            wv_copied = 0
            ns_copied = 0

            # tqdm progress bars
            trn_traces_bar = tqdm.tqdm(total=args.train_traces, desc='Train traces', position=0)
            tst_traces_bar = tqdm.tqdm(total=args.test_traces, desc='Test traces', position=1)
            trn_noise_bar = tqdm.tqdm(total=args.train_noise, desc='Train noise', position=2)
            tst_noise_bar = tqdm.tqdm(total=args.test_noise, desc='Test noise', position=3)

            # For every dataset in source seismic group
            for wv in src_wv:

                # Retrieve dataset object
                data = src_wv[wv]

                # Check creation conditions
                if (min(data.attrs['snr_db']) > args.snr_db and
                        float(data.attrs['source_magnitude']) > args.source_magnitude and
                        float(data.attrs['source_distance_km']) < args.source_distance_km):

                    # If not enough train seismic traces copied
                    if wv_copied < args.train_traces:

                        # Copy seismic trace to new train file
                        train_dst_wv.copy(data, wv)

                        # Count trace and update seismic train tqdm bar
                        wv_copied += 1
                        trn_traces_bar.update(1)

                    # If train traces already copied and test yet not
                    elif wv_copied < args.train_traces + args.test_traces:

                        # Copy seismic trace to new test file
                        test_dst_wv.copy(data, wv)

                        # Count trace and update seismic test tqdm bar
                        wv_copied += 1
                        tst_traces_bar.update(1)

                    else:
                        # Stop copying
                        break

                else:
                    # Don't copy
                    continue

            # For every dataset in source noise group
            for ns in src_ns:

                # Retrieve dataset object
                noise = src_ns[ns]

                # If not enough train noise traces copied
                if ns_copied < args.train_noise:

                    # Copy noise trace to new train file
                    train_dst_ns.copy(noise, ns)

                    # Count trace and update noise train tqdm bar
                    ns_copied += 1
                    trn_noise_bar.update(1)

                elif ns_copied < args.train_noise + args.test_noise:

                    # Copy noise trace to new test file
                    test_dst_ns.copy(noise, ns)

                    # Count trace and update noise test tqdm bar
                    ns_copied += 1
                    tst_noise_bar.update(1)

                else:
                    # Stop copying
                    break


if __name__ == '__main__':
    main()
