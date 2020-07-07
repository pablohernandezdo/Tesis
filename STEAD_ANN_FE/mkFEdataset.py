import h5py
import argparse

import tqdm

from .featex import *

def main():
    parser = argparse.ArgumentParser(description='Conditions for dataset to create')
    parser.add_argument('--source_file', default='STEAD.hdf5', help='Source HDF5 file path')
    parser.add_argument('--train_file', default='Train_data.hdf5', help='Output train HDF5 file path')
    parser.add_argument('--test_file', default='Test_data.hdf5', help='Output test HDF5 file path')
    parser.add_argument('--train_traces', type=int, default=1e4, help='Number of training seismic traces to copy')
    parser.add_argument('--train_noise', type=int, default=1e4, help='Number of training noise traces to copy')
    parser.add_argument('--test_traces', type=int, default=1e2, help='Number of test seismic traces to copy')
    parser.add_argument('--test_noise', type=int, default=1e2, help='Number of test noise traces to copy')
    parser.add_argument('--snr_db', type=float, default=0.0, help='Minimum signal to noise ratio')
    parser.add_argument('--azimuth', type=float, default=0.0, help='Back_azimuth_deg parameter')
    parser.add_argument('--source_magnitude', type=float, default=0.0, help='Minimum source magnitude')
    parser.add_argument('--source_distance_km', type=float, default=1000.0, help='Maximum source distance in km')
    args = parser.parse_args()

    with h5py.File(args.source_file, 'r') as source:
        # data from source file
        src_wv = source['earthquake']['local']
        src_ns = source['non_earthquake']['noise']

        with h5py.File(args.train_file, 'w') as train_dst, h5py.File(args.test_file, 'w') as test_dst:
            # new file groups
            train_dst_wv = train_dst.create_group('earthquake/local')
            train_dst_ns = train_dst.create_group('non_earthquake/noise')

            test_dst_wv = test_dst.create_group('earthquake/local')
            test_dst_ns = test_dst.create_group('non_earthquake/noise')

            wv_copied = 0
            # seismic waverforms

            trn_traces_bar = tqdm.tqdm(total=args.train_traces, desc='Train traces', position=0)
            tst_traces_bar = tqdm.tqdm(total=args.test_traces, desc='Test traces', position=1)
            trn_noise_bar = tqdm.tqdm(total=args.train_noise, desc='Train noise', position=2)
            tst_noise_bar = tqdm.tqdm(total=args.test_noise, desc='Test noise', position=3)

            for wv in src_wv:
                data = src_wv[wv]
                if (min(data.attrs['snr_db']) > args.snr_db and
                        float(data.attrs['source_magnitude']) > args.source_magnitude and
                        float(data.attrs['source_distance_km']) < args.source_distance_km):

                    if wv_copied < args.train_traces:
                        # HACER LA EXTRACCION DE CARACTERISTICAS
                        # GUARDAR LAS CARACTERISTICAS EN UN ARCHIVO
                        train_dst_wv.copy(data, wv)
                        wv_copied += 1
                        trn_traces_bar.update(1)

                    elif wv_copied < args.train_traces + args.test_traces:
                        # HACER LA EXTRACCION DE CARACTERISTICAS
                        # GUARDAR LAS CARACTERISTICAS EN UN ARCHIVO
                        test_dst_wv.copy(data, wv)
                        wv_copied += 1
                        tst_traces_bar.update(1)

                    else:
                        break

                else:
                    continue

            ns_copied = 0
            # noise waveforms
            for ns in src_ns:
                noise = src_ns[ns]
                if ns_copied < args.train_noise:
                    # HACER LA EXTRACCION DE CARACTERISTICAS
                    # GUARDAR LAS CARACTERISTICAS EN UN ARCHIVO
                    train_dst_ns.copy(noise, ns)
                    ns_copied += 1
                    trn_noise_bar.update(1)

                elif ns_copied < args.train_noise + args.test_noise:
                    # HACER LA EXTRACCION DE CARACTERISTICAS
                    # GUARDAR LAS CARACTERISTICAS EN UN ARCHIVO
                    test_dst_ns.copy(noise, ns)
                    ns_copied += 1
                    tst_noise_bar.update(1)

                else:
                    break


if __name__ == '__main__':
    main()
