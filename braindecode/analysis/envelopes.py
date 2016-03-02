#!/usr/bin/env python
import numpy as np
import scipy.signal
from braindecode.experiments.experiment_runner import create_experiment
from braindecode.datasets.generate_filterbank import generate_filterbank
from braindecode.analysis.util import (lowpass_topo,
                                       highpass_topo,
                                       bandpass_topo)
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
import logging
from braindecode.results.results import ResultPool
import argparse
import yaml
log = logging.getLogger(__name__)

def create_envelopes(folder_name, params):
    res_pool = ResultPool()
    res_pool.load_results(folder_name, params=params)
    res_file_names = res_pool.result_file_names()
    yaml_file_names = [name.replace('.result.pkl', '.yaml')
        for name in res_file_names]
    for i_file, file_name in enumerate(yaml_file_names):
        log.info("Running {:s} ({:d} of {:d})".format(
            file_name, i_file+1, len(yaml_file_names)))
        create_envelopes_for_experiment(file_name)


def create_envelopes_for_experiment(experiment_file_name):
    iterator, train_set, train_topo = _load_experiment(experiment_file_name)
    filterbands = generate_filterbank(min_freq=1, max_freq=99,
        last_low_freq=31, low_width=6, low_overlap=3,
        high_width=8, high_overlap=4)
    env_per_filterband = create_envelops_per_filterband(iterator,
        train_set, train_topo, filterbands)
    log.info("Saving...")
    np.save(experiment_file_name.replace('.yaml', '.env.npy'),
        env_per_filterband)
    np.save(experiment_file_name.replace('.yaml', '.filt_bands.npy'),
        filterbands)
    log.info("Done.")
    
def _load_experiment(experiment_file_name):
    exp = create_experiment(experiment_file_name)
    exp.dataset.load()
    train_set = exp.dataset_provider.get_train_merged_valid_test(exp.dataset)['train']
    train_topo = train_set.get_topological_view()
    return exp.iterator, train_set, train_topo

def create_envelops_per_filterband(iterator, train_set, train_topo,
    filterbands):
    env_per_filterband = []
    for low_cut_hz, high_cut_hz in filterbands:
        log.info("Compute filterband from {:.1f} to {:.1f}...".format(
            low_cut_hz, high_cut_hz))
        if low_cut_hz > 0 and high_cut_hz < 125:
            filtered = bandpass_topo(train_topo, low_cut_hz, 
                                     high_cut_hz, sampling_rate=250.0, axis=0, filt_order=4)
        elif low_cut_hz == 0:
            filtered = lowpass_topo(train_topo, high_cut_hz, 
                                sampling_rate=250.0, axis=0, filt_order=4)
        elif high_cut_hz == 125:
            filtered = highpass_topo(train_topo, low_cut_hz, 
                                sampling_rate=250.0, axis=0, filt_order=4)
        filt_set = DenseDesignMatrixWrapper(topo_view=filtered,y=train_set.y,
                                            axes=train_set.view_converter.axes)
        batches_topo = [b[0] for b in iterator.get_batches(filt_set, shuffle=False)]
        batches_topo = np.concatenate(batches_topo)
        log.info("Compute envelope...")
        env = np.abs(scipy.signal.hilbert(batches_topo, axis=2))
        env_per_filterband.append(env)
        
    log.info("Merge into one array...")
    env_per_filterband = np.array(env_per_filterband)
    log.info("Done.")
    return env_per_filterband
def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Launch an experiment from a YAML experiment file.
        Example: ./analysis/envelopes.py --folder data/models/paper/bci-competition/cnt/deep4/ --params layers=\$cnt_4_l pool_mode=max num_filters_4=200 filter_time_length=10 low_cut_off_hz='"null"'
        """
    )
    parser.add_argument('--folder', action='store',
        default=None,
        help='the folder of the experiment')
    parser.add_argument('--params', nargs='*', default=[],
                        help='''Parameters to override default values/other values given in experiment file.
                        Supply it in the form parameter1=value1 parameters2=value2, ...''')
    args = parser.parse_args()

    # dictionary values are given with = inbetween, parse them here by hand
    params_and_values = [(param_and_value.split('=')[0],
        yaml.load(param_and_value.split('=')[1]))
                        for param_and_value in args.params]
    param_dict =  dict(params_and_values)
    
    args.params = param_dict
    
    return args

if __name__ == "__main__":
    logging.basicConfig(level='DEBUG')
    args = parse_command_line_arguments()
    if args.folder is None:
        
        create_envelopes('data/models/paper/ours/cnt/deep4/',
            params=dict(layers='$cnt_4l',
                       pool_mode='max',
                       num_filters_4=200,
                       filter_time_length=10,
                       low_cut_off_hz="null"))
        """
        create_envelopes('data/models/paper/bci-competition/cnt/deep4/',
            params=dict(layers='$cnt_4l',
                       pool_mode='max',
                       num_filters_4=200,
                       filter_time_length=10,
                       low_cut_off_hz="null"))"""
    else:
        print args.experiments_folder
        print args.params
        create_envelopes(args.folder, args.params)
        
