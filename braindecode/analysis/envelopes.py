#!/usr/bin/env python
import numpy as np
import scipy.signal
from theano.tensor.signal import downsample
import theano.tensor as T
import argparse
import yaml
import theano
from braindecode.experiments.experiment_runner import create_experiment
from braindecode.datasets.generate_filterbank import generate_filterbank
from braindecode.analysis.util import (lowpass_topo,
                                       highpass_topo,
                                       bandpass_topo)
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
from braindecode.results.results import ResultPool
from braindecode.analysis.kaggle import  transform_to_trial_acts
import logging
log = logging.getLogger(__name__)

def compute_topo_corrs(trial_env, trial_acts):
    # sensors before filterbands
    flat_trial_env = trial_env.transpose(2,0,1,3).reshape(
        trial_env.shape[0] * trial_env.shape[2],
        trial_env.shape[1] * trial_env.shape[3])
    flat_trial_acts = trial_acts.transpose(1,0,2).reshape(
        trial_acts.shape[1],-1)
    flat_corrs = np.corrcoef(flat_trial_env, flat_trial_acts)
    relevant_corrs = flat_corrs[:flat_trial_env.shape[0],
              flat_trial_env.shape[0]:]
    topo_corrs = relevant_corrs.reshape(trial_env.shape[2], trial_env.shape[0],
        trial_acts.shape[1])
    return topo_corrs

def get_meaned_trial_env(env, field_size, n_trials, n_inputs_per_trial,
                        n_trial_len, n_sample_preds):
    inputs = T.ftensor4()
    pooled = downsample.max_pool_2d(inputs, ds=(field_size ,1), st=(1,1), 
                                    ignore_border=True, mode='average_exc_pad')
    pool_fn = theano.function([inputs], pooled)
    meaned_env = np.float32(np.ones(env.shape[0:3] + (env.shape[3] - field_size + 1,1)) * np.nan)
    for i_fb in xrange(env.shape[0]):
        meaned_env[i_fb] = pool_fn(env[i_fb])
    
    all_trial_envs = []
    for fb_env in meaned_env:
        
        fb_envs_per_trial = fb_env.reshape(n_trials,n_inputs_per_trial,fb_env.shape[1],
            fb_env.shape[2], fb_env.shape[3])
        trial_env = transform_to_trial_acts(fb_envs_per_trial, [n_inputs_per_trial] * n_trials,
                                            n_sample_preds=n_sample_preds,
                                            n_trial_len=n_trial_len)
        all_trial_envs.append(trial_env)
    all_trial_envs = np.array(all_trial_envs, dtype=np.float32)
    return all_trial_envs

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
        filtered = filtered.astype(np.float32)
        filt_set = DenseDesignMatrixWrapper(topo_view=filtered,y=train_set.y,
                                            axes=train_set.view_converter.axes)
        batches_topo = [b[0] for b in iterator.get_batches(filt_set, shuffle=False)]
        batches_topo = np.concatenate(batches_topo)
        log.info("Compute envelope...")
        env = np.abs(scipy.signal.hilbert(batches_topo, axis=2))
        env = env.astype(np.float32)
        env_per_filterband.append(env)
        
    log.info("Merge into one array...")
    env_per_filterband = np.array(env_per_filterband, dtype=np.float32)
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
        create_envelopes('data/models/paper/ours/cnt/deep4/car/',
            params=dict(low_cut_off_hz="null"))
        """
        create_envelopes('data/models/paper/ours/cnt/deep4/',
            params=dict(layers='$cnt_4l',
                       pool_mode='max',
                       num_filters_4=200,
                       filter_time_length=10,
                       low_cut_off_hz="null"))
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
        