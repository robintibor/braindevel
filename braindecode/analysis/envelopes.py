#!/usr/bin/env python
import numpy as np
import scipy.signal
from theano.tensor.signal import downsample
import theano.tensor as T
import argparse
import yaml
import theano
from braindecode.experiments.experiment import create_experiment
from braindecode.datasets.generate_filterbank import generate_filterbank
from braindecode.analysis.util import (lowpass_topo,
                                       highpass_topo,
                                       bandpass_topo)
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
from braindecode.results.results import ResultPool
from braindecode.analysis.kaggle import  transform_to_trial_acts
import lasagne
from braindecode.veganlasagne.layer_util import get_receptive_field_size
from braindecode.datahandling.batch_iteration import compute_trial_start_end_samples
from braindecode.veganlasagne.layers import get_n_sample_preds
import logging
log = logging.getLogger(__name__)



def load_trial_env(basename, model, i_layer, train_set, n_inputs_per_trial):
    log.info("Loading envelope...")
    env = np.load(basename + '.env.npy')
    env = [e for e in env] # transform that outer part is list so you can freely delete parts inside next function
    log.info("Transforming to trial envelope...")
    trial_env = transform_to_trial_env(env, model, i_layer, train_set, n_inputs_per_trial)
    return trial_env

def transform_to_trial_env(env, model, i_layer, train_set, n_inputs_per_trial):
    all_layers = lasagne.layers.get_all_layers(model)
    layer = all_layers[i_layer]
    field_size = get_receptive_field_size(layer)
    
    trial_starts, trial_ends = compute_trial_start_end_samples(train_set.y)
    assert len(np.unique(trial_ends - trial_starts)) == 1
    n_trials = len(trial_starts)
    n_trial_len = np.unique(trial_ends - trial_starts)[0]
    # Afterwards env is list empty lists(!!)
    n_sample_preds = get_n_sample_preds(model)
    trial_env = get_meaned_trial_env(env, field_size=field_size, n_trials=n_trials,
                                     n_inputs_per_trial=n_inputs_per_trial,
                                n_trial_len=n_trial_len, 
                                n_sample_preds=n_sample_preds)
    return trial_env

def compute_topo_corrs(trial_env, trial_acts):
    # sensors before filterbands
    flat_trial_env = trial_env.transpose(2,0,1,3).reshape(
        trial_env.shape[0] * trial_env.shape[2],
        trial_env.shape[1] * trial_env.shape[3])
    flat_trial_acts = trial_acts.transpose(1,0,2).reshape(
        trial_acts.shape[1],-1)
    #flat_corrs = np.corrcoef(flat_trial_env, flat_trial_acts)
    #relevant_corrs = flat_corrs[:flat_trial_env.shape[0],
    #          flat_trial_env.shape[0]:]
    
    relevant_corrs = corr(flat_trial_env, flat_trial_acts)
    topo_corrs = relevant_corrs.reshape(trial_env.shape[2], trial_env.shape[0],
        trial_acts.shape[1])
    return topo_corrs

def corr(x,y):
    # computing "unbiased" corr
    demeaned_x = x - np.mean(x, axis=1, keepdims=True)
    demeaned_y = y - np.mean(y, axis=1, keepdims=True)
    #ddof=1 for unbiased..
    divisor = np.outer(np.sqrt(np.var(x, axis=1, ddof=1)), 
        np.sqrt(np.var(y, axis=1, ddof=1)))
    
    cov = np.dot(demeaned_x,demeaned_y.T) / (y.shape[1] -1)
    return cov / divisor

def get_meaned_trial_env(env, field_size, n_trials, n_inputs_per_trial,
                        n_trial_len, n_sample_preds):
    inputs = T.ftensor4()
    pooled = downsample.max_pool_2d(inputs, ds=(field_size ,1), st=(1,1), 
                                    ignore_border=True, mode='average_exc_pad')
    pool_fn = theano.function([inputs], pooled)
    log.info("Computing meaned envelope...")
    meaned_env = []
    assert env[0].shape[0] == n_trials * n_inputs_per_trial
    expected_mean_env_shape =  ((len(env),) + env[0].shape[0:2] + 
        (env[0].shape[2] - field_size + 1,1))
    #meaned_env = np.float32(np.ones(env.shape[0:3] + (env.shape[3] - field_size + 1,1)) * np.nan)
    for i_fb in xrange(len(env)):
        #meaned_env[i_fb] = pool_fn(env[i_fb])
        meaned_env.append(pool_fn(env[i_fb]))
        # In order to save memory, delete env contents...
        env[i_fb] = []
    meaned_env = np.array(meaned_env)
    assert meaned_env.shape == expected_mean_env_shape
    log.info("Transforming to per trial...")
    all_trial_envs = []
    for fb_env in meaned_env:
        fb_envs_per_trial = fb_env.reshape(n_trials,n_inputs_per_trial,fb_env.shape[1],
            fb_env.shape[2], fb_env.shape[3])
        trial_env = transform_to_trial_acts(fb_envs_per_trial, [n_inputs_per_trial] * n_trials,
                                            n_sample_preds=n_sample_preds,
                                            n_trial_len=n_trial_len)
        all_trial_envs.append(trial_env)
    log.info("Merging to one array...")
    all_trial_envs = np.array(all_trial_envs, dtype=np.float32)
    log.info("Done...")
    return all_trial_envs

def create_envelopes(folder_name, params, start, stop):
    res_pool = ResultPool()
    res_pool.load_results(folder_name, params=params)
    res_file_names = res_pool.result_file_names()
    yaml_file_names = [name.replace('.result.pkl', '.yaml')
        for name in res_file_names]
    stop = stop or len(yaml_file_names)
    i_file = start
    for i_file in xrange(start, stop):
        file_name = yaml_file_names[i_file]
        log.info("Running {:s} ({:d} of {:d})".format(
            file_name, i_file+1, stop))
        create_envelopes_for_experiment(file_name)


def create_envelopes_for_experiment(experiment_file_name):
    iterator, train_set = _load_experiment(experiment_file_name)
    filterbands = generate_filterbank(min_freq=1, max_freq=115,
        last_low_freq=31, low_width=6, low_overlap=3,
        high_width=8, high_overlap=4)
    env_per_filterband = create_envelops_per_filterband(iterator,
        train_set, filterbands)
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
    return exp.iterator, train_set

def create_envelops_per_filterband(iterator, train_set, filterbands):
    env_per_filterband = []
    train_topo = train_set.get_topological_view()
    for low_cut_hz, high_cut_hz in filterbands:
        log.info("Compute filterband from {:.1f} to {:.1f}...".format(
            low_cut_hz, high_cut_hz))
        if low_cut_hz > 0 and high_cut_hz < 125:
            filtered = bandpass_topo(train_topo, low_cut_hz, 
                                     high_cut_hz, sampling_rate=250.0, axis=0, filt_order=4)
        elif low_cut_hz == 0:
            filtered = lowpass_topo(train_topo, high_cut_hz, 
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
    parser.add_argument('--start', default=1, type=int,
                        help='''Start with this envelope file index (1-based, after selection of results)''')
    parser.add_argument('--stop', default=None, type=int, 
                        help='''Stop at this envelope file index (1-based, after selection ofresults)''')
    args = parser.parse_args()

    # dictionary values are given with = inbetween, parse them here by hand
    params_and_values = [(param_and_value.split('=')[0],
        yaml.load(param_and_value.split('=')[1]))
                        for param_and_value in args.params]
    param_dict =  dict(params_and_values)

    
    args.params = param_dict
    # Correct for 1 based index, transform to 0-based
    args.start = args.start - 1    
    return args

if __name__ == "__main__":
    logging.basicConfig(level='DEBUG')
    args = parse_command_line_arguments()
    if args.folder is None:
        create_envelopes('data/models-backup/paper/ours/cnt/deep4/car/',
            params=dict(sensor_names="$all_EEG_sensors", batch_modifier="null",
                        low_cut_off_hz="null", first_nonlin="$elu"),
                        start=args.start, stop=args.stop)
    else:
        print args.experiments_folder
        print args.params
        create_envelopes(args.folder, args.params, start=args.start, stop=args.stop)
        
