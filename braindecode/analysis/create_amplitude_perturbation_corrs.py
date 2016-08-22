import sys
import os.path
import numpy as np
from numpy.random import RandomState
from braindecode.util import FuncAndArgs
from braindecode.analysis.stats import median_absolute_deviation, cov
from braindecode.experiments.load import load_exp_and_model
from braindecode.veganlasagne.layers import create_pred_fn, get_n_sample_preds
from braindecode.datasets.fft import amplitude_phase_to_complex
from braindecode.results.results import ResultPool
import logging
from lasagne.nonlinearities import identity
from braindecode.datahandling.batch_iteration import compute_trial_start_end_samples
from copy import deepcopy
log = logging.getLogger(__name__)

def create_all_amplitude_perturbation_corrs(folder_name, params,
        start, stop, with_blocks, with_square, with_square_cov, after_softmax,
        n_samples):
    assert not (with_square and with_square_cov)
    res_pool = ResultPool()
    res_pool.load_results(folder_name, params=params)
    res_file_names = res_pool.result_file_names()
    all_base_names = [name.replace('.result.pkl', '')
        for name in res_file_names]
    start = start or 0
    stop = stop or len(all_base_names)
    for i_file, base_name in enumerate(all_base_names[start:stop]):
        log.info("Running {:s} ({:d} of {:d})".format(
            base_name, i_file + start + 1, stop))
        create_amplitude_perturbation_corrs(base_name, 
            with_blocks=with_blocks,
            with_square=with_square, with_square_cov=with_square_cov,
            after_softmax=after_softmax,
            n_samples=n_samples)

def create_amplitude_perturbation_corrs(basename, with_blocks,
        with_square, with_square_cov, after_softmax,
        n_samples=30):
    assert not (with_square and with_square_cov)
    exp, pred_fn = load_exp_pred_fn(basename, after_softmax=after_softmax)
    log.info("Create fft trials...")
    trials, amplitudes, phases, targets = create_trials_and_do_fft(exp)
    log.info("Create all predictions...")
    all_preds = np.array([pred_fn(t) for t in trials] )
    for name, perturb_fn in (
                            ('no_dev', FuncAndArgs(rand_diff,
                                  with_blocks=with_blocks, #just return 1
                                  deviation_func=lambda arr,axis,keepdims: 1),
                            ('rand_mad',
                              FuncAndArgs(rand_diff,
                                  with_blocks=with_blocks,
                                  deviation_func=median_absolute_deviation)),
                            ('rand_std', FuncAndArgs(rand_diff,
                                  with_blocks=with_blocks, 
                                  deviation_func=np.std)),
                             )):
        
        file_name_end_prefix = '.{:d}'.format(n_samples)
        if with_square:
            file_name_end_prefix = ".square" + file_name_end_prefix
        if with_square_cov:
            file_name_end_prefix = ".covtosquare" + file_name_end_prefix
        if after_softmax:
            file_name_end_prefix = ".after_softmax" + file_name_end_prefix
        file_name_end = '.{:s}.amp_cov_vars.npz'.format(name)   
        save_filename = basename + file_name_end_prefix + file_name_end
        pred_file_name_end = '.{:s}.amp_cov_var_preds_targets.npz'.format(name)   
        pred_save_filename = basename + file_name_end_prefix + pred_file_name_end
        # check if file already exists, skip if it does and is loadable
        if os.path.isfile(save_filename):
            try:
                log.info("Trying to load {:s}".format(save_filename))
                np.load(save_filename)
                log.info("Skipping {:s}, already exists".format(
                    save_filename))
                continue
            except:
                pass
        rng = RandomState(49587489)
        log.info("Create perturbed preds covs for {:s}...".format(name))
        all_covs_and_vars, all_new_trial_preds = create_perturbed_preds_covs(
            amplitudes, 
            phases, all_preds, pred_fn, perturb_fn, rng,
            with_square=with_square,
            with_square_cov=with_square_cov,
            n_samples=n_samples)
        
        log.info("Saving to {:s} and {:s}...".format(save_filename,
            pred_save_filename))
        np.savez(save_filename, *all_covs_and_vars)
        trial_targets = np.mean(targets, axis=1)
        np.savez(pred_save_filename, all_new_trial_preds, trial_targets)
        log.info("Done.")
    
def load_exp_pred_fn(basename, after_softmax):
    exp, model = load_exp_and_model(basename)
    if not after_softmax:
        # replace softmax by identity to get better correlations
        assert (model.nonlinearity.func_name == 'softmax' or
            model.nonlinearity.func_name == 'safe_softmax')
        model.nonlinearity = identity
    # load dataset
    exp.dataset.load()
    log.info("Create prediction function...")
    pred_fn = create_pred_fn(model)
    log.info("Done.")
    return exp, pred_fn
    
def create_trials_and_do_fft(exp):
    train_set = exp.dataset_provider.get_train_merged_valid_test(
        exp.dataset)['train']
    n_sample_preds = get_n_sample_preds(exp.final_layer)
    trials, targets = get_trials_targets(train_set, n_sample_preds,
        exp.iterator)
    ffted = np.fft.rfft(trials, axis=3)
    amplitudes = np.abs(ffted)
    phases = np.angle(ffted)
    # targets are trials x samples x classes
    return trials, amplitudes, phases, targets

def get_trials_targets(train_set, n_sample_preds, iterator):
    iterator = deepcopy(iterator) # will be modified..
    starts, ends = compute_trial_start_end_samples(train_set.y,
        check_trial_lengths_equal=True)
    trial_len = ends[0] - starts[0]
    iterator.n_sample_preds = n_sample_preds
    # -> get one batch per trial by setting batch size
    iterator.batch_size = int(np.ceil(trial_len / float(n_sample_preds)))
    batches = list(iterator.get_batches(train_set, shuffle=False))
    trials, targets = zip(*batches)
    trials = np.array(trials)
    targets = np.array(targets)
    return trials, targets

def create_perturbed_preds_covs(amplitudes, phases, all_preds, 
        pred_fn, perturb_fn, rng,
        with_square, with_square_cov, n_samples=30):
    assert not (with_square and with_square_cov)
    rng = RandomState(3874638746)
    if with_square:
        amplitudes = np.square(amplitudes)
    dummy_diff_amp = perturb_fn(amplitudes, RandomState(34534))
    all_covs = []
    all_var_preds = []
    all_var_amps = []
    all_new_trial_preds = []
    for i_sample in xrange(n_samples):
        log.info("Sample {:d} of {:d}".format(i_sample+1, n_samples))
        diff_amp = perturb_fn(amplitudes, rng)
        # clip in case diff would move amplitudes to negative region
        # min over batches and empty dim
        diff_amp = np.maximum(-np.min(amplitudes, axis=(1,4), keepdims=True),
            diff_amp)
        new_amp = amplitudes + diff_amp
        if with_square:
            # invert square from above...
            new_amp = np.sqrt(new_amp)
        new_fft = amplitude_phase_to_complex(new_amp, phases)
        new_trials = np.fft.irfft(new_fft, axis=3).astype(np.float32)
        new_preds = np.array([pred_fn(t) for t in new_trials] )
        # new preds have shape trials x samples x classes
        if with_square_cov:
            diff_amp = np.square(new_amp) - np.square(amplitudes)
        pred_diffs =  new_preds - all_preds
        # probably not necessary to put in brackets, but unchecked...
        pred_diff_amp_cov, var_preds, var_amps = compute_class_amp_sensor_covs(
            pred_diffs, diff_amp)
        all_covs.append(pred_diff_amp_cov)
        all_var_preds.append(var_preds)
        all_var_amps.append(var_amps)
        # mean over samples -> afterwards trials x classes
        trial_preds =  np.mean(new_preds, axis=1)
        all_new_trial_preds.append(trial_preds)
    log.info("Make into array...")
    all_covs = np.array(all_covs,dtype=np.float32)
    all_var_preds = np.array(all_var_preds,dtype=np.float32)
    all_var_amps = np.array(all_var_amps,dtype=np.float32)
    all_new_trial_preds = np.array(all_new_trial_preds,dtype=np.float32)
    all_covs_and_vars = [all_covs, all_var_preds, all_var_amps]
    log.info("Done.")
    return all_covs_and_vars, all_new_trial_preds

def compute_class_amp_sensor_covs(pred_diffs, amp_diffs):
    # amp diffs shape:
    # trials x 1 x sensors x freqs x 1
    amp_diffs = amp_diffs.squeeze()
    # -> trials x sensors x freqs
    amp_for_cov = amp_diffs.T.reshape(amp_diffs.shape[-2] * amp_diffs.shape[-1], -1)
    # ->(freqs x sensors) x (trials)
    
    # pred diffs shape trials x samples x classes
    # mean pred diff per trial, then transpose to classes in first dim
    # -> classes x trials
    pred_diff_for_cov = np.mean(pred_diffs.T, axis=1)
    wanted_coeffs = cov(pred_diff_for_cov.astype(np.float32),
        amp_for_cov.astype(np.float32))
    # wanted coeffs shape: classes x freqs x sensors
    wanted_coeffs = wanted_coeffs.reshape(-1,amp_diffs.shape[2], amp_diffs.shape[1])
    # -> classes x sensors x freqs
    wanted_coeffs = wanted_coeffs.transpose(0,2,1)
    # ddof=1 for later unbiased corr
    vars_preds = np.var(pred_diff_for_cov, axis=1, ddof=1) # classes
    vars_amp = np.var(amp_diffs, axis=0, ddof=1) # sensors x freqs
    return wanted_coeffs, vars_preds, vars_amp 

def rand_diff(amplitudes, rng, with_blocks, deviation_func=median_absolute_deviation):
    all_diff_amp = np.zeros((amplitudes.shape[0],
                            1,
                            amplitudes.shape[2],
                            amplitudes.shape[3],
                            1), dtype=np.float32)
    if with_blocks:
        diff_amp = rng.randn(amplitudes.shape[0],
                             amplitudes.shape[2],
                             amplitudes.shape[3] // 20)
        diff_amp = np.repeat(diff_amp, 20, axis=2)
        assert diff_amp.shape[2] == all_diff_amp.shape[3] - 1
        all_diff_amp[:,:,:,:diff_amp.shape[2],:] = diff_amp[:,None,:,:,None]
    else:
        diff_amp = rng.randn(amplitudes.shape[0],
                             amplitudes.shape[2],
                             amplitudes.shape[3])
        all_diff_amp[:,:,:,:,:] = diff_amp[:,None,:,:,None]
        
    # calculate deviation over trials, batches and fourth emptydim, keep 
    # sensors and frequencies (one deviation value per sensor x frequency)
    all_diff_amp = all_diff_amp * deviation_func(amplitudes, axis=(0,1,4),
        keepdims=True)
    return all_diff_amp

def shuffle_per_freq_block(amplitudes, rng):
    indices = range(0,amplitudes.shape[0])
    diff_amp = np.zeros((amplitudes.shape[0],
                                1,
                                amplitudes.shape[2],
                                amplitudes.shape[3],
                                1), dtype=np.float32)
    for i_freq in xrange(0, amplitudes.shape[3], 20):
        rng.shuffle(indices)
        diff_amp[:,:,:,i_freq:i_freq+20] = (amplitudes[indices,1:2,:,i_freq:i_freq+20,:] -
                                        amplitudes[:,1:2,:,i_freq:i_freq+20,:])
    return diff_amp


def setup_logging():
    """ Set up a root logger so that other modules can use logging
    Adapted from scripts/train.py from pylearn"""

    from pylearn2.utils.logger import (CustomStreamHandler, CustomFormatter)

    root_logger = logging.getLogger()
    prefix = '%(asctime)s '
    formatter = CustomFormatter(prefix=prefix)
    handler = CustomStreamHandler(formatter=formatter)
    root_logger.handlers  = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
if __name__ == "__main__":
    setup_logging()
    start = None
    stop = None
    if len(sys.argv) > 1:
        start = int(sys.argv[1]) - 1 # from 1-based to 0-based
    if len(sys.argv) > 2:
        stop = int(sys.argv[2])
    folder = 'data/models/paper/ours/cnt/deep4/car/'
    params = dict(cnt_preprocessors="$cz_zero_resample_car_demean")
    with_square = False
    with_square_cov = False
    with_blocks=False
    after_softmax = False
    n_samples = 400
    create_all_amplitude_perturbation_corrs(folder,
             params=params, start=start,stop=stop,
             with_blocks=with_blocks,
             with_square=with_square, with_square_cov=with_square_cov,
             after_softmax=after_softmax,
             n_samples=n_samples)
#     create_all_amplitude_perturbation_corrs('data/models-backup/paper/ours/cnt/shallow/car/',
#         params=None)