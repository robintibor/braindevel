import sys
import os.path
import numpy as np
from numpy.random import RandomState
from braindecode.util import FuncAndArgs
from braindecode.analysis.stats import median_absolute_deviation, corr, cov
from braindecode.experiments.load import load_exp_and_model
from braindecode.veganlasagne.layers import create_pred_fn
from braindecode.datasets.fft import amplitude_phase_to_complex
from braindecode.results.results import ResultPool
import logging
log = logging.getLogger(__name__)

def create_all_amplitude_perturbation_corrs(folder_name, params,
        start, stop, with_blocks, with_square, with_square_cov, n_samples):
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
            n_samples=n_samples)

def create_amplitude_perturbation_corrs(basename, with_blocks,
        with_square, with_square_cov,
        n_samples=30):
    assert not (with_square and with_square_cov)
    exp, pred_fn = load_exp_pred_fn(basename)
    log.info("Create fft trials...")
    trials, amplitudes, phases = create_trials_and_do_fft(exp)
    log.info("Create all predictions...")
    all_preds = np.array([pred_fn(t) for t in trials] )
    for name, perturb_fn in (('rand_mad',
                              FuncAndArgs(rand_diff,
                                  with_blocks=with_blocks,
                                  deviation_func=median_absolute_deviation)),
                              ('rand_std', FuncAndArgs(rand_diff,
                                  with_blocks=with_blocks, 
                                  deviation_func=np.std)),
                             ('shuffle', shuffle_per_freq_block)):
        file_name_end = '.{:s}.amp_covs.npy'.format(name)
        if with_square:
            file_name_end = ".square" + file_name_end
        if with_square_cov:
            file_name_end = ".covtosquare" + file_name_end
            
        save_filename = basename + file_name_end
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
        all_covs = create_perturbed_preds_covs(amplitudes, 
            phases, all_preds, pred_fn, perturb_fn, rng, with_square=with_square,
            with_square_cov=with_square_cov,
            n_samples=n_samples)
        
        log.info("Saving...")
        np.save(save_filename,  all_covs)
        log.info("Done.")
    
def load_exp_pred_fn(basename):
    exp, model = load_exp_and_model(basename)
    exp.dataset.load()
    log.info("Create prediction function...")
    pred_fn = create_pred_fn(model.input_layer)
    log.info("Done.")
    return exp, pred_fn
    
def create_trials_and_do_fft(exp):
    # -> hack to get one batch per trial
    exp.iterator.batch_size = 2
    batches = list(exp.iterator.get_batches(exp.dataset.train_set, shuffle=False))
    trials, _ = zip(*batches)
    trials = np.array(trials)
    ffted = np.fft.rfft(trials, axis=3)
    amplitudes = np.abs(ffted)
    phases = np.angle(ffted)
    return trials, amplitudes, phases

def create_perturbed_preds_covs(amplitudes, phases, all_preds, 
        pred_fn, perturb_fn, rng,
        with_square, with_square_cov, n_samples=30):
    assert not (with_square and with_square_cov)
    rng = RandomState(3874638746)
    if with_square:
        amplitudes = np.square(amplitudes)
    dummy_diff_amp = perturb_fn(amplitudes, RandomState(34534))
    all_covs = []
    
    for i_sample in xrange(n_samples):
        log.info("Sample {:d} of {:d}".format(i_sample+1, n_samples))
        diff_amp = perturb_fn(amplitudes, rng)
        if with_square:
            # invert square...
            # clip for unlikely case diff is below 0
            new_amp = np.sqrt(np.maximum(amplitudes + diff_amp,0))
        else:
            new_amp = amplitudes + diff_amp
        new_fft = amplitude_phase_to_complex(new_amp, phases)
        new_trials = np.fft.irfft(new_fft, axis=3).astype(np.float32)
        new_preds = np.array([pred_fn(t) for t in new_trials] )
        if with_square_cov:
            # done here already, not outside funciton,
            # since i assume outside funciton might create
            # memory problems due to copies...
            # but never tested this
            diff_amp = np.square(diff_amp) * np.sign(diff_amp)
        pred_diffs = all_preds - new_preds
        # probably not necessary to put in brackets, but unchecked...
        pred_diff_amp_cov = compute_class_amp_sensor_covs(
            np.array([pred_diffs]), 
            np.array([diff_amp]))
        all_covs.append(pred_diff_amp_cov)
    log.info("Make into array...")
    all_covs = np.array(all_covs,dtype=np.float32)
    log.info("Done.")
    return all_covs

def compute_class_amp_sensor_covs(pred_diffs, amp_diffs):
    amp_diffs = amp_diffs.squeeze()
    amp_for_cov = amp_diffs.T.reshape(amp_diffs.shape[-2] * amp_diffs.shape[-1], -1)
    pred_diff_for_cov = np.mean(pred_diffs.T, axis=1).reshape(4,-1)
    wanted_coeffs = cov(pred_diff_for_cov.astype(np.float32),
        amp_for_cov.astype(np.float32))
    wanted_coeffs = wanted_coeffs.reshape(-1,amp_diffs.T.shape[0], amp_diffs.T.shape[1])
    return wanted_coeffs

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
        start = int(sys.argv[1])
    if len(sys.argv) > 2:
        stop = int(sys.argv[2])
    folder = 'data/models/paper/ours/cnt/deep4/car/'
    params = dict(sensor_names="$all_EEG_sensors", batch_modifier="null",
                         low_cut_off_hz="null", first_nonlin="$elu")
    with_square = False
    with_square_cov = False
    with_blocks=False
    n_samples = 400
    create_all_amplitude_perturbation_corrs(folder,
             params=params, start=start,stop=stop,
             with_blocks=with_blocks,
             with_square=with_square, with_square_cov=with_square_cov,
             n_samples=n_samples)
#     create_all_amplitude_perturbation_corrs('data/models-backup/paper/ours/cnt/shallow/car/',
#         params=None)