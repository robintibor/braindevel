import numpy as np
import sys
from braindecode.analysis.envelopes import (load_trial_env,
    compute_topo_corrs)
from braindecode.experiments.load import load_exp_and_model
from braindecode.results.results import ResultPool
from braindecode.datahandling.batch_iteration import (
    compute_trial_start_end_samples) 
import logging
from braindecode.experiments.experiment import create_experiment
from braindecode.analysis.create_env_corrs import compute_trial_topo_corrs
from braindecode.veganlasagne.layers import create_pred_fn
from braindecode.veganlasagne.monitors import compute_preds_per_trial
log = logging.getLogger(__name__)


def create_env_class_corrs(folder, params,start,stop):
    res_pool = ResultPool()
    res_pool.load_results(folder, params=params)
    res_file_names = res_pool.result_file_names()

    all_base_names = [name.replace('.result.pkl', '')
            for name in res_file_names]
    start = start or 0
    stop = stop or len(all_base_names)
    
    
    with_square = True    
    for i_exp, base_name in enumerate(all_base_names[start:stop]):
        log.info("Running {:s} ({:d} of {:d})".format(base_name,
            i_exp + start + 1, stop))
        create_env_class_corr_file(base_name, with_square)

def create_env_class_corr_file(base_name, with_square):
    exp, model = load_exp_and_model(base_name)
    exp.dataset.load()    
    train_set = exp.dataset_provider.get_train_merged_valid_test(exp.dataset)['train']

    trial_env = load_trial_env(base_name + '.env.npy',
           model, i_layer=26, # 26 is last max-pool i think 
           train_set=train_set,
          n_inputs_per_trial=2,
          square_before_mean=with_square)
    topo_corrs = compute_env_class_corr(exp, trial_env)
    rand_model = create_experiment(base_name + '.yaml').final_layer

    rand_topo_corrs = compute_rand_preds_topo_corr(exp, rand_model, 
        trial_env)
    file_name_end = 'class.npy'
    if with_square:
        file_name_end = 'square.' + file_name_end
    np.save('{:s}.env_corrs.{:s}'.format(base_name, file_name_end), topo_corrs)
    np.save('{:s}.env_rand_corrs.{:s}'.format(base_name, file_name_end), rand_topo_corrs)

def compute_env_class_corr(exp, trial_env):
    train_set = exp.dataset_provider.get_train_merged_valid_test(
        exp.dataset)['train']

    i_trial_starts, i_trial_ends = compute_trial_start_end_samples(
        train_set.y,
        check_trial_lengths_equal=True,
        input_time_length=exp.iterator.input_time_length)
    assert len(i_trial_ends) == trial_env.shape[1]
    # +1 as i_end is inclusive
    y_signal = [exp.dataset.train_set.y[i_start:i_end+1]
        for i_start, i_end in zip(i_trial_starts, i_trial_ends)]
    y_signal = np.array(y_signal).transpose(0,2,1)
    assert y_signal.shape[2] == trial_env.shape[3]
    topo_corrs = compute_topo_corrs(trial_env, y_signal)
    return topo_corrs

def compute_rand_preds_topo_corr(exp, rand_model, trial_env):
    pred_fn = create_pred_fn(rand_model)
    train_set = exp.dataset_provider.get_train_merged_valid_test(exp.dataset)['train']
    batches = [b[0] for b in exp.iterator.get_batches(train_set, shuffle=False)]
    all_batch_sizes = [len(b) for b in batches]
    
    all_preds = [pred_fn(b) for b in batches]
    preds_per_trial = compute_preds_per_trial(train_set.y, all_preds, all_batch_sizes,
        exp.iterator.input_time_length)
    preds_per_trial = np.array(preds_per_trial)
    # transpose from trialxsamplesxclasses to trialsxclassesxsamples
    # as compute topo corr expects like this
    rand_topo_corrs = compute_topo_corrs(trial_env, 
        preds_per_trial.transpose(0,2,1)) 
    return rand_topo_corrs

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
    

if __name__ == '__main__':
    setup_logging()
    start = None
    stop = None
    if len(sys.argv) > 1:
        start = int(sys.argv[1]) - 1 #1-based to 0-based
    if len(sys.argv) > 2:
        stop = int(sys.argv[2])
    folder = 'data/models-backup/paper/ours/cnt/deep4/car/'
    params = dict(sensor_names="$all_EEG_sensors", batch_modifier="null",
        low_cut_off_hz="null", first_nonlin="$elu")
    create_env_class_corrs(folder, params, start, stop)





