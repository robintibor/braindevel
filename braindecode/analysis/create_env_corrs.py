import numpy as np
import os.path
import sys
from braindecode.results.results import ResultPool
from braindecode.experiments.load import load_exp_and_model
from braindecode.analysis.envelopes import load_trial_env, compute_topo_corrs
from braindecode.veganlasagne.layer_util import compute_trial_acts
from braindecode.datahandling.batch_iteration import compute_trial_start_end_samples
import logging
from braindecode.experiments.experiment import create_experiment
log = logging.getLogger(__name__)

def create_env_corrs(folder_name, params, start, stop):
    from braindecode.analysis.create_env_class_corrs import create_env_class_corr_file
    res_pool = ResultPool()
    res_pool.load_results(folder_name, params=params)
    res_file_names = res_pool.result_file_names()
    all_base_names = [name.replace('.result.pkl', '')
        for name in res_file_names]
    start = start or 0
    stop = stop or len(all_base_names) 
    # Hackhack hardcoded layers, since I know this is correct layers atm
    i_all_layers = [8,14,20,26,28] #for shallow [3, 4, 5, 7]
    for i_file, base_name in enumerate(all_base_names[start:stop]):
        with_square = True
        log.info("Running {:s} ({:d} of {:d})".format(
            base_name, i_file+start+1, stop))
        create_topo_env_corrs_files(base_name, i_all_layers, with_square)
        create_env_class_corr_file(base_name, with_square)
 
def create_topo_env_corrs_files(base_name, i_all_layers, with_square):
    # Load env first to make sure env is actually there.
    result = np.load(base_name + '.result.pkl')
    env_file_name = dataset_to_env_file(result.parameters['dataset_filename'])
    exp, model = load_exp_and_model(base_name)
    exp.dataset.load()
    train_set = exp.dataset_provider.get_train_merged_valid_test(
        exp.dataset)['train']
    rand_model = create_experiment(base_name + '.yaml').final_layer
    for i_layer in i_all_layers:
        log.info("Layer {:d}".format(i_layer))
        trial_env = load_trial_env(env_file_name, model, 
            i_layer, train_set, n_inputs_per_trial=2, square_before_mean=with_square)
        topo_corrs = compute_trial_topo_corrs(model, i_layer, train_set, 
            exp.iterator, trial_env, split_per_class=True)
        
        rand_topo_corrs = compute_trial_topo_corrs(rand_model, i_layer, train_set, 
            exp.iterator, trial_env, split_per_class=True)
        file_name_end = '{:d}.npy'.format(i_layer)
        if with_square:
            file_name_end = 'square.' + file_name_end
        np.save('{:s}.labelsplitted.env_corrs.{:s}'.format(base_name, file_name_end), topo_corrs)
        np.save('{:s}.labelsplitted.env_rand_corrs.{:s}'.format(base_name, file_name_end), rand_topo_corrs)
    return

def compute_trial_topo_corrs(model, i_layer, train_set, iterator, trial_env,
    split_per_class):
    """Now with split per class arg."""
    
    trial_acts = compute_trial_acts(model, i_layer, iterator, train_set)
    trial_labels = determine_trial_labels(train_set, iterator)
    if split_per_class:
        topo_corrs_per_class = []
        for i_label in xrange(trial_labels.shape[1]):
            this_trial_mask = trial_labels[:,i_label] == 1
            this_trial_env = trial_env[:, this_trial_mask]
            this_trial_acts = trial_acts[this_trial_mask]
            topo_corrs = compute_topo_corrs(this_trial_env, this_trial_acts)
            topo_corrs_per_class.append(topo_corrs)
        topo_corrs_per_class = np.array(topo_corrs_per_class)
        # TODO compute within trial corrs -> remove mean for each trial
        # compute between trial corrs -> compute means for trial env and trial acts
        # with keepdims=True
        return topo_corrs_per_class
    else:
        topo_corrs = compute_topo_corrs(trial_env, trial_acts)
        return topo_corrs
    
def determine_trial_labels(dataset, iterator):
    # determine trial labels
    i_trial_starts, _ =  compute_trial_start_end_samples(
                dataset.y, check_trial_lengths_equal=True,
                input_time_length=iterator.input_time_length)
    trial_labels = dataset.y[i_trial_starts]
    assert np.all(np.sum(trial_labels, axis=1) == 1)
    return trial_labels

def dataset_to_env_file(wanted_dataset_filename):
    """ For any dataset filename, give envelope filename
    These experiments are, where envelopes were calculated from originally"""
    res_pool= ResultPool()
    res_pool.load_results('data/models-backup/paper/ours/cnt/deep4/car/',
             params=dict(cnt_preprocessors="$cz_zero_resample_car_demean",
                 trial_start=1500, trial_stop=4000))

    dataset_to_env_file_name = dict()
    
    for result, res_file_name in zip(res_pool.result_objects(), res_pool.result_file_names()):
        dataset_file_name = result.parameters['dataset_filename']
        envelope_file_name = res_file_name.replace('.result.pkl', '.env.npy')
        assert os.path.isfile(envelope_file_name)
        dataset_to_env_file_name[dataset_file_name] = envelope_file_name
    return dataset_to_env_file_name[wanted_dataset_filename]
    
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
        start = int(sys.argv[1]) - 1 #1-based to 0-based
    if len(sys.argv) > 2:
        stop = int(sys.argv[2])
    create_env_corrs('data/models-backup/paper/ours/cnt/deep4/car/',
             params=dict(cnt_preprocessors="$cz_zero_resample_car_demean",
                 trial_start=1500, trial_stop=4000),
             start=start, stop=stop)
#    create_env_corrs('data/models-backup/paper/ours/cnt/shallow/car/',
#        params=None)
    
