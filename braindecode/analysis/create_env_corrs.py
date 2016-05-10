import numpy as np
import os.path
from braindecode.results.results import ResultPool
from braindecode.experiments.load import load_exp_and_model
from braindecode.analysis.envelopes import load_trial_env, compute_topo_corrs
from braindecode.veganlasagne.layer_util import compute_trial_acts
import logging
from braindecode.experiments.experiment import create_experiment
log = logging.getLogger(__name__)

def create_env_corrs(folder_name, params):
    res_pool = ResultPool()
    res_pool.load_results(folder_name, params=params)
    res_file_names = res_pool.result_file_names()
    all_base_names = [name.replace('.result.pkl', '')
        for name in res_file_names]
    for i_file, base_name in enumerate(all_base_names):
        assert os.path.isfile(base_name + ".env.npy") 
    for i_file, base_name in enumerate(all_base_names):
        log.info("Running {:s} ({:d} of {:d})".format(
            base_name, i_file+1, len(all_base_names)))
        topo_corrs, rand_topo_corrs = create_topo_env_corrs(base_name)
        np.save(base_name + '.env_corrs.npy', topo_corrs)
        np.save(base_name + '.env_rand_corrs.npy', rand_topo_corrs)
    
def create_topo_env_corrs(base_name):
    exp, model = load_exp_and_model(base_name)
    exp.dataset.load()
    # Hackhack since I know this is correct layer atm
    i_layer = 26
    train_set = exp.dataset_provider.get_train_merged_valid_test(
        exp.dataset)['train']
        
    result = np.load(base_name + '.result.pkl')
    env_file_name = dataset_to_env_file_dict()[result.parameters['dataset_filename']]
        
    trial_env = load_trial_env(env_file_name, model, 
        i_layer, train_set, n_inputs_per_trial=2)
    topo_corrs = compute_trial_topo_corrs(model, i_layer, train_set, 
        exp.iterator, trial_env)
    
    
    rand_model = create_experiment(base_name + '.yaml').final_layer
    rand_topo_corrs = compute_trial_topo_corrs(rand_model, i_layer, train_set, 
        exp.iterator, trial_env)
    
    return topo_corrs, rand_topo_corrs

def compute_trial_topo_corrs(model, i_layer, train_set, iterator, trial_env):
    trial_acts = compute_trial_acts(model, i_layer, iterator, train_set)
    topo_corrs = compute_topo_corrs(trial_env, trial_acts)
    return topo_corrs
    
def dataset_to_env_file_dict():
    """ FOr any dataset filename, give envelope filename
    These experiments are, where envelopes were calculated from originally"""
    res_pool= ResultPool()
    res_pool.load_results('data/models/paper/ours/cnt/deep4/car/',
                              params=dict(sensor_names="$all_EEG_sensors", batch_modifier="null",
                            low_cut_off_hz="null", first_nonlin="$elu"))

    dataset_to_env_file_name = dict()
    
    for result, res_file_name in zip(res_pool.result_objects(), res_pool.result_file_names()):
        
        dataset_file_name = result.parameters['dataset_filename']
        envelope_file_name = res_file_name.replace('.result.pkl', '.env.npy')
        print envelope_file_name
        assert os.path.isfile(envelope_file_name)
        dataset_to_env_file_name[dataset_file_name] = envelope_file_name
    return dataset_to_env_file_name
    
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
#     create_env_corrs('data/models-backup/paper/ours/cnt/deep4/car/',
#             params=dict(sensor_names="$all_EEG_sensors", batch_modifier="null",
#                         low_cut_off_hz="null", first_nonlin="$elu"))
    create_env_corrs('data/models/paper/ours/cnt/shallow/car/')
    
