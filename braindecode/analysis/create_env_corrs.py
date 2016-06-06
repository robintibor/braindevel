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
    
    # Hackhack hardcoded layers, since I know this is correct layers atm
    i_all_layers = [8,14,20,26,28] #for shallow [3, 4, 5, 7]
    for i_file, base_name in enumerate(all_base_names):
        log.info("Running {:s} ({:d} of {:d})".format(
            base_name, i_file+1, len(all_base_names)))
        create_topo_env_corrs_files(base_name, i_all_layers)
 
def create_topo_env_corrs_files(base_name, i_all_layers):
    # Load env first to make sure env is actually there.
    result = np.load(base_name + '.result.pkl')
    print base_name
    env_file_name = dataset_to_env_file(result.parameters['dataset_filename'])
    exp, model = load_exp_and_model(base_name)
    exp.dataset.load()
    train_set = exp.dataset_provider.get_train_merged_valid_test(
        exp.dataset)['train']
        
    for i_layer in i_all_layers:
        log.info("Layer {:d}".format(i_layer))
        trial_env = load_trial_env(env_file_name, model, 
            i_layer, train_set, n_inputs_per_trial=2)
        topo_corrs = compute_trial_topo_corrs(model, i_layer, train_set, 
            exp.iterator, trial_env)
        
        rand_model = create_experiment(base_name + '.yaml').final_layer
        rand_topo_corrs = compute_trial_topo_corrs(rand_model, i_layer, train_set, 
            exp.iterator, trial_env)
 
        np.save('{:s}.env_corrs.{:d}.npy'.format(base_name, i_layer), topo_corrs)
        np.save('{:s}.env_rand_corrs.{:d}.npy'.format(base_name, i_layer), rand_topo_corrs)
    return

def compute_trial_topo_corrs(model, i_layer, train_set, iterator, trial_env):
    trial_acts = compute_trial_acts(model, i_layer, iterator, train_set)
    topo_corrs = compute_topo_corrs(trial_env, trial_acts)
    return topo_corrs
    
def dataset_to_env_file(wanted_dataset_filename):
    """ For any dataset filename, give envelope filename
    These experiments are, where envelopes were calculated from originally"""
    res_pool= ResultPool()
    res_pool.load_results('data/models-backup/paper/ours/cnt/deep4/car/',
                              params=dict(sensor_names="$all_EEG_sensors", batch_modifier="null",
                            low_cut_off_hz="null", first_nonlin="$elu"))

    dataset_to_env_file_name = dict()
    
    for result, res_file_name in zip(res_pool.result_objects(), res_pool.result_file_names()):
        
        dataset_file_name = result.parameters['dataset_filename']
        envelope_file_name = res_file_name.replace('.result.pkl', '.env.npy')
        if not  os.path.isfile(envelope_file_name):
            log.warn("Env does not exist, {:s}".format(envelope_file_name))
            continue
        #assert os.path.isfile(envelope_file_name) TODOREENABLE
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
    create_env_corrs('data/models-backup/paper/ours/cnt/deep4/car/',
             params=dict(sensor_names="$all_EEG_sensors", batch_modifier="null",
                         low_cut_off_hz="null", first_nonlin="$elu"))
#    create_env_corrs('data/models-backup/paper/ours/cnt/shallow/car/',
#        params=None)
    
