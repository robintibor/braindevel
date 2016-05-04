import numpy as np
import os.path
from braindecode.results.results import ResultPool
from braindecode.experiments.load import load_exp_and_model
from braindecode.analysis.envelopes import load_trial_env, compute_topo_corrs
from braindecode.veganlasagne.layer_util import compute_trial_acts
import logging
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
        topo_corrs = create_topo_env_corrs(base_name)
        np.save(base_name + '.env_corrs.npy', topo_corrs)
    
def create_topo_env_corrs(base_name):
    exp, model = load_exp_and_model(base_name)
    exp.dataset.load()
    # Hackhack since I know this is correct layer atm
    i_layer = 26
    train_set = exp.dataset_provider.get_train_merged_valid_test(
        exp.dataset)['train']
    trial_env = load_trial_env(base_name, model, 
        i_layer, train_set, n_inputs_per_trial=2)
    trial_acts = compute_trial_acts(model, i_layer, exp.iterator, train_set)
    topo_corrs = compute_topo_corrs(trial_env, trial_acts)
    return topo_corrs
    
if __name__ == "__main__":
    logging.basicConfig(level='DEBUG')
    create_env_corrs('data/models-backup/paper/ours/cnt/deep4/car/',
            params=dict(sensor_names="$all_EEG_sensors", batch_modifier="null",
                        low_cut_off_hz="null"))
    
