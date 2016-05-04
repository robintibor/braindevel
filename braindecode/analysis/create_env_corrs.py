import numpy as np
from braindecode.results.results import ResultPool
from braindecode.experiments.load import load_exp_and_model
from braindecode.analysis.envelopes import load_trial_env, compute_topo_corrs
from braindecode.veganlasagne.layer_util import compute_trial_acts


def create_env_corrs(folder_name, params):
    res_pool = ResultPool()
    res_pool.load_results(folder_name, params=params)
    res_file_names = res_pool.result_file_names()[1:]
    all_base_names = [name.replace('.result.pkl', '')
        for name in res_file_names]
    for base_name in all_base_names:
        topo_corrs = create_topo_env_corrs(base_name)
        np.save(topo_corrs, base_name + 'env_corrs.npy')
    
def create_topo_env_corrs(base_name):
    exp, model = load_exp_and_model(base_name)
    exp.dataset.load()
    # Hackhack since i know this is correct layer atm
    i_layer = 26
    train_set = exp.dataset_provider.get_train_merged_valid_test(
        exp.dataset)['train']
    trial_env = load_trial_env(base_name, model, 
        i_layer, train_set, n_inputs_per_trial=2)
    trial_acts = compute_trial_acts(model, i_layer, exp.iterator, train_set)
    topo_corrs = compute_topo_corrs(trial_env, trial_acts)
    
    
if __name__ == "__main__":
    