import numpy as np
import logging
from braindecode.experiments.load import load_exp_and_model
from braindecode.analysis.envelopes import load_trial_env, compute_topo_corrs
from braindecode.veganlasagne.layer_util import compute_trial_acts
from braindecode.analysis.create_env_corrs import setup_logging
log = logging.getLogger(__name__)

def check_shallow_env_corrs():
    i_layer = 7
    for i_exp in xrange(20):
        file_base_name = 'data/models/paper/ours/cnt/shallow//car/{:d}'.format(
            i_exp)
        result = np.load(file_base_name + '.result.pkl')
        log.info("Running {:d} of {:d}:\n{:s}".format(i_exp, 20,
            file_base_name))
        exp_shallow, model_shallow = load_exp_and_model(file_base_name)
        exp_shallow.dataset.load()
        env_filename = dataset_to_env_file(result.parameters['dataset_filename'])
        trial_env = load_trial_env(env_filename, model_shallow,
                            i_layer, exp_shallow.dataset.sets[0],
                            n_inputs_per_trial=2)
        trial_acts = compute_trial_acts(model_shallow, i_layer,
            exp_shallow.iterator, exp_shallow.dataset.train_set)
        topo_corrs = compute_topo_corrs(trial_env, trial_acts)
        topo_corrs_old = np.load(file_base_name  + '.env_corrs.npy')
        diff = np.mean(np.abs(topo_corrs - topo_corrs_old))
        assert diff < 1e-3, (
            "Diff too big for {:s}: {:f}".format(file_base_name, diff))
        

def dataset_to_env_file(wanted_dataset_filename):
    dataset_to_exp_nr = {
       'BhNo': 47,
       'LuFi': 48,
       'AnWe': 58,
       'FaMa': 59,
       'FrTh': 60,
       'GuJo': 61,
       'JoBe': 62,
       'KaUs': 63,
       'LaKa': 64,
       'MaGl': 65,
       'MaJa': 66,
       'MaVo': 67,
       'MaKi': 68,
       'PiWi': 69,
       'NaMa': 70,
       'RoBe': 71,
       'RoSc': 72,
       'StHe': 73,
       'SvMu': 74,
       'OlIl': 75,
    }
    
    for file_part, i_file in dataset_to_exp_nr.iteritems():
        if file_part in wanted_dataset_filename:
            return 'data/models/paper/ours/cnt/deep4/car/{:d}.env.npy'.format(
                i_file)
    raise ValueError("No envelope found for {:s}".format(wanted_dataset_filename))


if __name__ == "__main__":
    setup_logging()
    check_shallow_env_corrs()