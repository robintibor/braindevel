import lasagne
import logging
import sys
import numpy as np
from braindecode.analysis.stats import wrap_reshape_topo, corr
from braindecode.experiments.experiment import create_experiment
from braindecode.experiments.load import load_exp_and_model
from braindecode.results.results import ResultPool
from braindecode.veganlasagne.layer_util import compute_trial_acts
from braindecode.veganlasagne.layers import get_n_sample_preds
from braindecode.analysis.create_amplitude_perturbation_corrs import (
    get_trials_targets)
log = logging.getLogger(__name__)

def create_unit_output_class_corrs_for_files(folder_name, params,
        start, stop, i_all_layers):
    res_pool = ResultPool()
    res_pool.load_results(folder_name, params=params)
    res_file_names = res_pool.result_file_names()
    all_base_names = [name.replace('.result.pkl', '')
        for name in res_file_names]
    start = start or 0
    stop = stop or len(all_base_names)
    for i_file, basename in enumerate(all_base_names[start:stop]):
        log.info("Running {:s} ({:d} of {:d})".format(
            basename, i_file + start + 1, stop))
        create_unit_output_class_corrs(basename, i_all_layers)

def create_unit_output_class_corrs(basename, i_all_layers):
    exp, model = load_exp_and_model(basename)
    exp.dataset.load()
    train_set = exp.dataset_provider.get_train_merged_valid_test(
        exp.dataset)['train']
    rand_model = create_experiment(basename + '.yaml').final_layer
    for i_layer in i_all_layers:
        trained_corrs = unit_output_class_corrs(model, exp.iterator,
            train_set, i_layer)
        untrained_corrs = unit_output_class_corrs(rand_model, exp.iterator,
            train_set, i_layer)
        file_name_end = '{:d}.npy'.format(i_layer)
        trained_filename = '{:s}.unit_class_corrs.{:s}'.format(basename,
            file_name_end)
        untrained_filename = '{:s}.rand_unit_class_corrs.{:s}'.format(basename,
            file_name_end)
        log.info("Saving to {:s} and {:s}".format(trained_filename,
            untrained_filename))
        np.save(trained_filename, trained_corrs)
        np.save(untrained_filename, untrained_corrs)

def unit_output_class_corrs(model, iterator, train_set, i_layer):
    # only need targets, ignore trials
    # always get targets as from final layer
    this_final_layer = lasagne.layers.get_all_layers(model)[-1]
    _, targets = get_trials_targets(train_set,
        get_n_sample_preds(this_final_layer), iterator)
    trial_acts = compute_trial_acts(this_final_layer, i_layer, iterator,
        train_set)
    # only take those targets where we have predictions for
    # a bit hacky: we know targets are same for each trial, so we just 
    # take last ones, eventhough they come form overlapping batches
    # targets are #trials x #samples x #classes
    unmeaned_targets = targets - np.mean(targets, axis=(1), keepdims=True)
    assert np.all(unmeaned_targets == 0), ("Each trial should only have one "
        "unique label")
    relevant_targets = targets[:,:trial_acts.shape[2]]
    unit_class_corrs = wrap_reshape_topo(corr, trial_acts, relevant_targets,
                      axis_a=(0,2), axis_b=(0,1))
    return unit_class_corrs

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
    i_all_layers = [8,14,20,26,28]
    create_unit_output_class_corrs_for_files(folder,
             params=params, start=start,stop=stop, i_all_layers=i_all_layers)
