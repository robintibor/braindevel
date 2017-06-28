import logging
import numpy as np
import os.path
import time
from collections import OrderedDict

import torch as th
from torch import optim

from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact

from braindecode2.datasets.loaders import BCICompetition4Set2A
from braindecode2.datasets.signal_target import SignalAndTarget
from braindecode2.experiments.experiment import Experiment
from braindecode2.torchext.objectives import log_categorical_crossentropy
from braindecode2.trial_segment import create_target_series
from braindecode2.experiments.monitors import (LossMonitor, MisclassMonitor,
                                               RuntimeMonitor,
                                               CntTrialMisclassMonitor)
from braindecode2.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode2.iterators import CntWindowTrialIterator
from braindecode2.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode2.modules.expression import Expression
from braindecode2.mywyrm.processing import exponential_standardize_cnt, \
    bandpass_cnt
from braindecode2.splitters import split_into_two_sets
from braindecode2.torchext.constraints import MaxNormDefaultConstraint
from braindecode2.torchext.util import (set_random_seeds,to_net_in_output,
    convert_to_dense_prediction_model)

log = logging.getLogger(__name__)


def get_templates():
    return  {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{ 
        'save_folder': './data/models/pytorch/bcic-iv-2a/shallow/cnt-eps-fixed/',
        'only_return_exp': False,
        'n_chans': 22
        }]
    subject_folder_params = dictlistprod({
        'subject_id': range(1,10),
        'data_folder': ['/home/schirrmr/data/bci-competition-iv/2a-gdf/',]
    })

    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        subject_folder_params,
        ])
    
    return grid_params

def sample_config_params(rng, params):
    return params
        
def run(ex, data_folder, subject_id, n_chans,
        only_return_exp,):
    start_time = time.time()
    assert only_return_exp is False
    assert (only_return_exp is False) or (n_chans is not None) 
    ex.info['finished'] = False
    train_filename = 'A{:02d}T.gdf'.format(subject_id)
    test_filename = 'A{:02d}E.gdf'.format(subject_id)
    train_filepath = os.path.join(data_folder, train_filename)
    test_filepath = os.path.join(data_folder, test_filename)
    train_label_filepath = train_filepath.replace('.gdf', '.mat')
    test_label_filepath = test_filepath.replace('.gdf', '.mat')

    
    train_loader = BCICompetition4Set2A(
        train_filepath, labels_filename=train_label_filepath)
    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_label_filepath)
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()

    train_cnt = bandpass_cnt(train_cnt, 0, 38, filt_order=3)
    test_cnt = bandpass_cnt(test_cnt, 0, 38, filt_order=3)
    train_cnt = exponential_standardize_cnt(train_cnt, factor_new=1e-3,
                                            init_block_size=1000, eps=1e-4)
    test_cnt = exponential_standardize_cnt(test_cnt, factor_new=1e-3,
                                           init_block_size=1000, eps=1e-4)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])

    ival = [1500, 4000]



    train_y = create_target_series(train_cnt, marker_def, ival)
    train_set = SignalAndTarget(train_cnt.data.astype(np.float32),
                                y=train_y.astype(np.int64))
    test_y = create_target_series(test_cnt, marker_def, ival)
    test_set = SignalAndTarget(test_cnt.data.astype(np.float32),
                               y=test_y.astype(np.int64))
    train_set, valid_set = split_into_two_sets(train_set,
                                               first_set_fraction=0.8)


    set_random_seeds(seed=3984734, cuda=True)

    n_classes = train_set.y.shape[1]
    n_chans = train_set.X.shape[1]
    # final_dense_length=69 for full trial length at -500,4000
    input_time_length = 1000
    model = ShallowFBCSPNet(n_chans, n_classes,
                            input_time_length=input_time_length,
                            final_conv_length=30).create_network()
    model.cuda()
    convert_to_dense_prediction_model(model)

    model.add_module('squeeze', Expression(lambda x: x.squeeze()))

    out = model(to_net_in_output(
        np.ones((2, n_chans, input_time_length, 1), dtype=np.float32)).cuda())
    n_preds_per_input = out.cpu().data.numpy().shape[2]

    iterator = CntWindowTrialIterator(batch_size=60,
                                      input_time_length=input_time_length,
                                      n_preds_per_input=n_preds_per_input)

    eps = 1e-8
    # categorical cross entropy plus tied neighbour loss
    loss_function = lambda p, t: th.mean(
        log_categorical_crossentropy(p, t.float())) + th.mean(
        log_categorical_crossentropy(
            th.clamp(p[:, :, 1:], min=np.log(eps), max=np.log(1 - eps)),
            th.exp(p[:, :, :-1])))
    optimizer = optim.Adam(model.parameters())

    stop_criterion = Or([MaxEpochs(800),
                         NoDecrease('valid_misclass', 80)])
    monitors = [LossMonitor(), MisclassMonitor(exponentiate_preds=True,col_suffix='sample_misclass'),
                CntTrialMisclassMonitor(input_time_length=input_time_length), RuntimeMonitor()]
    model_constraint = MaxNormDefaultConstraint()

    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, cuda=True)
    exp.run()

    end_time = time.time()
    run_time = end_time - start_time
    
    ex.info['finished'] = True

    last_row = exp.epochs_df.iloc[-1]

    for key, val in last_row.iteritems():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
    save_pkl_artifact(ex, exp.epochs_df, 'epochs_df.pkl')
    save_pkl_artifact(ex, exp.before_stop_df, 'before_stop_df.pkl')

