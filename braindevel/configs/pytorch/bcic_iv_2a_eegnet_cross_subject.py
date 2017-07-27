import logging
import os.path
import time
from collections import OrderedDict

import numpy as np
import torch.nn.functional as F
from torch import optim

from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact

from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.models.eegnet import EEGNet
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.splitters import concatenate_sets
from braindecode.torch_ext.losses import l1_loss, l2_loss



log = logging.getLogger(__name__)


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': './data/models/pytorch/bcic-iv-2a/eegnet-cross-subject-to-mV-before/',
        'only_return_exp': False,
    }]
    subject_folder_params = dictlistprod({
        'subject_id': range(1, 10),
        'data_folder': ['/home/schirrmr/data/bci-competition-iv/2a-gdf/', ]
    })

    preproc_params = dictlistprod({
        'low_cut_hz': [4],
    })

    stop_params = dictlistprod({
        'max_epochs': [500],
    })

    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        subject_folder_params,
        preproc_params,
        stop_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def load_and_create_set(loader, low_cut_hz):
    # Preprocessing
    high_cut_hz = 40.0
    # lets convert to millvolt for numerical stability of next operations
    cnt = loader.load()

    cnt = cnt.drop_channels(['STI 014', 'EOG-left',
                             'EOG-central', 'EOG-right'])
    assert len(cnt.ch_names) == 22
    cnt = mne_apply(lambda a: a * 1e6, cnt)
    cnt = resample_cnt(cnt, 128.0)
    cnt = mne_apply(lambda a: bandpass_cnt(
        a, low_cut_hz, high_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
                    cnt)

    # Trial segementation
    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                                  ('Foot', [3]), ('Tongue', [4])])
    ival = [500, 2500]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    return dataset

def run_exp(data_folder, subject_id, low_cut_hz, cuda,
            max_epochs):
    test_subject_id = subject_id
    train_subject_ids = [sub_id for sub_id in range(1, 10) if
                         sub_id != test_subject_id]
    train_filenames = ['A{:02d}T.gdf'.format(sub_id) for sub_id in
                       train_subject_ids]
    valid_filename = 'A{:02d}T.gdf'.format(test_subject_id)
    test_filename = 'A{:02d}E.gdf'.format(test_subject_id)

    data_folder = 'data/bci-competition-iv/2a-gdf/'
    train_filepaths = [os.path.join(data_folder, fname) for fname in
                       train_filenames]
    valid_filepath = os.path.join(data_folder, valid_filename)
    test_filepath = os.path.join(data_folder, test_filename)
    train_loaders = [BCICompetition4Set2A(
        fpath,  labels_filename=fpath.replace('.gdf','.mat'))
        for fpath in train_filepaths]

    valid_loader = BCICompetition4Set2A(
        valid_filepath, labels_filename=valid_filepath.replace('.gdf', '.mat'))

    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_filepath.replace('.gdf', '.mat'))


    train_sets = [load_and_create_set(loader, low_cut_hz=low_cut_hz)
                  for loader in train_loaders]

    train_set = concatenate_sets(train_sets)

    valid_set = load_and_create_set(valid_loader, low_cut_hz=low_cut_hz)
    test_set = load_and_create_set(test_loader, low_cut_hz=low_cut_hz)

    set_random_seeds(seed=20190706, cuda=cuda)

    n_classes = 4
    n_chans = int(train_set.X.shape[1])
    input_time_length = train_set.X.shape[2]
    model = EEGNet(n_chans, n_classes,
                   input_time_length=input_time_length, ).create_network()

    if cuda:
        model.cuda()
    log.info("Model: \n{:s}".format(str(model)))

    optimizer = optim.Adam(model.parameters())

    def elastic_net_loss(model):
        return l2_loss(model) * 1e-4 + l1_loss(model) * 1e-4

    iterator = BalancedBatchSizeIterator(batch_size=16) # todo should be 16

    stop_criterion = MaxEpochs(max_epochs)

    monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]

    model_constraint = None

    model_loss_function = elastic_net_loss

    # todo not clear if final model or best from validation loss
    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                 loss_function=F.nll_loss, optimizer=optimizer,
                 model_constraint=model_constraint,
                 monitors=monitors,
                 stop_criterion=stop_criterion,
                 remember_best_column='valid_loss',# todo valid loss
                 run_after_early_stop=False, cuda=cuda,
                model_loss_function=model_loss_function)

    exp.run()
    return exp


def run(ex, data_folder, subject_id, low_cut_hz, only_return_exp, max_epochs):
    cuda = True
    start_time = time.time()
    assert only_return_exp is False
    assert (only_return_exp is False) or (n_chans is not None)
    ex.info['finished'] = False


    exp = run_exp(data_folder, subject_id, low_cut_hz, cuda,
                  max_epochs=max_epochs)
    last_row = exp.epochs_df.iloc[-1]
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    for key, val in last_row.iteritems():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
    save_pkl_artifact(ex, exp.epochs_df, 'epochs_df.pkl')
    save_pkl_artifact(ex, exp.before_stop_df, 'before_stop_df.pkl')

