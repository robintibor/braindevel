import os
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/braindecode/')
os.sys.path.append('/home/schirrmr/braindecode/code/arl-eegmodels/')
import logging
import time
from collections import OrderedDict
from copy import copy
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow
import keras
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from keras.utils import to_categorical

from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact, save_npy_artifact, save_torch_artifact
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.signalproc import highpass_cnt, bandpass_cnt
from braindecode.torch_ext.util import confirm_gpu_availability
import torch.nn.functional as F
import torch as th
from torch import optim
from braindecode.torch_ext.util import set_random_seeds

from braindecode.datautil.iterators import get_balanced_batches
from braindecode.datautil.splitters import select_examples
from braindecode.datautil.splitters import concatenate_sets
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import exponential_running_standardize


log = logging.getLogger(__name__)
log.setLevel('DEBUG')

def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': '/data/schirrmr/schirrmr/models/eegnet-comparison/bcic-iv-2a-tf-128-hz/',
    }]

    data_params = dictlistprod({
        'subject_id': list(range(1,10)),
        'i_test_fold': list(range(3)),
    })

    preproc_params  = dictlistprod({
        'resample_fs': [128],
    })

    model_params = dictlistprod({
        'modelname': ['EEGNet-8'],
    })

    seed_params = dictlistprod({
        'np_th_seed': list(range(1)),
    })

    debug_params = [{
        'debug': False,
    }]

    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        data_params,
        preproc_params,
        model_params,
        seed_params,
        debug_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def load_data(data_folder, subject_id, low_cut_hz, resample_fs):
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

    # Preprocessing

    train_cnt = train_cnt.drop_channels(['STI 014', 'EOG-left',
                                         'EOG-central', 'EOG-right'])
    assert len(train_cnt.ch_names) == 22
    # lets convert to millvolt for numerical stability of next operations
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    train_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, 40, train_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), train_cnt)
    train_cnt = resample_cnt(train_cnt, resample_fs)
    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        train_cnt)

    test_cnt = test_cnt.drop_channels(['STI 014', 'EOG-left',
                                       'EOG-central', 'EOG-right'])
    assert len(test_cnt.ch_names) == 22
    test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
    test_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, 40, test_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt)
    test_cnt = resample_cnt(test_cnt, resample_fs)
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        test_cnt)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])
    ival = [500, 2500]

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

    train_set, valid_set = split_into_two_sets(train_set,
                                               first_set_fraction=0.8)
    return train_set, valid_set, test_set




def resplit_as_cross_validation(train_set, valid_set, test_set, i_test_fold):
    full_set = concatenate_sets((train_set, valid_set, test_set))
    n_folds = 4
    fold_inds = get_balanced_batches(
        len(full_set.X), None, shuffle=False, n_batches=n_folds)
    fold_sets = [select_examples(full_set, inds) for inds in
                 fold_inds]
    test_set = fold_sets[i_test_fold]
    train_folds = np.arange(n_folds)
    train_folds = np.delete(train_folds, np.where(train_folds == i_test_fold))
    i_valid_fold = (i_test_fold - 1) % n_folds
    train_folds = np.delete(train_folds, np.where(train_folds == i_valid_fold))
    valid_set = fold_sets[i_valid_fold]
    assert i_valid_fold not in train_folds
    assert i_test_fold != i_valid_fold
    assert i_test_fold not in train_folds

    train_fold_sets = [fold_sets[i] for i in train_folds]
    train_set = concatenate_sets(train_fold_sets)
    # Some checks
    assert len(train_set.X) + len(valid_set.X) + len(test_set.X) == len(
        full_set.X)
    return train_set, valid_set, test_set


def resplit_as_cross_validation_valid_on_train(
        train_set, valid_set, test_set, i_test_fold):
    i_valid_fold = i_test_fold # ! test always same in this split
    full_train_set = concatenate_sets((train_set, valid_set))
    n_folds = 3
    fold_inds = get_balanced_batches(
        len(full_train_set.X), None, shuffle=False, n_batches=n_folds)
    fold_sets = [select_examples(full_train_set, inds) for inds in
                 fold_inds]
    valid_set = fold_sets[i_valid_fold]
    train_folds = np.arange(n_folds)
    train_folds = np.delete(train_folds, np.where(train_folds == i_valid_fold))
    train_fold_sets = [fold_sets[i] for i in train_folds]
    train_set = concatenate_sets(train_fold_sets)
    # Some checks
    assert len(train_set.X) + len(valid_set.X) == len(
        full_train_set.X)
    # test set remains the same! always same, the original test set!
    return train_set, valid_set, test_set


def run_exp_on_split(train_set, valid_set, n_epochs, modelname):
    n_classes = len(np.unique(train_set.y))
    n_chans = train_set.X.shape[1]
    if modelname == 'EEGNet-4':
        model = EEGNet(
            nb_classes=n_classes, Chans=n_chans, Samples=train_set.X.shape[2],
            F1=4, D=2, F2=8,)
    elif modelname == 'EEGNet-8':
        model = EEGNet(
            nb_classes=n_classes, Chans=n_chans, Samples=train_set.X.shape[2],
            F1=8, D=2, F2=16,)
    elif modelname == 'Shallow':
        model = ShallowConvNet(
            nb_classes=n_classes, Chans=n_chans, Samples=train_set.X.shape[2])
    elif modelname == 'Deep':
        model = DeepConvNet(
            nb_classes=n_classes, Chans=n_chans, Samples=train_set.X.shape[2])




    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['acc'])
    best_val_loss = np.inf
    epochs_df = pd.DataFrame()
    for i_epoch in range(n_epochs):
        fittedModel = model.fit(train_set.X[:, None],
                                to_categorical(train_set.y), verbose=True,
                                epochs=1,
                                batch_size=64,
                                validation_data=(valid_set.X[:, None],
                                                 to_categorical(valid_set.y)))
        val_loss = fittedModel.history['val_loss'][0]
        if val_loss < best_val_loss:
            best_weights = deepcopy(model.get_weights())
            best_val_loss = val_loss
            print("best val loss", best_val_loss)
        history_dict = dict(
            [(key, val[0]) for key, val in fittedModel.history.items()])
        epochs_df = epochs_df.append(history_dict, ignore_index=True)
    return epochs_df, model, best_weights


def run_exp(subject_id, i_test_fold, resample_fs, modelname, debug):
    n_epochs = 500
    if debug is True:
        n_epochs = 5
    data_folder = '/home/schirrmr/data/bci-competition-iv/2a-gdf/'
    low_cut_hz = 4
    train_set, valid_set, test_set = load_data(data_folder, subject_id,
                                               low_cut_hz,
                                               resample_fs=resample_fs)
    train_set, valid_set, test_set = resplit_as_cross_validation(
        train_set, valid_set, test_set, i_test_fold)
    epochs_df, model, best_weights = run_exp_on_split(
        train_set, valid_set, n_epochs, modelname)
    final_test_loss, final_test_acc = model.evaluate(
        test_set.X[:, None], to_categorical(test_set.y))
    model.set_weights(best_weights)
    test_loss, test_acc = model.evaluate(
        test_set.X[:, None], to_categorical(test_set.y))
    return epochs_df, test_loss, test_acc, final_test_acc, final_test_loss


def run(
        ex, subject_id, i_test_fold, resample_fs, modelname,
        np_th_seed,
        debug,):
    kwargs = locals()
    kwargs.pop('ex')
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    start_time = time.time()
    ex.info['finished'] = False

    # check that gpu is available -> should lead to crash if gpu not there
    confirm_gpu_availability()
    set_random_seeds(np_th_seed, cuda=True)
    epochs_df, test_loss, test_acc, final_test_acc, final_test_loss = run_exp(
        subject_id, i_test_fold, resample_fs, modelname, debug)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    ex.info['test_loss'] = float(test_loss)
    ex.info['test_acc'] = float(test_acc)
    ex.info['final_test_loss'] = float(final_test_loss)
    ex.info['final_test_acc'] = float(final_test_acc)
    last_row = epochs_df.iloc[-1]
    for key, val in last_row.iteritems():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
    save_pkl_artifact(ex, epochs_df, 'epochs_df.pkl')
