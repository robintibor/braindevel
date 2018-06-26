import os
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/braindecode/')
import logging
import time
from collections import OrderedDict
from copy import copy

import numpy as np

from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact, save_npy_artifact, save_torch_artifact
from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
from braindecode.torch_ext.util import confirm_gpu_availability
import torch.nn.functional as F
import torch as th
from torch import optim
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.experiment import Experiment
from braindecode.torch_ext.util import np_to_var
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or

from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor

from braindecode.datautil.splitters import split_into_two_sets
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import exponential_running_standardize


log = logging.getLogger(__name__)
log.setLevel('DEBUG')

def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': '/data/schirrmr/schirrmr/models/eegconvnet-public-data-simple-no-multiplication/',
    }]

    train_test_filenames = [{
        'train_filename': '/data/schirrmr/schirrmr/HGD-public/reduced/train/{:d}.mat'.format(i),
        'test_filename': '/data/schirrmr/schirrmr/HGD-public/reduced/test/{:d}.mat'.format(i),
    } for i in range(1,15)]




    preproc_params = dictlistprod({
        'low_cut_hz': [0,4]#0
    })

    stop_params = [{
        'max_epochs': 800,
        'max_increase_epochs': 80,
    }]


    model_params = dictlistprod({
        'model_name': ['deep', 'shallow']
    })

    seed_params = dictlistprod({
        'np_th_seed': [0,1,2,]#3,4
    })

    debug_params = [{
        'debug': False,
    }]


    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        train_test_filenames,
        preproc_params,
        model_params,
        stop_params,
        seed_params,
        debug_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def load_bbci_data(filename, low_cut_hz, debug=False):
    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']
    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)

    log.info("Loading data...")
    cnt = loader.load()

    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                  clean_ival)

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))

    # lets convert to millivolt for numerical stability of next operations
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    if debug:
        C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors)
    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)

    # Trial interval, start at -500 already, since improved decoding for networks
    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    return dataset


def load_train_valid_test(
        train_filename, test_filename, low_cut_hz, debug=False):
    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    log.info("Loading train...")
    full_train_set = load_bbci_data(
        train_filename, low_cut_hz=low_cut_hz, debug=debug)

    log.info("Loading test...")
    test_set = load_bbci_data(
        test_filename, low_cut_hz=low_cut_hz, debug=debug)
    valid_set_fraction = 0.8
    train_set, valid_set = split_into_two_sets(full_train_set,
                                               valid_set_fraction)

    log.info("Train set with {:4d} trials".format(len(train_set.X)))
    if valid_set is not None:
        log.info("Valid set with {:4d} trials".format(len(valid_set.X)))
    log.info("Test set with  {:4d} trials".format(len(test_set.X)))

    return train_set, valid_set, test_set


def run_exp_on_high_gamma_dataset(train_filename, test_filename,
                  low_cut_hz, model_name,
                  max_epochs, max_increase_epochs,
                  np_th_seed,
                  debug):
    train_set, valid_set, test_set = load_train_valid_test(
        train_filename=train_filename,
        test_filename=test_filename,
        low_cut_hz=low_cut_hz, debug=debug)
    if debug:
        max_epochs = 4

    set_random_seeds(np_th_seed, cuda=True)
    #torch.backends.cudnn.benchmark = True# sometimes crashes?
    n_classes = int(np.max(train_set.y) + 1)
    n_chans = int(train_set.X.shape[1])
    input_time_length = 1000
    if model_name == 'deep':
        model = Deep4Net(n_chans, n_classes,
                         input_time_length=input_time_length,
                         final_conv_length=2).create_network()
    elif model_name == 'shallow':
        model = ShallowFBCSPNet(
            n_chans, n_classes, input_time_length=input_time_length,
            final_conv_length=30).create_network()

    to_dense_prediction_model(model)
    model.cuda()
    model.eval()

    out = model(np_to_var(train_set.X[:1, :, :input_time_length, None]).cuda())

    n_preds_per_input = out.cpu().data.numpy().shape[2]
    optimizer = optim.Adam(model.parameters(), weight_decay=0,
                           lr=1e-3)

    iterator = CropsFromTrialsIterator(batch_size=60,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input,
                                       seed=np_th_seed)

    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(
                    input_time_length=input_time_length), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    loss_function = lambda preds, targets: F.nll_loss(th.mean(preds, dim=2),
                                                      targets)

    run_after_early_stop = True
    do_early_stop = True
    remember_best_column = 'valid_misclass'
    stop_criterion = Or([MaxEpochs(max_epochs),
                         NoDecrease('valid_misclass', max_increase_epochs)])

    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column=remember_best_column,
                     run_after_early_stop=run_after_early_stop, cuda=True,
                     do_early_stop=do_early_stop)
    exp.run()
    return exp



def run(
        ex, train_filename, test_filename,
        low_cut_hz, model_name,
        max_epochs, max_increase_epochs,
        np_th_seed,
        debug):
    kwargs = locals()
    kwargs.pop('ex')
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    start_time = time.time()
    ex.info['finished'] = False

    # check that gpu is available -> should lead to crash if gpu not there
    confirm_gpu_availability()

    exp = run_exp_on_high_gamma_dataset(**kwargs)
    end_time = time.time()
    last_row = exp.epochs_df.iloc[-1]
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    for key, val in last_row.iteritems():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
    save_pkl_artifact(ex, exp.epochs_df, 'epochs_df.pkl')
    save_pkl_artifact(ex, exp.before_stop_df, 'before_stop_df.pkl')
    if np_th_seed == 0:
        save_torch_artifact(ex, exp.model.state_dict(), "model_params.pkl")