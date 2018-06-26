import os
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/braindecode/')
os.sys.path.insert(0, '/home/schirrmr/code/adamw-evaluation/')
import logging
import time
from collections import OrderedDict
from copy import copy

import numpy as np
from numpy.random import RandomState

from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact, save_npy_artifact
from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
from braindecode.torch_ext.util import confirm_gpu_availability
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
import numpy as np
import torch.nn.functional as F
import torch as th
from torch import optim
import torch.backends.cudnn
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.experiment import Experiment
from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or

from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor

from adamweegeval.optimizers import AdamW
from adamweegeval.schedulers import (ScheduledOptimizer, CosineAnnealing,
                                     CutCosineAnnealing)
from adamweegeval.resnet import EEGResNet
import logging
import os
from collections import OrderedDict

import numpy as np
from braindecode.datautil.splitters import select_examples, concatenate_sets, split_into_two_sets
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
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
        'save_folder': '/data/schirrmr/schirrmr/models/adameegeval/eegconvnet-public-data-independent-2/',
    }]

    train_test_filenames = [{
        'train_filename': '/data/schirrmr/schirrmr/HGD-public/reduced/train/{:d}.mat'.format(i),
        'test_filename': '/data/schirrmr/schirrmr/HGD-public/reduced/test/{:d}.mat'.format(i),
    } for i in range(1,15)]

    data_split_params = [{
        'n_folds': None,
        'i_test_fold': None,
        'valid_set_fraction': 0.8,
        'use_validation_set': True,
        'test_on_eval_set': True,
    }]



    preproc_params = dictlistprod({
        'low_cut_hz': [4]#0
    })

    stop_params = [{
        'max_epochs': 800,
        'max_increase_epochs': 80,
    }]

    optim_params = [{
        'optimizer_name': 'adam',
        'scheduler_name': None,
        'use_norm_constraint': True,
        'weight_decay': 0,
        'init_lr': 1e-3,
        'schedule_weight_decay': False,
        'restarts': None,
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
        data_split_params,
        preproc_params,
        model_params,
        optim_params,
        stop_params,
        seed_params,
        debug_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    if params['test_on_eval_set'] == False:
        params['test_filename'] = None
    params.pop('test_on_eval_set')
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
    cnt = mne_apply(lambda a: a * 1e6, cnt)
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


    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    return dataset


def load_train_valid_test(train_filename, test_filename, n_folds, i_test_fold,
                          valid_set_fraction,
                          use_validation_set, low_cut_hz, debug=False):
    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    if test_filename is None:
        assert n_folds is not None
        assert i_test_fold is not None
        assert valid_set_fraction is None
    else:
        assert n_folds is None
        assert i_test_fold is None
        assert use_validation_set == (valid_set_fraction is not None)

    train_folder = '/home/schirrmr/data/BBCI-without-last-runs/'
    log.info("Loading train...")
    full_train_set = load_bbci_data(os.path.join(train_folder, train_filename),
                                    low_cut_hz=low_cut_hz, debug=debug)

    if test_filename is not None:
        test_folder = '/home/schirrmr/data/BBCI-only-last-runs/'
        log.info("Loading test...")
        test_set = load_bbci_data(os.path.join(test_folder, test_filename),
                                  low_cut_hz=low_cut_hz, debug=debug)
        if use_validation_set:
            assert valid_set_fraction is not None
            train_set, valid_set = split_into_two_sets(full_train_set,
                                                       valid_set_fraction)
        else:
            train_set = full_train_set
            valid_set = None

    # Split data
    if n_folds is not None:
        fold_inds = get_balanced_batches(
            len(full_train_set.X), None, shuffle=False, n_batches=n_folds)

        fold_sets = [select_examples(full_train_set, inds) for inds in
                     fold_inds]

        test_set = fold_sets[i_test_fold]
        train_folds = np.arange(n_folds)
        train_folds = np.setdiff1d(train_folds, [i_test_fold])
        if use_validation_set:
            i_valid_fold = (i_test_fold - 1) % n_folds
            train_folds = np.setdiff1d(train_folds, [i_valid_fold])
            valid_set = fold_sets[i_valid_fold]
            assert i_valid_fold not in train_folds
            assert i_test_fold != i_valid_fold
        else:
            valid_set = None

        assert i_test_fold not in train_folds

        train_fold_sets = [fold_sets[i] for i in train_folds]
        train_set = concatenate_sets(train_fold_sets)
        # Some checks
        if valid_set is None:
            assert len(train_set.X) + len(test_set.X) == len(full_train_set.X)
        else:
            assert len(train_set.X) + len(valid_set.X) + len(test_set.X) == len(
                full_train_set.X)

    log.info("Train set with {:4d} trials".format(len(train_set.X)))
    if valid_set is not None:
        log.info("Valid set with {:4d} trials".format(len(valid_set.X)))
    log.info("Test set with  {:4d} trials".format(len(test_set.X)))

    return train_set, valid_set, test_set

def run_4_sec_exp(train_filename, test_filename, n_folds,
                  i_test_fold, valid_set_fraction, use_validation_set,
                  low_cut_hz, model_name, optimizer_name, init_lr,
                  scheduler_name, use_norm_constraint,
                    restarts,
                  weight_decay, schedule_weight_decay,
                  max_epochs, max_increase_epochs,
                  np_th_seed,
                  debug):
    train_set, valid_set, test_set = load_train_valid_test(
        train_filename=train_filename,
        test_filename=test_filename,
        n_folds=n_folds,
        i_test_fold=i_test_fold, valid_set_fraction=valid_set_fraction,
        use_validation_set=use_validation_set,
        low_cut_hz=low_cut_hz, debug=debug)
    if debug:
        if restarts is None:
            max_epochs = 4
        else:
            assert max_epochs is None
            restarts = [1,3,5]

    return run_experiment(
        train_set, valid_set, test_set,
        model_name, optimizer_name,
        init_lr=init_lr,
        scheduler_name=scheduler_name,
        use_norm_constraint=use_norm_constraint,
        weight_decay=weight_decay,
        schedule_weight_decay=schedule_weight_decay,
        restarts=restarts,
        max_epochs=max_epochs,
        max_increase_epochs=max_increase_epochs,
        np_th_seed=np_th_seed, )

def run_experiment(
        train_set, valid_set, test_set, model_name, optimizer_name,
        init_lr,
        scheduler_name,
        use_norm_constraint, weight_decay,
        schedule_weight_decay,
        restarts,
        max_epochs,
        max_increase_epochs,
        np_th_seed):
    set_random_seeds(np_th_seed, cuda=True)
    #torch.backends.cudnn.benchmark = True# sometimes crashes?
    if valid_set is not None:
        assert max_increase_epochs is not None
    assert (max_epochs is None) != (restarts is None)
    if max_epochs is None:
        max_epochs = np.sum(restarts)
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
    elif model_name in ['resnet-he-uniform', 'resnet-he-normal',
                        'resnet-xavier-normal', 'resnet-xavier-uniform']:
        init_name = model_name.lstrip('resnet-')
        from torch.nn import init
        init_fn = {'he-uniform': lambda w: init.kaiming_uniform(w, a=0),
                   'he-normal': lambda w: init.kaiming_normal(w, a=0),
                   'xavier-uniform': lambda w: init.xavier_uniform(w, gain=1),
                   'xavier-normal': lambda w: init.xavier_normal(w, gain=1)}[init_name]
        model = EEGResNet(in_chans=n_chans, n_classes=n_classes,
                          input_time_length=input_time_length,
                          final_pool_length=10, n_first_filters=48,
                          conv_weight_init_fn=init_fn).create_network()
    else:
        raise ValueError("Unknown model name {:s}".format(model_name))
    if 'resnet' not in model_name:
        to_dense_prediction_model(model)
    model.cuda()
    model.eval()

    out = model(np_to_var(train_set.X[:1, :, :input_time_length, None]).cuda())

    n_preds_per_input = out.cpu().data.numpy().shape[2]


    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay,
                               lr=init_lr)
    elif optimizer_name == 'adamw':
        optimizer = AdamW(model.parameters(), weight_decay=weight_decay,
                          lr=init_lr)

    iterator = CropsFromTrialsIterator(batch_size=60,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input,
                                       seed=np_th_seed)

    if scheduler_name is not None:
        assert schedule_weight_decay == (optimizer_name == 'adamw')
        if scheduler_name == 'cosine':
            n_updates_per_epoch = sum(
                [1 for _ in iterator.get_batches(train_set, shuffle=True)])
            if restarts is None:
                n_updates_per_period = n_updates_per_epoch * max_epochs
            else:
                n_updates_per_period = np.array(restarts) * n_updates_per_epoch
            scheduler = CosineAnnealing(n_updates_per_period)
            optimizer = ScheduledOptimizer(scheduler, optimizer,
                                           schedule_weight_decay=schedule_weight_decay)
        elif scheduler_name == 'cut_cosine':
            # TODO: integrate with if clause before, now just separate
            # to avoid messing with code
            n_updates_per_epoch = sum(
                [1 for _ in iterator.get_batches(train_set, shuffle=True)])
            if restarts is None:
                n_updates_per_period = n_updates_per_epoch * max_epochs
            else:
                n_updates_per_period = np.array(restarts) * n_updates_per_epoch
            scheduler = CutCosineAnnealing(n_updates_per_period)
            optimizer = ScheduledOptimizer(scheduler, optimizer,
                                           schedule_weight_decay=schedule_weight_decay)
        else:
            raise ValueError("Unknown scheduler")
    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(
                    input_time_length=input_time_length), RuntimeMonitor()]

    if use_norm_constraint:
        model_constraint = MaxNormDefaultConstraint()
    else:
        model_constraint = None
    # change here this cell
    loss_function = lambda preds, targets: F.nll_loss(th.mean(preds, dim=2),
                                                      targets)

    if valid_set is not None:
        run_after_early_stop = True
        do_early_stop = True
        remember_best_column = 'valid_misclass'
        stop_criterion = Or([MaxEpochs(max_epochs),
                             NoDecrease('valid_misclass', max_increase_epochs)])
    else:
        run_after_early_stop = False
        do_early_stop = False
        remember_best_column = None
        stop_criterion = MaxEpochs(max_epochs)

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
        ex, train_filename, test_filename, n_folds,
        i_test_fold, valid_set_fraction, use_validation_set,
        low_cut_hz, model_name, optimizer_name,
        scheduler_name, use_norm_constraint,
        weight_decay, max_epochs, max_increase_epochs,
        restarts, schedule_weight_decay, init_lr,
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

    exp = run_4_sec_exp(**kwargs)
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
