import logging
import os.path
import time
from collections import OrderedDict
import sys

import numpy as np
import torch.nn.functional as F
from torch import optim
import torch as th

from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact

from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

log = logging.getLogger(__name__)
subject_to_filenames = {
    'anwe':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/AnWeMoSc1S001R01_ds10_1-12.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/AnWeMoSc1S001R13_ds10_1-2BBCI.mat',
        },
    'bhno':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/BhNoMoSc1S001R01_ds10_1-12.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/BhNoMoSc1S001R13_ds10_1-2BBCI.mat',
        },
    'famo':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/FaMaMoSc1S001R01_ds10_1-14.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/FaMaMoSc1S001R15_ds10_1-2BBCI.mat',
        },
    'frth':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/FrThMoSc1S001R01_ds10_1-11.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/FrThMoSc1S001R12_ds10_1-2BBCI.mat',
        },
    'gujo':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/GuJoMoSc01S001R01_ds10_1-11.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/GuJoMoSc01S001R12_ds10_1-2BBCI.mat',
        },
    'jobe':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/JoBeMoSc01S001R01_ds10_1-11.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/JoBeMoSc01S001R12_ds10_1-2BBCI.mat',
        },
    'kaus':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/KaUsMoSc1S001R01_ds10_1-11.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/KaUsMoSc1S001R12_ds10_1-2BBCI.mat',
        },
    'laka':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/LaKaMoSc1S001R01_ds10_1-9.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/LaKaMoSc1S001R10_ds10_1-2BBCI.mat',
        },
    'lufi':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/LuFiMoSc3S001R01_ds10_1-11.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/LuFiMoSc3S001R12_ds10_1-2BBCI.mat',
        },
    'magl':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/MaGlMoSc2S001R01_ds10_1-12.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/MaGlMoSc2S001R13_ds10_1-2BBCI.mat',
        },
    'maja':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/MaJaMoSc1S001R01_ds10_1-11.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/MaJaMoSc1S001R12_ds10_1-2BBCI.mat',
        },
    'maki':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/MaKiMoSC01S001R01_ds10_1-4.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/MaKiMoSC01S001R05_ds10_1-2BBCI.mat',
        },
    'mavo':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/MaVoMoSc1S001R01_ds10_1-11.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/MaVoMoSc1S001R12_ds10_1-2BBCI.mat',
        },
    'nama':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/NaMaMoSc1S001R01_ds10_1-11.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/NaMaMoSc1S001R12_ds10_1-2BBCI.mat',
        },
    'olil':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/OlIlMoSc01S001R01_ds10_1-11.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/OlIlMoSc01S001R12_ds10_1-2BBCI.mat',
        },
    'piwi':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/PiWiMoSc1S001R01_ds10_1-11.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/PiWiMoSc1S001R12_ds10_1-2BBCI.mat',
        },
    'robe':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/RoBeMoSc03S001R01_ds10_1-9.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/RoBeMoSc03S001R10_ds10_1-2BBCI.mat',
        },
    'rosc':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/RoScMoSc1S001R01_ds10_1-11.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/RoScMoSc1S001R12_ds10_1-2BBCI.mat',
        },
    'sthe':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/StHeMoSc01S001R01_ds10_1-10.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/StHeMoSc01S001R11_ds10_1-2BBCI.mat',
        },
    'svmu':
        {
            'train': '/home/schirrmr/data/BBCI-without-last-runs/SvMuMoSc1S001R01_ds10_1-12.BBCI.mat',
            'test': '/home/schirrmr/data/BBCI-only-last-runs/SvMuMoSc1S001R13_ds10_1-2BBCI.mat',
        },
}

def run_exp(subject_name, low_cut_hz, model, cuda):
    log.info("Running for {:s}".format(subject_name))
    train_filename = subject_to_filenames[subject_name]['train']
    test_filename = subject_to_filenames[subject_name]['test']

    train_loader = BBCIDataset(train_filename)
    test_loader = BBCIDataset(test_filename)
    log.info("Loading train data...")
    train_cnt = train_loader.load()
    log.info("Loading test data...")
    test_cnt = test_loader.load()

    # Clean
    # Remember which trials are clean, i,e, no max abs values above 800
    # We did this on all sensors in paper, so we do it already here.
    log.info("Cleaning...")
    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def,
                                                  clean_ival)
    max_val_per_trial = np.max(np.abs(train_set.X), axis=(1, 2))
    train_clean_trial_mask = max_val_per_trial < 800

    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def,
                                                 clean_ival)
    max_val_per_trial = np.max(np.abs(test_set.X), axis=(1, 2))
    test_clean_trial_mask = max_val_per_trial < 800

    del train_set, test_set
    log.info("Train")
    log.info("Total:   {:d}".format(len(train_clean_trial_mask)))
    log.info("Clean:   {:d}".format(np.sum(train_clean_trial_mask)))
    log.info("Unclean: {:d}".format(np.sum(~train_clean_trial_mask)))
    log.info("Test")
    log.info("Total:   {:d}".format(len(test_clean_trial_mask)))
    log.info("Clean:   {:d}".format(np.sum(test_clean_trial_mask)))
    log.info("Unclean: {:d}".format(np.sum(~test_clean_trial_mask)))

    # Preprocessing
    # Without Cz, as Cz was reference
    log.info("Preprocessing...")
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

    train_cnt = train_cnt.pick_channels(C_sensors)
    assert len(train_cnt.ch_names) == 44

    log.info("Highpass...")
    train_cnt = mne_apply(
        lambda a: highpass_cnt(a, low_cut_hz, train_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), train_cnt)
    log.info("Standardizing...")
    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        train_cnt)

    test_cnt = test_cnt.pick_channels(C_sensors)
    assert len(test_cnt.ch_names) == 44

    log.info("Highpass...")
    test_cnt = mne_apply(
        lambda a: highpass_cnt(a, low_cut_hz, test_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt)
    log.info("Standardizing...")
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        test_cnt)
    ival = [-500, 4000]

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    train_set.X = train_set.X[train_clean_trial_mask]
    train_set.y = train_set.y[train_clean_trial_mask]
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
    test_set.X = test_set.X[test_clean_trial_mask]
    test_set.y = test_set.y[test_clean_trial_mask]

    train_set, valid_set = split_into_two_sets(train_set,
                                               first_set_fraction=0.8)

    set_random_seeds(seed=20190706, cuda=cuda)

    n_classes = 4
    n_chans = int(train_set.X.shape[1])
    input_time_length=1000
    if model == 'shallow':
        model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length=30).create_network()
    elif model == 'deep':
        model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length=2).create_network()


    to_dense_prediction_model(model)
    if cuda:
        model.cuda()

    log.info("Model: \n{:s}".format(str(model)))
    dummy_input = np_to_var(train_set.X[:1, :, :, None])
    if cuda:
        dummy_input = dummy_input.cuda()
    out = model(dummy_input)

    n_preds_per_input = out.cpu().data.numpy().shape[2]

    optimizer = optim.Adam(model.parameters())

    iterator = CropsFromTrialsIterator(batch_size=60,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    stop_criterion = Or([MaxEpochs(10),
                         NoDecrease('valid_misclass', 80)])

    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(
                    input_time_length=input_time_length), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    loss_function = lambda preds, targets: F.nll_loss(
        th.mean(preds, dim=2)[:, :, 0], targets)

    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, cuda=cuda)
    exp.run()
    return exp

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    # Should contain both .gdf files and .mat-labelfiles from competition
    clean_subjects = ['bhno', 'famo', 'frth', 'gujo', 'kaus', 'laka', 'lufi',
                      'maja', 'maki', 'mavo', 'piwi', 'robe', 'rosc', 'sthe']
    subject_name = clean_subjects[0]
    low_cut_hz = 4  # 0 or 4
    model = 'shallow'  # 'shallow' or 'deep'
    cuda = True
    exp = run_exp(subject_name, low_cut_hz, model, cuda)
    log.info("Last 10 epochs")
    log.info("\n" + str(exp.epochs_df.iloc[-10:]))