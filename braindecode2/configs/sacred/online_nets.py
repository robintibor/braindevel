import time
import numpy as np

from lasagne.objectives import categorical_crossentropy
from lasagne.updates import adam
import theano.tensor as T
from sklearn.metrics.classification import cohen_kappa_score
from wyrm.processing import select_channels

from hyperoptim.util import save_pkl_artifact, save_npy_artifact

from braindecode.experiments.experiment import Experiment
from braindecode.util import FuncAndArgs
from braindecode.datasets.combined import construct_folder_combined_set
from braindecode.mywyrm.processing import resample_cnt, bandpass_cnt, \
    exponential_standardize_cnt
from braindecode.veganlasagne.layers import get_n_sample_preds
from braindecode.veganlasagne.monitors import Monitor, \
    compute_preds_per_trial, LossMonitor, RuntimeMonitor
from braindecode.datahandling.splitters import SeveralSetsSplitter
from braindecode.datahandling.batch_iteration import \
    BalancedCntWindowTrialIterator
from braindecode.veganlasagne.objectives import sum_of_losses, \
    tied_neighbours_cnt_model_masked
from braindecode.configs.sacred.deep_shallow_hybrid import * # PARENTCONFIG
from braindecode.configs.sacred.online import *  # PARENTCONFIG
from braindecode.veganlasagne.stopping import Or, NoDecrease, MaxEpochs
from braindecode.veganlasagne.update_modifiers import \
    MaxNormConstraintWithDefaults


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': './data/models/sacred/paper/online/with-kappa/',
        'only_return_exp': False,
    }]
    subject_id_params = dictlistprod({
      'subject_id': ['anla','hawe', 'lufi', 'sama' ]#
    })
    preproc_params = dictlistprod({
        'low_cut_hz': [0],
    })
    stop_params = dictlistprod({
        'max_epochs': [800],
    })
    break_params = dictlistprod({
      'with_breaks': [False, ],#False
    })
    model_params = dictlistprod({
        'network': ['deep_max_time','shallow'],#,'deep_max_time' 'deep', ,'shallow', 'merged'

    })


    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        subject_id_params,
        preproc_params,
        break_params,
        stop_params,
        model_params
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


class OnlineWindowMisclassMonitor(Monitor):
    def __init__(self, input_time_length=None, window_stride=None):
        self.input_time_length = input_time_length
        self.window_stride=window_stride

    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_single_misclass".format(setname)
            monitor_chans[monitor_key] = []
            monitor_key = "{:s}_mean_misclass".format(setname)
            monitor_chans[monitor_key] = []
            monitor_key = "{:s}_single_kappa".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_preds, losses,
                    all_batch_sizes, targets, dataset):

        preds_per_trial = compute_preds_per_trial(dataset.y, all_preds,
                                                  all_batch_sizes,
                                                  self.input_time_length)
        targets_per_trial = compute_preds_per_trial(dataset.y, targets,
                                                    all_batch_sizes,
                                                    self.input_time_length)
        preds_per_window = []
        targets_per_window = []
        meaned_preds_per_window = []
        for p, t in zip(preds_per_trial, targets_per_trial):
            for i_eval_point in range(0, len(p) + self.window_stride - 1,
                                      self.window_stride):
                real_eval_point = min(len(p) - 1, i_eval_point)
                preds_per_window.append(p[real_eval_point])
                if real_eval_point > 0:
                    i_start = max(0,real_eval_point - self.window_stride)
                    meaned_pred = np.mean(p[i_start:real_eval_point], axis=0)
                    meaned_preds_per_window.append(meaned_pred)
                else:
                    meaned_preds_per_window.append(p[real_eval_point])
                targets_per_window.append(t[real_eval_point])
                if real_eval_point == len(p) - 1: break
            assert real_eval_point == len(p) - 1
        assert np.all(np.sum(np.asarray(targets_per_window), axis=1) == 1)
        single_window_preds = np.argmax(preds_per_window, axis=1)
        window_labels = np.argmax(targets_per_window,axis=1)

        acc = np.mean(single_window_preds == window_labels)
        kappa = cohen_kappa_score(single_window_preds, window_labels)

        meaned_pred_acc = np.mean(
            np.argmax(meaned_preds_per_window, axis=1) == np.argmax(
                targets_per_window, axis=1))
        monitor_key = "{:s}_single_misclass".format(setname)
        monitor_chans[monitor_key].append(float(1 - acc))
        monitor_key = "{:s}_mean_misclass".format(setname)
        monitor_chans[monitor_key].append(float(1 - meaned_pred_acc))
        monitor_key = "{:s}_single_kappa".format(setname)
        monitor_chans[monitor_key].append(kappa)


def run(ex, subject_id, with_breaks, low_cut_hz, network, max_epochs,
        only_return_exp=False):
    start_time = time.time()
    ex.info['finished'] = False
    segment_ival = [0, 2000]
    n_chans = 63
    marker_def = {'1 - Right Hand': [1], '2 - Feet': [4],
                  '3 - Rotation': [8], '4 - Words': [10]}
    trial_classes = ['1 - Right Hand', '2 - Feet',
                     '3 - Rotation', '4 - Words']
    n_classes = 4
    if with_breaks:
        min_break_length_ms = 6000
        max_break_length_ms = 8000
        break_start_offset_ms = 3000
        break_stop_offset_ms = -500
        n_classes = 5
    else:
        min_break_length_ms = None
        max_break_length_ms = None
        break_start_offset_ms = None
        break_stop_offset_ms = None


    window_stride = int(500 * 250.0 / 1000.0)

    all_end_marker_vals, train_folders, test_folders = get_subject_config(
        subject_id
    )
    end_marker_def = {}
    cnt_prepreprocessors = [
        (select_channels, {'regexp_list': ['Cz'], 'invert': True}),
        (resample_cnt, {'newfs': 250.0}),
        (bandpass_cnt, {'low_cut_hz': low_cut_hz, 'high_cut_hz': 38, 'filt_order': 3}),
        (exponential_standardize_cnt, {}),

    ]
    for key in marker_def:
        end_marker_def[key] = all_end_marker_vals

    dataset = construct_folder_combined_set(train_folders + test_folders, None,
                                            cnt_prepreprocessors, marker_def,
                                            end_marker_def,
                                            trial_classes, segment_ival[1], 0,
                                            min_break_length_ms,
                                            max_break_length_ms,
                                            break_start_offset_ms,
                                            break_stop_offset_ms, None,
                                            add_trial_breaks=with_breaks
                                            )

    if not only_return_exp:
        dataset.load()

    input_time_length = 800
    # ensure reproducibility by resetting lasagne/theano random generator
    lasagne.random.set_rng(RandomState(34734))
    if network == 'deep':
        final_layer = create_deep_net(n_chans, input_time_length, 1, n_classes)
    elif network == 'deep_max_time':
        final_layer = create_deep_net(n_chans, input_time_length, 2, n_classes,
                                      filter_length_4=9)
    elif network == 'shallow':
        final_layer = create_shallow_net(n_chans, input_time_length, 27,
                                         n_classes)
    elif network == 'merged':
        final_layer = create_merged_net(n_chans, input_time_length, 1, 24,
                                        n_classes, 13)
    def masked_loss(preds, targets, final_layer, loss_expression):
        """masked by any target being active"""
        assert targets.ndim == 2
        valid_inds = T.nonzero(T.gt(T.sum(targets, axis=1), 0))
        valid_preds = preds[valid_inds]
        valid_targets = targets[valid_inds]
        return loss_expression(valid_preds, valid_targets, final_layer)

    splitter = SeveralSetsSplitter(valid_set_fraction=0.2,
                                   use_test_as_valid=False)
    preprocessor = None
    iterator = BalancedCntWindowTrialIterator(batch_size=45,
                                              input_time_length=input_time_length,
                                              n_sample_preds=get_n_sample_preds(
                                                  final_layer),
                                              check_preds_smaller_trial_len=False)

    loss_expression = FuncAndArgs(
        sum_of_losses, loss_expressions=[
            FuncAndArgs(masked_loss, loss_expression=lambda p, t,
                                                            l: categorical_crossentropy(
                p, t)),
            tied_neighbours_cnt_model_masked
        ])  # tied_neighbours_cnt_model
    # loss_expression = categorical_crossentropy
    updates_expression = adam
    updates_modifier = MaxNormConstraintWithDefaults({})
    monitors = [LossMonitor(),
                OnlineWindowMisclassMonitor(window_stride=window_stride,
                                            input_time_length=input_time_length),
                RuntimeMonitor()]
    remember_chan = 'valid_single_misclass'
    stop_criterion = Or([NoDecrease(remember_chan, 80), MaxEpochs(max_epochs)])
    remember_best_chan = remember_chan
    run_after_early_stop = True
    exp = Experiment(final_layer, dataset, splitter, preprocessor,
                     iterator, loss_expression, updates_expression,
                     updates_modifier,
                     monitors, stop_criterion, remember_best_chan,
                     run_after_early_stop, batch_modifier=None)
    if only_return_exp:
        return exp

    exp.setup()
    exp.run()

    end_time = time.time()
    run_time = end_time - start_time

    ex.info['finished'] = True
    for key in exp.monitor_chans:
        ex.info[key] = exp.monitor_chans[key][-1]
    ex.info['runtime'] = run_time
    save_pkl_artifact(ex, exp.monitor_chans, 'monitor_chans.pkl')
    save_npy_artifact(ex, lasagne.layers.get_all_param_values(exp.final_layer),
        'model_params.npy')
