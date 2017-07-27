import logging
import time
import itertools
import os.path
from glob import glob
import numpy as np

from wyrm.processing import select_channels

from braindevel.datasets.trial_segmenter import extract_break_start_stops_ms
from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact
from braindevel.csp.results import CSPResult
from braindevel.csp.experiment import TwoFileCSPExperiment
from braindevel.datasets.loaders import MultipleBBCIDataset
from braindevel.mywyrm.clean import NoCleaner

from braindevel.configs.sacred.online import *  # PARENTCONFIG
log = logging.getLogger(__name__)


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': './data/models/sacred/paper/csp/online/remove-cz/',
        'only_return_exp': False,
    }]
    subject_id_params = dictlistprod({
      'subject_id': ['elkh'] #'anla', 'hawe', 'lufi', 'sama',
    })

    break_params = dictlistprod({
      'with_breaks': [True, False],
    })

    preproc_params = dictlistprod({
        'min_freq': [1],
    })


    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        subject_id_params,
        break_params,
        preproc_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def set_windowed_markers(cnt, all_start_marker_vals, all_end_marker_vals,
                         window_len, window_stride, ):
    new_markers = []
    i_marker = 0
    n_removed_trials = 0
    while i_marker < len(cnt.markers):
        # first find start marker
        while ((i_marker < len(cnt.markers)) and
                   (cnt.markers[i_marker][1] not in all_start_marker_vals)):
            i_marker += 1
        if i_marker < len(cnt.markers):
            start_marker_ms = cnt.markers[i_marker][0]
            start_marker_val = cnt.markers[i_marker][1]
            # find end marker
            i_marker += 1  # advance one past start marker already
            while ((i_marker < len(cnt.markers)) and
                       (cnt.markers[i_marker][1] not in all_end_marker_vals)):
                # Check if there is a new start marker already
                if cnt.markers[i_marker][1] in all_start_marker_vals:
                    log.warn("New start marker  {:.0f} at {:.3f} sec found, "
                             "no end marker for earlier start marker {:.0f} "
                             "at {:.3f} sec found.".format(
                        cnt.markers[i_marker][1],
                        cnt.markers[i_marker][0] / 1000.0,
                        start_marker_val, start_marker_ms / 1000.0))
                    start_marker_ms = cnt.markers[i_marker][0]
                    start_marker_val = cnt.markers[i_marker][1]
                i_marker += 1
            if i_marker == len(cnt.markers):
                log.warn(("No end marker for start marker code {:d} "
                          "at {:.3f} sec found.").format(start_marker_val,
                                                         start_marker_ms / 1000.0))
                break
            end_marker_ms = cnt.markers[i_marker][0]
            end_marker_val = cnt.markers[i_marker][1]
            assert end_marker_val in all_end_marker_vals
            # + window_stride should only create maximum one extra window at the end
            #  to account for fact there may be extra data which does not fill a whole window
            # at the end
            if start_marker_ms + window_len <= end_marker_ms:
                for eval_point in np.arange(start_marker_ms + window_len,
                                          end_marker_ms + window_stride,
                                          window_stride):
                    real_eval_point = min(end_marker_ms, eval_point)
                    new_markers.append((real_eval_point-window_len,
                                        start_marker_val,))
                    if real_eval_point == end_marker_ms: break
                assert real_eval_point == end_marker_ms
            else:
                n_removed_trials += 1
    log.info("Removed Trials since too short: {:d}".format(n_removed_trials))

    cnt.markers = new_markers


def add_break_start_stop_markers(cnt, all_start_marker_vals,
                      all_end_marker_vals,
                      min_break_length_ms, max_break_length_ms,
                      break_start_offset_ms, break_stop_offset_ms,
                      break_start_marker, break_end_marker):
    break_starts, break_stops = extract_break_start_stops_ms(
        cnt.markers, all_start_marker_vals, all_end_marker_vals,)
    break_markers = []
    for break_start_ms, break_stop_ms in zip(break_starts, break_stops):
        break_len_ms = break_stop_ms - break_start_ms
        if (break_len_ms >= min_break_length_ms) and (
                    break_len_ms <= max_break_length_ms):
            mrk_break_start_ms = break_start_ms + break_start_offset_ms
            mrk_break_end_ms = break_stop_ms + break_stop_offset_ms
            break_markers.append((mrk_break_start_ms, break_start_marker))
            break_markers.append((mrk_break_end_ms, break_end_marker))
    times = np.concatenate(
        (np.array(cnt.markers)[:, 0], np.array(break_markers)[:, 0]))
    mrk_codes = np.concatenate(
        (np.array(cnt.markers)[:, 1], np.array(break_markers)[:, 1]))
    sort_inds = np.argsort(times)
    new_markers = []
    for i_mrk in sort_inds:
        new_markers.append((times[i_mrk], mrk_codes[i_mrk]))
    # check that is sorted
    for i_mrk in range(len(new_markers) - 1):
        assert new_markers[i_mrk][0] <= new_markers[i_mrk + 1][0]
    cnt.markers = new_markers
    log.info("#{:d} of {:d} possible breaks added".format(
        len(break_markers) / 2, len(break_starts)))
    return


def run(ex, subject_id, with_breaks, min_freq, only_return_exp):
    start_time = time.time()
    ex.info['finished'] = False


    window_len = 2000
    window_stride = 500
    marker_def = {'1- Right Hand': [1], '2 - Feet': [4],
                  '3 - Rotation': [8], '4 - Words': [10]}
    segment_ival = [0, window_len]
    n_selected_features = 20  # 20

    all_start_marker_vals = [1, 4, 8, 10]
    all_end_marker_vals, train_folders, test_folders = get_subject_config(
        subject_id)

    if with_breaks:
        min_break_length_ms = 6000
        max_break_length_ms = 8000
        break_start_offset_ms = 1000
        break_stop_offset_ms = -500
        break_start_marker = 300
        break_end_marker = 301
        all_start_marker_vals.append(break_start_marker)
        all_end_marker_vals.append(break_end_marker)
        marker_def['5 - Break'] = [break_start_marker]

    train_files_list = [sorted(glob(os.path.join(folder, '*.BBCI.mat')))
                        for folder in train_folders]

    train_files = list(itertools.chain(*train_files_list))
    test_files_list = [sorted(glob(os.path.join(folder, '*.BBCI.mat')))
                       for folder in test_folders]
    test_files = list(itertools.chain(*test_files_list))
    train_set = MultipleBBCIDataset(train_files)
    test_set = MultipleBBCIDataset(test_files)

    csp_exp = TwoFileCSPExperiment(train_set, test_set,
                                   NoCleaner(marker_def=marker_def,
                                             segment_ival=segment_ival),
                                   NoCleaner(marker_def=marker_def,
                                             segment_ival=segment_ival),
                                   resample_fs=250, standardize_cnt=False,
                                   min_freq=min_freq, max_freq=34, last_low_freq=10,
                                   low_width=6, low_overlap=3, high_overlap=4,
                                   high_width=8,
                                   filt_order=3, standardize_filt_cnt=False,
                                   segment_ival=[0, 2000],
                                   standardize_epo=False, n_folds=None,
                                   n_top_bottom_csp_filters=5,
                                   n_selected_filterbands=None, forward_steps=2,
                                   backward_steps=1,
                                   stop_when_no_improvement=False,
                                   n_selected_features=n_selected_features,
                                   only_last_fold=True,
                                   restricted_n_trials=None,
                                   common_average_reference=False,
                                   ival_optimizer=None,
                                   shuffle=False, marker_def=marker_def,
                                   set_cz_to_zero=False, low_bound=0.)
    if only_return_exp:
        return csp_exp
    log.info("Loading train set...")
    csp_exp.load_bbci_set()
    log.info("Loading test set...")
    csp_exp.load_bbci_test_set()
    csp_exp.cnt = select_channels(csp_exp.cnt,['Cz'], invert=True)
    assert len(csp_exp.cnt.axes[1]) == 63
    csp_exp.test_cnt = select_channels(csp_exp.test_cnt,['Cz'], invert=True)
    assert len(csp_exp.test_cnt.axes[1]) == 63
    if with_breaks:
        add_break_start_stop_markers(csp_exp.cnt, all_start_marker_vals,
                                     all_end_marker_vals, min_break_length_ms,
                                     max_break_length_ms, break_start_offset_ms,
                                     break_stop_offset_ms, break_start_marker,
                                     break_end_marker)
    set_windowed_markers(csp_exp.cnt, all_start_marker_vals,
                         all_end_marker_vals, window_len, window_stride, )

    if with_breaks:
        add_break_start_stop_markers(csp_exp.test_cnt, all_start_marker_vals,
                                     all_end_marker_vals, min_break_length_ms,
                                     max_break_length_ms, break_start_offset_ms,
                                     break_stop_offset_ms, break_start_marker,
                                     break_end_marker)
    set_windowed_markers(csp_exp.test_cnt, all_start_marker_vals,
                         all_end_marker_vals, window_len, window_stride, )

    log.info("Cleaning both sets...")
    csp_exp.clean_both_sets()
    log.info("Preprocessing train set...")
    csp_exp.preprocess_set()
    log.info("Preprocessing test set...")
    csp_exp.preprocess_test_set()
    csp_exp.remember_sensor_names()
    csp_exp.init_training_vars()
    log.info("Running Training...")
    csp_exp.run_training()
    end_time = time.time()
    run_time = end_time - start_time

    ex.info['finished'] = True
    result = CSPResult(
        csp_trainer=csp_exp,
        parameters={},
        training_time=run_time)
    assert len(csp_exp.multi_class.test_accuracy) == 1
    assert len(csp_exp.multi_class.train_accuracy) == 1
    ex.info['train_misclass'] = 1 - csp_exp.multi_class.train_accuracy[0]
    ex.info['test_misclass'] = 1 - csp_exp.multi_class.test_accuracy[0]
    ex.info['runtime'] = run_time
    save_pkl_artifact(ex, result, 'csp_result.pkl')
