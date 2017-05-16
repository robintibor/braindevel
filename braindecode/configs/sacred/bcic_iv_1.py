import logging
import time
import numpy as np
from numpy.random import RandomState
import lasagne
from lasagne.layers.merge import ConcatLayer
from lasagne.updates import adam
from lasagne.objectives import squared_error, categorical_crossentropy
from lasagne.nonlinearities import elu, softmax, identity, tanh, sigmoid
from lasagne.layers.conv import Conv2DLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import batch_norm as batch_norm_fn

from braindecode.veganlasagne.clip import ClipLayer
from hyperoptim.parse import cartesian_dict_of_lists_product,\
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_npy_artifact, save_pkl_artifact
from braindecode.mywyrm.processing import resample_cnt, bandpass_cnt, exponential_standardize_cnt
from braindecode.datasets.cnt_signal_matrix import SetWithMarkers
from braindecode.datasets.loaders import BCICompetition4Set1
from braindecode.models.deep5 import Deep5Net
from braindecode.datahandling.splitters import CntTrialSingleFoldSplitter, \
    CntTrialSeveralSetsSplitter
from braindecode.datahandling.batch_iteration import (
    CntWindowTrialBCICompIVSet1Iterator, CntWindowTrialIterator)
from braindecode.veganlasagne.layers import get_n_sample_preds,\
    FinalReshapeLayer
from braindecode.veganlasagne.monitors import LossMonitor, RuntimeMonitor, \
    CorrelationMonitor, MeanSquaredErrorClassMonitor, CorrelationClassMonitor, \
    CntTrialMisclassMonitor, MeanSquaredErrorMonitor
from braindecode.experiments.experiment import Experiment
from braindecode.veganlasagne.stopping import MaxEpochs, NoDecrease, Or
from braindecode.veganlasagne.update_modifiers import MaxNormConstraintWithDefaults
from braindecode.util import FuncAndArgs
from braindecode.veganlasagne.objectives import sum_of_losses, \
    tied_neighbours_cnt_model
from braindecode.datasets.trial_segmenter import PipelineSegmenter,\
    MarkerSegmenter, AddTrialBreaks
from braindecode.veganlasagne.objectives import (
    tied_neighbours_cnt_model_custom_loss)
from braindecode.datasets.combined import CombinedSet

log = logging.getLogger(__name__)


def get_templates():
    return {'squared_error': lambda : squared_error,
            'tied_loss_squared': lambda : FuncAndArgs(
                sum_of_losses,
                loss_expressions=[
                    squared_error,
                    FuncAndArgs(
                        tied_neighbours_cnt_model_custom_loss,
                        loss_fn=squared_error)]),
            'tied_loss_crossent': lambda : FuncAndArgs(
                sum_of_losses,
                loss_expressions=[
                    categorical_crossentropy,
                    tied_neighbours_cnt_model,]),
            'crossent': lambda : categorical_crossentropy}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{ 
        'save_folder': './data/models/sacred/paper/bcic-iv-1/linear/',
        'only_return_exp': False,
        'n_chans': 59
        }]
    subject_folder_params = dictlistprod({
        'subject_id': ['a', 'b', 'f', 'g'],
        'data_folder': ['data/bci-competition-iv/1/',]
    })
    preproc_params = dictlistprod({
        'filt_order': [3, ], # 10#10
        'low_cut_hz': [4,], # 4
        'high_cut_hz': [38, ],}) # NoneNone
    loss_params = dictlistprod({
        'loss_expression': ['$tied_loss_squared',]}) #, ''
    final_part_params = dictlistprod({
        'n_dense_filters': [3, None],#50
        'final_part': ['linear' #'sigmoid_gated_tanh', 'tanh', #''sigmoid_gated_tanh', #'softmax_diff', 'softmax_diff_product', 'tanh', 'linear',
                       ]
    })
    monitor_params = dictlistprod({
        'out_factor': [0.8,],
        'demean_preds': [False, True],
        'early_stop_chan': ['valid_mse']
    })
    classification_or_regression = dictlistprod({
        'treat_as_classification': [False]
    })

    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        subject_folder_params,
        preproc_params,
        loss_params,
        final_part_params,
        classification_or_regression,
        monitor_params,
        ])
    
    return grid_params


def sample_config_params(rng, params):
    return params


def _create_deep_net(input_time_length, in_chans, n_dense_filters, final_part,
                     treat_as_classification):
    # implies how many crops are processed in parallel,
    # does _not_ determine receptive field size
    # receptive field size is determined by model architecture
    num_filters_time = 25
    filter_time_length = 10
    num_filters_spat = 25
    pool_time_length = 3
    pool_time_stride = 3
    num_filters_2 = 50
    filter_length_2 = 10
    num_filters_3 = 100
    filter_length_3 = 10
    num_filters_4 = 200
    filter_length_4 = 10
    final_dense_length = 2
    n_classes = 3
    final_nonlin = softmax
    first_nonlin = elu
    first_pool_mode = 'max'
    first_pool_nonlin = identity
    later_nonlin = elu
    later_pool_mode = 'max'
    later_pool_nonlin = identity
    drop_in_prob = 0.0
    drop_prob = 0.5
    batch_norm_alpha = 0.1
    double_time_convs = False
    split_first_layer = True
    batch_norm = True
    # ensure reproducibility by resetting lasagne/theano random generator
    lasagne.random.set_rng(RandomState(34734))

    d5net = Deep5Net(
        in_chans=in_chans, input_time_length=input_time_length,
        num_filters_time=num_filters_time,
        filter_time_length=filter_time_length,
        num_filters_spat=num_filters_spat, pool_time_length=pool_time_length,
        pool_time_stride=pool_time_stride,
        num_filters_2=num_filters_2, filter_length_2=filter_length_2,
        num_filters_3=num_filters_3, filter_length_3=filter_length_3,
        num_filters_4=num_filters_4, filter_length_4=filter_length_4,
        final_dense_length=final_dense_length, n_classes=n_classes,
        final_nonlin=final_nonlin, first_nonlin=first_nonlin,
        first_pool_mode=first_pool_mode, first_pool_nonlin=first_pool_nonlin,
        later_nonlin=later_nonlin, later_pool_mode=later_pool_mode,
        later_pool_nonlin=later_pool_nonlin,
        drop_in_prob=drop_in_prob, drop_prob=drop_prob,
        batch_norm_alpha=batch_norm_alpha,
        double_time_convs=double_time_convs,
        split_first_layer=split_first_layer, batch_norm=batch_norm)
    final_layer = d5net.get_layers()[-1]
    if not treat_as_classification:
        l = final_layer.input_layer.input_layer
        if (n_dense_filters is not None) and (n_dense_filters > 0):
            l = Conv2DLayer(l, num_filters=n_dense_filters,
                            filter_size=(2, 1), nonlinearity=elu)
            l = batch_norm_fn(l)
        # determine next filter size
        if (n_dense_filters is None) or (n_dense_filters == 0):
            filter_size = (2, 1)
        else:
            filter_size = (1, 1)

        if final_part == 'softmax_diff' or final_part == 'softmax_diff_product':
            l = Conv2DLayer(l, num_filters=3, filter_size=filter_size,
                            nonlinearity=identity)
            l = FinalReshapeLayer(l, flatten=True)
            l = NonlinearityLayer(l, softmax)
            if final_part == 'softmax_diff':
                l = ExpressionLayer(l, lambda x: x[:, 0:1] - x[:, 1:2],
                                    lambda s: (s[0], 1))
            else:
                assert final_part == 'softmax_diff_product'
                l = ExpressionLayer(l,
                                    lambda x: (x[:, 0:1] - x[:, 1:2] *
                                              (1 - x[:, 2:3])),
                                    lambda s: (s[0], 1))

        elif final_part == 'linear':
            l = Conv2DLayer(l, num_filters=1, filter_size=filter_size,
                            nonlinearity=identity)
            l = FinalReshapeLayer(l, flatten=True)
        elif final_part == 'tanh':
            l = Conv2DLayer(l, num_filters=1, filter_size=filter_size,
                            nonlinearity=tanh)
            l = FinalReshapeLayer(l, flatten=True)
        elif final_part == 'pow3':
            l = Conv2DLayer(l, num_filters=1, filter_size=filter_size,
                            nonlinearity=lambda x: x * x * x)
            l = FinalReshapeLayer(l, flatten=True)
        elif final_part == 'sigmoid_gated_tanh':
            l1 = Conv2DLayer(l,
                             num_filters=1,
                             filter_size=filter_size, nonlinearity=tanh)
            l2 = Conv2DLayer(l,
                             num_filters=1,
                             filter_size=filter_size, nonlinearity=sigmoid)
            l = ConcatLayer([l1, l2])
            l = ExpressionLayer(l, lambda x: x[:, 0:1] * (1 - x[:, 1:2]),
                                lambda s: (s[0], 1, s[2], s[3]))
            l = FinalReshapeLayer(l, flatten=True)
            l = ClipLayer(l, 1e-8, 1-1e-8)
        else:
            assert False
        final_layer = l
    else:
        assert treat_as_classification
        final_layer = ClipLayer(final_layer, 1e-8, 1-1e-8)
    return final_layer


def _run_exp(valid_set_fraction, dataset, in_chans, n_dense_filters, final_part,
             loss_expression, treat_as_classification, out_factor,
             demean_preds, early_stop_chan, only_return_exp):
    input_time_length = 645
    final_layer = _create_deep_net(input_time_length, in_chans, n_dense_filters,
                                   final_part, treat_as_classification)

    dataset_splitter = CntTrialSeveralSetsSplitter(
        valid_set_fraction=valid_set_fraction, use_test_as_valid=False)
    batch_size = 90
    if treat_as_classification:
        iterator = CntWindowTrialIterator(batch_size=batch_size,
                                          input_time_length=input_time_length,
                                          n_sample_preds=get_n_sample_preds(
                                              final_layer))
        monitors = [LossMonitor(), MeanSquaredErrorClassMonitor(
            out_factor=out_factor, demean_preds=demean_preds),
                    CorrelationClassMonitor(),
                    CntTrialMisclassMonitor(input_time_length),
                    RuntimeMonitor()]
    else:
        iterator = CntWindowTrialBCICompIVSet1Iterator(
            batch_size=batch_size, input_time_length=input_time_length,
            n_sample_preds=get_n_sample_preds(final_layer))

        monitors = [LossMonitor(), MeanSquaredErrorMonitor(
            out_factor=out_factor, demean_preds=demean_preds),
                    CorrelationMonitor(),
                    RuntimeMonitor()]

    # debug: n_no_decrease_max_epochs = 2
    # debug: n_max_epochs = 4
    n_no_decrease_max_epochs = 80
    n_max_epochs = 800  # 100
    # real values for paper were 80 and 800
    stop_criterion = Or(
        [NoDecrease(early_stop_chan, num_epochs=n_no_decrease_max_epochs),
         MaxEpochs(num_epochs=n_max_epochs)])

    splitter = dataset_splitter
    updates_expression = adam
    updates_modifier = MaxNormConstraintWithDefaults({})
    remember_best_chan = early_stop_chan
    run_after_early_stop = True
    exp = Experiment(final_layer, dataset, splitter, None, iterator,
                     loss_expression, updates_expression, updates_modifier,
                     monitors,
                     stop_criterion, remember_best_chan, run_after_early_stop,
                     batch_modifier=None)
    if only_return_exp:
        return exp

    exp.setup()
    exp.run()
    return exp


def run(ex, data_folder, subject_id, n_chans,
        low_cut_hz, high_cut_hz,
        filt_order,
        loss_expression,
        n_dense_filters,
        final_part,
        treat_as_classification,
        out_factor,
        demean_preds,
        early_stop_chan,
        only_return_exp,):
    start_time = time.time()
    assert (only_return_exp is False) or (n_chans is not None)
    train_loader = BCICompetition4Set1(subject_id, 'train', data_folder)
    test_loader = BCICompetition4Set1(subject_id, 'test', data_folder)

    # Preprocessing pipeline in [(function, {args:values)] logic
    cnt_preprocessors = [
        (resample_cnt , {'newfs': 250.0}),
        (bandpass_cnt, {
            'low_cut_hz': low_cut_hz,
            'high_cut_hz': high_cut_hz,
            'filt_order': filt_order
         }),
         (exponential_standardize_cnt, {})
    ]
    train_segment_ival = [1000,4000]
    test_segment_ival = [1000, 0]
    marker_def = {'1- Left Hand': [1], '2 - Right Hand': [2]}
    trial_classes = ['1- Left Hand', '2 - Right Hand']
    test_end_marker_def = {'1- Left Hand': [11], '2 - Right Hand': [12]}
    train_end_marker_def = None
    train_segmenter = PipelineSegmenter(
        [MarkerSegmenter(train_segment_ival, marker_def, trial_classes,
                         end_marker_def=train_end_marker_def),
         AddTrialBreaks(min_length_ms=1000, max_length_ms=9000,
                        start_offset_ms=1000, stop_offset_ms=0,
                        start_marker_def=marker_def, trial_to_break_ms=4000)])
    test_segmenter = PipelineSegmenter(
        [MarkerSegmenter(test_segment_ival, marker_def, trial_classes,
                         end_marker_def=test_end_marker_def),
         AddTrialBreaks(min_length_ms=1500, max_length_ms=9000,
                        start_offset_ms=1000, stop_offset_ms=0,
                        start_marker_def=marker_def,
                        end_marker_def=test_end_marker_def)])
    ex.info['finished'] = False
    train_set = SetWithMarkers(train_loader, cnt_preprocessors, train_segmenter)
    test_set = SetWithMarkers(test_loader, cnt_preprocessors, test_segmenter)
    if not only_return_exp:
        train_set.load()
        test_set.load()
        in_chans = train_set.get_topological_view().shape[1]
    else:
        in_chans = n_chans

    dataset  = CombinedSet([train_set, test_set])
    all_monitor_chans = []
    valid_set_fraction = 0.1
    exp = _run_exp(valid_set_fraction, dataset, in_chans, n_dense_filters,
                   final_part, loss_expression, treat_as_classification,
                   out_factor, demean_preds, early_stop_chan, only_return_exp)
    if only_return_exp:
        return exp

    end_time = time.time()
    run_time = end_time - start_time
    
    ex.info['finished'] = True
    for key in exp.monitor_chans:
        ex.info[key] = exp.monitor_chans[key][-1]
    ex.info['runtime'] = run_time
    save_pkl_artifact(ex, all_monitor_chans, 'monitor_chans.pkl')
    save_npy_artifact(ex, lasagne.layers.get_all_param_values(exp.final_layer),
        'model_params.npy')
