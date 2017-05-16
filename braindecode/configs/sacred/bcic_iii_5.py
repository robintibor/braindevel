import logging
import time
from numpy.random import RandomState
import lasagne
import numpy as np

from braindecode.datasets.trial_segmenter import PipelineSegmenter, \
    MarkerSegmenter
from braindecode.veganlasagne.nonlinearities import square, safe_log
from lasagne.updates import adam
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import elu,softmax,identity

from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from hyperoptim.parse import cartesian_dict_of_lists_product,\
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_npy_artifact, save_pkl_artifact
from braindecode.datasets.combined import CombinedSet
from braindecode.mywyrm.processing import resample_cnt, bandpass_cnt, exponential_standardize_cnt
from braindecode.datasets.cnt_signal_matrix import SetWithMarkers
from braindecode.datasets.loaders import BCICompetition3Set5
from braindecode.models.deep5 import Deep5Net
from braindecode.datahandling.splitters import CntTrialSeveralSetsSplitter
from braindecode.datahandling.batch_iteration import CntWindowTrialIterator
from braindecode.veganlasagne.layers import get_n_sample_preds, \
    get_model_input_window, create_pred_fn
from braindecode.veganlasagne.monitors import CntTrialMisclassMonitor, \
    LossMonitor, RuntimeMonitor,  MisclassMonitor
from braindecode.experiments.experiment import Experiment
from braindecode.veganlasagne.stopping import MaxEpochs, NoDecrease, Or
from braindecode.veganlasagne.update_modifiers import MaxNormConstraintWithDefaults
from braindecode.veganlasagne.objectives import tied_neighbours_cnt_model,\
    sum_of_losses
from braindecode.veganlasagne.monitors import  compute_preds_per_trial_from_start_end
from braindecode.util import FuncAndArgs
from braindecode.veganlasagne.clip import ClipLayer
from braindecode.datahandling.batch_iteration import get_start_end_blocks_for_trial, create_batch
from braindecode.veganlasagne.layers import get_n_sample_preds, get_model_input_window, get_input_time_length

log = logging.getLogger(__name__)

def get_templates():
    return  {'categorical_crossentropy': lambda : categorical_crossentropy,
        'tied_loss': lambda : FuncAndArgs(sum_of_losses,
            loss_expressions=[categorical_crossentropy,
                tied_neighbours_cnt_model ,
            ]
        )
        }

def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{ 
        'save_folder': './data/models/sacred/paper/bcic-iii-5/variable-len-bcic-metric/',
        'only_return_exp': False,
        'n_chans': 32,
        'run_after_early_stop': True,
        }]
    subject_folder_params = dictlistprod({
        'subject_id': range(1,4),
        'data_folder': ['/home/schirrmr/data/bci-competition-iii/5/'],
    })
    model_params = dictlistprod({
        'network': ['deep', 'shallow'],#'deep'#'shallow'
    })
    stop_params = dictlistprod({
        'stop_chan': ['sample_misclass']})#'misclass',
    
    loss_params = dictlistprod({
        'loss_expression': ['$tied_loss']})#'misclass', 
    preproc_params = dictlistprod({
        'resample_fs': [256.0],
        'filt_order': [3,],#10
        'low_cut_hz': [0,4],
        'high_cut_hz': [38, ],
        'fix_start_train': [True]})#None

    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        subject_folder_params,
        model_params,
        stop_params,
        preproc_params,
        loss_params,
        ])
    
    return grid_params


def sample_config_params(rng, params):
    return params


def create_deep_net(in_chans, input_time_length):
    # receptive field size is determined by model architecture
    num_filters_time = 25
    filter_time_length = 10
    num_filters_spat = 25
    pool_time_length = 2
    pool_time_stride = 2
    num_filters_2 = 50
    filter_length_2 = 10
    num_filters_3 = 100
    filter_length_3 = 10
    num_filters_4 = 200
    filter_length_4 = 10
    final_dense_length = 7
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

    d5net = Deep5Net(in_chans=in_chans, input_time_length=input_time_length,
                     num_filters_time=num_filters_time,
                     filter_time_length=filter_time_length,
                     num_filters_spat=num_filters_spat,
                     pool_time_length=pool_time_length,
                     pool_time_stride=pool_time_stride,
                     num_filters_2=num_filters_2,
                     filter_length_2=filter_length_2,
                     num_filters_3=num_filters_3,
                     filter_length_3=filter_length_3,
                     num_filters_4=num_filters_4,
                     filter_length_4=filter_length_4,
                     final_dense_length=final_dense_length, n_classes=n_classes,
                     final_nonlin=final_nonlin, first_nonlin=first_nonlin,
                     first_pool_mode=first_pool_mode,
                     first_pool_nonlin=first_pool_nonlin,
                     later_nonlin=later_nonlin, later_pool_mode=later_pool_mode,
                     later_pool_nonlin=later_pool_nonlin,
                     drop_in_prob=drop_in_prob, drop_prob=drop_prob,
                     batch_norm_alpha=batch_norm_alpha,
                     double_time_convs=double_time_convs,
                     split_first_layer=split_first_layer, batch_norm=batch_norm)
    final_layer = d5net.get_layers()[-1]
    final_layer = ClipLayer(final_layer, 1e-4, 1 - 1e-4)
    return final_layer


def create_shallow_net(in_chans, input_time_length):
    # receptive field size is determined by model architecture
    n_classes = 3
    # ensure reproducibility by resetting lasagne/theano random generator
    lasagne.random.set_rng(RandomState(34734))

    shallow_net = ShallowFBCSPNet(in_chans, input_time_length, n_classes,
            n_filters_time=40,
            filter_time_length=26,
            n_filters_spat=40,
            pool_time_length=50,
            pool_time_stride=10,
            final_dense_length=19,
            conv_nonlin=square,
            pool_mode='average_exc_pad',
            pool_nonlin=safe_log,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            drop_prob=0.5)
    final_layer = shallow_net.get_layers()[-1]
    final_layer = ClipLayer(final_layer, 1e-4, 1 - 1e-4)
    return final_layer


def compute_bcic_metric(test_set, final_layer, ):
    pred_fn = create_pred_fn(final_layer)

    model_input_window = get_model_input_window(final_layer)
    n_sample_preds = get_n_sample_preds(final_layer)
    input_time_length = get_input_time_length(final_layer)
    trial_start = model_input_window - 1
    trial_end = len(test_set.y) - 1
    start_end_blocks = get_start_end_blocks_for_trial(
        trial_start, trial_end, input_time_length=input_time_length,
        n_sample_preds=n_sample_preds)

    full_batch = create_batch(test_set.get_topological_view(), test_set.y,
                              start_end_blocks, n_sample_preds)
    all_outs = pred_fn(full_batch[0])
    cnt_preds = compute_preds_per_trial_from_start_end(
        [all_outs], [len(full_batch)], [trial_start], [trial_end])[0]
    padded_preds = np.concatenate(
        (np.zeros((model_input_window - 1, 3)), cnt_preds), axis=0)

    # one sample should be lost form resampling so pad pred with 0 at start
    assert padded_preds.shape == test_set.y.shape
    assert padded_preds.shape[0] % 16 == 15
    padded_preds = np.concatenate((np.zeros((1, 3)), padded_preds), axis=0)
    assert padded_preds.shape[0] % 256 == 0
    secs_in_recording = padded_preds.shape[0] / 256
    pred_labels = []
    for sec_to_predict in np.arange(1, secs_in_recording + 0.1, 0.5):
        start_sec = sec_to_predict - 1
        i_start_sample = int(start_sec * 256)
        i_stop_sample = int(sec_to_predict * 256)
        label = np.argmax(
            np.mean(padded_preds[i_start_sample:i_stop_sample], axis=0))
        pred_labels.append(label)
    pred_labels = np.array(pred_labels)
    y = np.loadtxt(
        'data/bci-competition-iii/5/labels/labels8_subject{:d}_raw.asc'.format(
            test_set.set_loader.subject_id
        ))
    y[y == 2] = 0
    y[y == 7] = 2
    y[y == 3] = 1
    y = np.array(y)
    assert len(pred_labels) == len(y)
    acc = np.mean(pred_labels == y)
    return acc


def run(
        ex, data_folder, subject_id, n_chans,
        stop_chan, resample_fs, filt_order, low_cut_hz, high_cut_hz,
        fix_start_train,
        loss_expression, network, only_return_exp, run_after_early_stop):
    start_time = time.time()
    assert (only_return_exp is False) or (n_chans is not None) 
    ex.info['finished'] = False
    
    # trial ival in milliseconds
    # these are the samples that will be predicted, so for a 
    # network with 2000ms receptive field
    # 1500 means the first receptive field goes from -500 to 1500
    train_segment_ival = [500,0]
    test_segment_ival = [0, 0]

    train_loader = BCICompetition3Set5(subject_id=subject_id,
                                       folder=data_folder,
                                       train_or_test='train')

    test_loader = BCICompetition3Set5(subject_id=subject_id,
                                       folder=data_folder,
                                       train_or_test='test')
    
    # Preprocessing pipeline in [(function, {args:values)] logic
    cnt_preprocessors = [
        (resample_cnt , {'newfs': resample_fs}),
        (bandpass_cnt, {
            'low_cut_hz': low_cut_hz,
            'high_cut_hz': high_cut_hz,
            'filt_order': filt_order,
         }),
         (exponential_standardize_cnt, {})
    ]
    
    marker_def = {'1- Left Hand': [2],  '2 - Right Hand': [3],
                  '3 - Words': [7]}

    end_marker_def = {'1- Left Hand': [12],  '2 - Right Hand': [13],
                      '3 - Words': [17]}

    trial_classes = ['1- Left Hand', '2 - Right Hand', '3 - Words']
    train_segmenter = PipelineSegmenter(
        [MarkerSegmenter(train_segment_ival, marker_def, trial_classes,
                         end_marker_def=end_marker_def)])
    test_segmenter = PipelineSegmenter(
        [MarkerSegmenter(test_segment_ival, marker_def, trial_classes,
                         end_marker_def=end_marker_def)])

    train_set = SetWithMarkers(train_loader, cnt_preprocessors, train_segmenter)
    test_set = SetWithMarkers(test_loader, cnt_preprocessors, test_segmenter)
    input_time_length = 500 # implies how many crops are processed in parallel, does _not_ determine receptive field size
    # receptive field size is determined by model architecture
    if not only_return_exp:
        train_set.load()
        test_set.load()
        if fix_start_train:
            train_set.y[:input_time_length + int(4 * resample_fs)] = 0
        in_chans = train_set.get_topological_view().shape[1]
    else:
        in_chans = n_chans

    dataset  = CombinedSet([train_set, test_set])

    lasagne.random.set_rng(RandomState(34734))

    if network == 'deep':
        final_layer = create_deep_net(in_chans, input_time_length)
    else:
        assert network == 'shallow'
        final_layer = create_shallow_net(in_chans, input_time_length)

    log.info("Model input window: {:d}".format(
        get_model_input_window(final_layer)))
    dataset_splitter = CntTrialSeveralSetsSplitter(valid_set_fraction=0.2,
                                                   use_test_as_valid=False)
    iterator = CntWindowTrialIterator(batch_size=45,input_time_length=input_time_length,
                                     n_sample_preds=get_n_sample_preds(final_layer))
        
    monitors = [LossMonitor(),
                CntTrialMisclassMonitor(input_time_length=input_time_length),
                MisclassMonitor(chan_name='sample_misclass'),
                RuntimeMonitor()]
    
    
    
    #debug: n_no_decrease_max_epochs = 2
    #debug: n_max_epochs = 4
    n_no_decrease_max_epochs = 80
    n_max_epochs = 800#100
    # real values for paper were 80 and 800
    remember_best_chan = 'valid_' + stop_chan
    stop_criterion = Or([NoDecrease(remember_best_chan, num_epochs=n_no_decrease_max_epochs),
                         MaxEpochs(num_epochs=n_max_epochs)])
    
    splitter = dataset_splitter
    updates_expression = adam
    updates_modifier = MaxNormConstraintWithDefaults({})
    exp = Experiment(final_layer, dataset,splitter,None,iterator, loss_expression,updates_expression, updates_modifier, monitors, 
               stop_criterion, remember_best_chan, run_after_early_stop, batch_modifier=None)

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
    log.info("Compute bcic metric")
    acc = compute_bcic_metric(test_set, exp.final_layer)
    ex.info['bcic_misclass'] = 1 - acc
