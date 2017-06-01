import logging
import time
import os.path
from numpy.random import RandomState
import lasagne
from lasagne.updates import adam
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import elu,softmax,identity
from hyperoptim.parse import cartesian_dict_of_lists_product,\
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_npy_artifact, save_pkl_artifact
from braindecode.datasets.combined import CombinedCleanedSet
from braindecode.mywyrm.processing import resample_cnt, bandpass_cnt, exponential_standardize_cnt
from braindecode.datasets.cnt_signal_matrix import CntSignalMatrix
from braindecode.datasets.signal_processor import SignalProcessor
from braindecode.datasets.loaders import BCICompetition4Set2A
from braindecode.models.deep5 import Deep5Net
from braindecode.veganlasagne.layer_util import print_layers
from braindecode.datahandling.splitters import SeveralSetsSplitter
from braindecode.datahandling.batch_iteration import CntWindowTrialIterator
from braindecode.veganlasagne.layers import get_n_sample_preds
from braindecode.veganlasagne.monitors import CntTrialMisclassMonitor, LossMonitor, RuntimeMonitor
from braindecode.experiments.experiment import Experiment
from braindecode.veganlasagne.stopping import MaxEpochs, NoDecrease, Or
from braindecode.veganlasagne.update_modifiers import MaxNormConstraintWithDefaults
from braindecode.results.results import Result

from braindecode.configs.sacred.super_conf import * # PARENTCONFIG

log = logging.getLogger(__name__)

def get_templates():
    lalalala_fn()
    return  {}

def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{ 
        'save_folder': './data/models/sacred/paper/bcic-iv-2a/repl/',
        'only_return_exp': False,
        'n_chans': 22
        }]
    subject_folder_params = dictlistprod({
        'subject_id': range(1,10),
        'data_folder': ['/home/schirrmr/data/bci-competition-iv/2a/',]
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
    assert (only_return_exp is False) or (n_chans is not None) 
    ex.info['finished'] = False
    load_sensor_names = None
    train_filename = 'A{:02d}T.mat'.format(subject_id)
    test_filename = 'A{:02d}E.mat'.format(subject_id)
    train_filepath = os.path.join(data_folder, train_filename)
    test_filepath = os.path.join(data_folder, test_filename)
    
    # trial ivan in milliseconds
    # these are the samples that will be predicted, so for a 
    # network with 2000ms receptive field
    # 1500 means the first receptive field goes from -500 to 1500
    segment_ival = [1500,4000]
    
    
    train_loader = BCICompetition4Set2A(train_filepath, load_sensor_names=load_sensor_names)
    test_loader = BCICompetition4Set2A(test_filepath, load_sensor_names=load_sensor_names)
    
    # Preprocessing pipeline in [(function, {args:values)] logic
    cnt_preprocessors = [
        (resample_cnt , {'newfs': 250.0}),
        (bandpass_cnt, {
            'low_cut_hz': 0,
            'high_cut_hz': 38,
         }),
         (exponential_standardize_cnt, {})
    ]
    
    marker_def = {'1- Right Hand': [1],  '2 - Left Hand': [2], '3 - Rest': [3],
                                                   '4 - Feet': [4]}
    
    train_signal_proc = SignalProcessor(set_loader=train_loader,
        segment_ival=segment_ival,
                                       cnt_preprocessors=cnt_preprocessors,
                                       marker_def=marker_def)
    train_set = CntSignalMatrix(signal_processor=train_signal_proc, sensor_names='all')
    
    test_signal_proc = SignalProcessor(set_loader=test_loader,
        segment_ival=segment_ival,
                                       cnt_preprocessors=cnt_preprocessors,
                                       marker_def=marker_def)
    test_set = CntSignalMatrix(signal_processor=test_signal_proc, sensor_names='all')
    
    from braindecode.mywyrm.clean import MaxAbsCleaner
    train_cleaner = MaxAbsCleaner(segment_ival=[0,4000], threshold=800, marker_def=marker_def)
    test_cleaner = MaxAbsCleaner(segment_ival=[0,4000], threshold=800, marker_def=marker_def)
    combined_set = CombinedCleanedSet(train_set, test_set,train_cleaner, test_cleaner)
    if not only_return_exp:
        combined_set.load()
        
    in_chans = train_set.get_topological_view().shape[1]
    input_time_length = 1000 # implies how many crops are processed in parallel, does _not_ determine receptive field size
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
    n_classes = 4
    final_nonlin=softmax
    first_nonlin=elu
    first_pool_mode='max'
    first_pool_nonlin=identity
    later_nonlin=elu
    later_pool_mode='max'
    later_pool_nonlin=identity
    drop_in_prob=0.0
    drop_prob=0.5
    batch_norm_alpha=0.1
    double_time_convs=False
    split_first_layer=True
    batch_norm=True
    # ensure reproducibility by resetting lasagne/theano random generator
    lasagne.random.set_rng(RandomState(34734))
    
    d5net = Deep5Net(in_chans=in_chans, input_time_length=input_time_length, num_filters_time=num_filters_time,
             filter_time_length=filter_time_length,
             num_filters_spat=num_filters_spat, pool_time_length=pool_time_length, pool_time_stride=pool_time_stride,
             num_filters_2=num_filters_2, filter_length_2=filter_length_2,
             num_filters_3=num_filters_3, filter_length_3=filter_length_3,
             num_filters_4=num_filters_4, filter_length_4=filter_length_4,
             final_dense_length=final_dense_length, n_classes=n_classes,
             final_nonlin=final_nonlin, first_nonlin=first_nonlin,
             first_pool_mode=first_pool_mode, first_pool_nonlin=first_pool_nonlin,
             later_nonlin=later_nonlin, later_pool_mode=later_pool_mode, later_pool_nonlin=later_pool_nonlin,
             drop_in_prob=drop_in_prob, drop_prob=drop_prob, batch_norm_alpha=batch_norm_alpha,
             double_time_convs=double_time_convs,  split_first_layer=split_first_layer, batch_norm=batch_norm)
    final_layer = d5net.get_layers()[-1]
    print_layers(final_layer)
    
    dataset_splitter = SeveralSetsSplitter(valid_set_fraction=0.2, use_test_as_valid=False)
    iterator = CntWindowTrialIterator(batch_size=45,input_time_length=input_time_length,
                                     n_sample_preds=get_n_sample_preds(final_layer))
        
    monitors = [LossMonitor(), CntTrialMisclassMonitor(input_time_length=input_time_length), RuntimeMonitor()]
    
    
    
    
    #debug: n_no_decrease_max_epochs = 2
    #debug: n_max_epochs = 4
    n_no_decrease_max_epochs = 80
    n_max_epochs = 800#100
    # real values for paper were 80 and 800
    stop_criterion = Or([NoDecrease('valid_misclass', num_epochs=n_no_decrease_max_epochs),
                         MaxEpochs(num_epochs=n_max_epochs)])
    
    dataset = combined_set
    splitter = dataset_splitter
    loss_expression = categorical_crossentropy
    updates_expression = adam
    updates_modifier = MaxNormConstraintWithDefaults({})
    remember_best_chan = 'valid_misclass'
    run_after_early_stop=True
    exp = Experiment(final_layer, dataset,splitter,None,iterator, loss_expression,updates_expression, updates_modifier, monitors, 
               stop_criterion, remember_best_chan, run_after_early_stop, batch_modifier=None)

    if only_return_exp:
        return exp
    
    exp.setup()
    exp.run()
    end_time = time.time()
    run_time = end_time - start_time
    
    ex.info['finished'] = True
    ex.info['runtime'] = run_time
    for key in exp.monitor_chans:
        ex.info[key] = exp.monitor_chans[key][-1]
    save_pkl_artifact(ex, exp.monitor_chans, 'monitor_chans.pkl')
