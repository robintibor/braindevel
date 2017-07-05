import logging
import time
import numpy as np
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
from braindecode.datasets.loaders import MultipleBCICompetition4Set2B
from braindecode.models.deep5 import Deep5Net
from braindecode.datahandling.splitters import SeveralSetsSplitter,\
    concatenate_sets, CntTrialSingleFoldSplitter
from braindecode.datahandling.batch_iteration import CntWindowTrialIterator
from braindecode.veganlasagne.layers import get_n_sample_preds
from braindecode.veganlasagne.monitors import CntTrialMisclassMonitor, LossMonitor, RuntimeMonitor,\
    KappaMonitor
from braindecode.experiments.experiment import Experiment
from braindecode.veganlasagne.stopping import MaxEpochs, NoDecrease, Or
from braindecode.veganlasagne.update_modifiers import MaxNormConstraintWithDefaults
from braindecode.veganlasagne.objectives import tied_neighbours_cnt_model,\
    sum_of_losses
from braindecode.util import FuncAndArgs
from braindecode.mywyrm.clean import NoCleaner, BCICompetitionIV2ABArtefactMaskCleaner
from braindecode.veganlasagne.clip import ClipLayer

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
        'save_folder': './data/models/sacred/paper/bcic-iv-2b/cv-proper-sets/',
        'only_return_exp': False,
        'n_chans': 3,
        }]
    subject_folder_params = dictlistprod({
        'subject_id': range(1,10),
        'data_folder': ['/home/schirrmr/data/bci-competition-iv/2b/'],
    })
    
    exp_params = dictlistprod({
        'run_after_early_stop': [True,],})
    stop_params = dictlistprod({
        'stop_chan': ['misclass']})#, 
    
    loss_params = dictlistprod({
        'loss_expression': ['$tied_loss']})#'misclass', 
    preproc_params = dictlistprod({
        'filt_order': [3,],#10
        'low_cut_hz': [4],
        'sets_like_fbcsp_paper': [False, True]})

    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        exp_params,
        subject_folder_params,
        stop_params,
        preproc_params,
        loss_params,
        ])
    
    return grid_params

def sample_config_params(rng, params):
    return params
        
def run(ex, data_folder, subject_id, n_chans,
    stop_chan, filt_order, low_cut_hz, loss_expression,
        only_return_exp, run_after_early_stop, sets_like_fbcsp_paper):
    start_time = time.time()
    assert (only_return_exp is False) or (n_chans is not None) 
    ex.info['finished'] = False
    
    # trial ivan in milliseconds
    # these are the samples that will be predicted, so for a 
    # network with 2000ms receptive field
    # 1500 means the first receptive field goes from -500 to 1500
    train_segment_ival = [1500,4000]
    test_segment_ival = [1500,4000]
    
    
    add_additional_set = True
    session_ids = [1,2,]
    if sets_like_fbcsp_paper:
        if subject_id in [4,5,6,7,8,9]:
            session_ids = [3] # dummy
            add_additional_set = False
        elif subject_id == 1:
            session_ids = [1,]
        else:
            assert subject_id in [2,3]
            session_ids = [1,2]
    
    train_loader = MultipleBCICompetition4Set2B(subject_id,
        session_ids=session_ids, data_folder=data_folder)
    
    test_loader = MultipleBCICompetition4Set2B(subject_id,
        session_ids=[3], data_folder=data_folder)
    
    # Preprocessing pipeline in [(function, {args:values)] logic
    cnt_preprocessors = [
        (resample_cnt , {'newfs': 250.0}),
        (bandpass_cnt, {
            'low_cut_hz': low_cut_hz,
            'high_cut_hz': 38,
            'filt_order': filt_order,
         }),
         (exponential_standardize_cnt, {})
    ]
    
    marker_def = {'1- Left Hand': [1],  '2 - Right Hand': [2]}
    
    train_signal_proc = SignalProcessor(set_loader=train_loader, 
        segment_ival=train_segment_ival,
                                       cnt_preprocessors=cnt_preprocessors,
                                       marker_def=marker_def)
    train_set = CntSignalMatrix(signal_processor=train_signal_proc, sensor_names='all')
    
    test_signal_proc = SignalProcessor(set_loader=test_loader, 
        segment_ival=test_segment_ival,
        cnt_preprocessors=cnt_preprocessors,
        marker_def=marker_def)
    test_set = CntSignalMatrix(signal_processor=test_signal_proc, sensor_names='all')
    
    train_cleaner = BCICompetitionIV2ABArtefactMaskCleaner(marker_def=marker_def)
    test_cleaner = BCICompetitionIV2ABArtefactMaskCleaner(marker_def=marker_def)
    combined_set = CombinedCleanedSet(train_set, test_set,train_cleaner, test_cleaner)
    if not only_return_exp:
        combined_set.load()
        # only need train set actually, split is done later per fold
        combined_set = combined_set.test_set
        if add_additional_set:
            combined_set.additional_set = train_set
        
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
    n_classes = 2
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
    
    def run_exp(i_fold):
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
        final_layer = ClipLayer(final_layer, 1e-4, 1-1e-4)
        dataset_splitter = CntTrialSingleFoldSplitter(n_folds=10, i_test_fold=i_fold,
            shuffle=True)
        iterator = CntWindowTrialIterator(batch_size=45,input_time_length=input_time_length,
                                         n_sample_preds=get_n_sample_preds(final_layer))
            
        monitors = [LossMonitor(), CntTrialMisclassMonitor(input_time_length=input_time_length),
            KappaMonitor(input_time_length=iterator.input_time_length,
                mode='max'), RuntimeMonitor()]
        
        
        #n_no_decrease_max_epochs = 2
        #n_max_epochs = 4
        n_no_decrease_max_epochs = 80
        n_max_epochs = 800
        # real values for paper were 80 and 800
        remember_best_chan = 'valid_' + stop_chan
        stop_criterion = Or([NoDecrease(remember_best_chan, num_epochs=n_no_decrease_max_epochs),
                             MaxEpochs(num_epochs=n_max_epochs)])
        
        dataset = combined_set
        splitter = dataset_splitter
        updates_expression = adam
        updates_modifier = MaxNormConstraintWithDefaults({})
        preproc = None
        exp = Experiment(final_layer, dataset,splitter,preproc,iterator,
            loss_expression,updates_expression, updates_modifier, monitors, 
                   stop_criterion, remember_best_chan, run_after_early_stop,
                   batch_modifier=None)
    
        if only_return_exp:
            return exp
        
        exp.setup()
        exp.run()
        return exp
    all_monitor_chans = []
    n_folds = 10
    for i_fold in range(n_folds):
        log.info("Running fold {:d} of {:d}".format(i_fold+1, n_folds))
        exp = run_exp(i_fold)
        if only_return_exp:
            return exp
        all_monitor_chans.append(exp.monitor_chans)
    end_time = time.time()
    run_time = end_time - start_time
    
    ex.info['finished'] = True
    keys = all_monitor_chans[0].keys()
    for key in keys:
        ex.info[key] = np.mean([mchans[key][-1] for mchans in 
            all_monitor_chans])
    ex.info['runtime'] = run_time
    save_pkl_artifact(ex, all_monitor_chans, 'all_monitor_chans.pkl')
