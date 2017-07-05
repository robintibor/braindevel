from braindecode.experiments.parse import product_of_list_of_lists_of_dicts,\
    cartesian_dict_of_lists_product
import pickle
import time
from braindecode.datasets.loaders import MultipleBBCIDataset
from braindecode.csp.experiment import TwoFileCSPExperiment
from braindecode.mywyrm.clean import NoCleaner, ChanMaxAbsVarCleaner
from braindecode.csp.results import CSPResult
import logging
log = logging.getLogger(__name__)

def get_templates():
    return  {
    'martin_train_new': 
    [
        'data/dry-cap-ilmenau/MaVoMoSc2_-_2017-01-13_10-09-54_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/MaVoMoSc2_-_2017-01-13_10-22-34_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/MaVoMoSc2_-_2017-01-13_10-38-37_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/MaVoMoSc2_-_2017-01-13_10-51-20_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/MaVoMoSc2_-_2017-01-13_11-05-13_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/MaVoMoSc2_-_2017-01-13_11-29-01_1-1_500Hz.BBCI.mat',
     ],
     'martin_test_new': [
        'data/dry-cap-ilmenau/MaVoMoSc2_-_2017-01-13_11-46-32_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/MaVoMoSc2_-_2017-01-13_11-58-15_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/MaVoMoSc2_-_2017-01-13_12-16-52_1-1_500Hz.BBCI.mat',
    ],
    'roland_train_new': 
        [
        'data/dry-cap-ilmenau/RoBeMoSc4_-_2017-01-12_13-25-00_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/RoBeMoSc4_-_2017-01-12_14-23-48_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/RoBeMoSc4_-_2017-01-12_14-33-08_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/RoBeMoSc4_-_2017-01-12_14-46-02_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/RoBeMoSc4_-_2017-01-12_15-04-04_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/RoBeMoSc4_-_2017-01-12_15-14-08_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/RoBeMoSc4_-_2017-01-12_15-36-07_1-1_500Hz.BBCI.mat'
     ],
     'roland_test_new': [
        'data/dry-cap-ilmenau/RoBeMoSc4_-_2017-01-12_15-54-01_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/RoBeMoSc4_-_2017-01-12_16-11-13_1-1_500Hz.BBCI.mat',
        'data/dry-cap-ilmenau/RoBeMoSc4_-_2017-01-12_16-24-31_1-1_500Hz.BBCI.mat',
    ],
    'abs_var_cleaner': lambda clean_marker_def, clean_ival: ChanMaxAbsVarCleaner(
        marker_def=clean_marker_def,
                    segment_ival=clean_ival),
    'no_cleaner': lambda clean_marker_def, clean_ival: NoCleaner(
        marker_def=clean_marker_def, segment_ival=clean_ival),
    'without_rest': {'1 - Left Hand': [56], '2 - Right Hand': [88], 
                    '3 - Feet': [4]},
    'with_rest': {'1 - Left Hand': [56], '2 - Right Hand': [88], 
                    '3 - Feet': [4], '4 - Rest': [120]},
    }

def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{'only_create_exp': False,
        'save_path': './data/models/dry-ilmenau/csp/'
        }]

    filterbank = [
        #gamma
        {
            'min_freq': 60,
            'max_freq': 96,
            'last_low_freq': 60,
            'low_width': 6,
            'high_width': 8,
            'low_overlap': 3,
            'high_overlap': 4,
        },
        # alphabeta
        {
            'min_freq': 2,
            'max_freq': 35,
            'last_low_freq': 35,
            'low_width': 6,
            'high_width': 8,
            'low_overlap': 3,
            'high_overlap': 4,
        },
        # paper
        {
            'min_freq': 1,
            'max_freq': 118,
            'last_low_freq': 10,
            'low_width': 6,
            'high_width': 8,
            'low_overlap': 3,
            'high_overlap': 4,
        },
    ]
    subject_params = [
        {
        'train_filenames': '$martin_train_new',
        'test_filenames': '$martin_test_new'
    }, 
    {
        'train_filenames': '$roland_train_new',
        'test_filenames': '$roland_test_new'
    }
    ]
    ival = [{
        'clean_ival': [-3500, 0],
        'segment_ival': [-3500, 0],
    }]
    markers = [{
        'marker_def': '$with_rest',
        'clean_marker_def': '$with_rest',
        },{
        'marker_def': '$without_rest',
        'clean_marker_def': '$without_rest',
    }]
    cleaning = dictlistprod({
        'cleaner': ['$no_cleaner','$abs_var_cleaner'],
    })
    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        filterbank,
        subject_params,
        ival,
        markers,
        cleaning,
        ])
    return grid_params

def sample_config_params(rng, params):
    return params

def run(only_create_exp, cleaner, min_freq, max_freq, last_low_freq, low_width,
        high_width, low_overlap, high_overlap, train_filenames,
        test_filenames, marker_def, segment_ival, save_base_name, orig_params):
    starttime = time.time()
    
    
    load_sensor_names = None#['2LA','4Z', '5R', '0Z', '2Z', '3Z'] # None means all
    
    set_loader = MultipleBBCIDataset(train_filenames,
                                    load_sensor_names=load_sensor_names)
    test_set_loader = MultipleBBCIDataset(test_filenames,
                                         load_sensor_names=load_sensor_names)
    
    sensor_names_after_cleaning = None # none means all
    csp_experiment = TwoFileCSPExperiment(set_loader, test_set_loader,
            sensor_names=sensor_names_after_cleaning,
            train_cleaner=cleaner, test_cleaner=cleaner,
            resample_fs=250,
            min_freq=min_freq,
            max_freq=max_freq,
            last_low_freq=last_low_freq,
            low_width=low_width,
            high_width=high_width,
            low_overlap=low_overlap,
            high_overlap=high_overlap,
            filt_order=4,
            segment_ival=segment_ival, # trial interval
            standardize_filt_cnt=False,
            standardize_epo=False, # standardize the data?..
            n_folds=None, 
            n_top_bottom_csp_filters=5, # this number times two will be number of csp filters per filterband before feature selection 
            n_selected_filterbands=None, # how many filterbands to select?
            n_selected_features=20, # how many Features to select with the feature selection?
            forward_steps=2,  # feature selection param
            backward_steps=1, # feature selection param
            stop_when_no_improvement=False, # feature selection param
            only_last_fold=True, # Split into number of folds, but only run the last fold (i.e. last fold as test fold)?
            restricted_n_trials=None, # restrict to certain number of _clean_ trials?
            common_average_reference=False,
            ival_optimizer=None, # optimize the trial ival with some optimizer?
            shuffle=False, # shuffle or do blockwise folds?
            marker_def=marker_def)
    result_file_name = save_base_name + '.result.pkl'
    csp_experiment.run()
    endtime = time.time()
    result = CSPResult(
            csp_trainer=csp_experiment,
            parameters=orig_params,
            training_time=endtime - starttime)
    with open(result_file_name, 'w') as resultfile:
        log.info("Saving to {:s}...\n".format(result_file_name))
        pickle.dump(result, resultfile)   
    