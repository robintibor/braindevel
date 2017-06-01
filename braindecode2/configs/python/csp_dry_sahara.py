from braindecode.experiments.parse import product_of_list_of_lists_of_dicts
import pickle
import time
from braindecode.datasets.loaders import BBCIDataset
from braindecode.csp.experiment import CSPExperiment
from braindecode.mywyrm.clean import NoCleaner, ChanMaxAbsVarCleaner
from braindecode.csp.results import CSPResult
import logging
log = logging.getLogger(__name__)

def get_templates():
    return  {
    'abs_var_cleaner': lambda clean_ival: ChanMaxAbsVarCleaner(
        marker_def={'1': [1], '2': [2], '4': [4]},
                    segment_ival=clean_ival),
    'no_cleaner': lambda clean_ival: NoCleaner(
        marker_def={'1': [1], '2': [2], '4': [4]},
                    segment_ival=clean_ival)
    }

def get_grid_param_list():
    default_params = [{'only_create_exp': False,
        'save_path': './data/models/dry-sahara/offline-execution/'
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
    subject_params = [{
        'filename': './data/BBCI-dry-electrodes/offline-execution/MartinS001R01_1-11BBCI.mat',
    }]
    epoching = [{
        'clean_ival': [500,4000],
        'segment_ival': [500, 4000],
        'restricted_n_trials': 540,
        'n_folds': 3,
    }]
    cleaning = [{
        'cleaner': '$abs_var_cleaner',
    },{
        'cleaner': '$no_cleaner',
    },
    ]
    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        filterbank,
        subject_params,
        epoching,
        cleaning,
        ])
    return grid_params

def sample_config_params(rng, params):
    return params

def run(only_create_exp, cleaner, min_freq, max_freq, last_low_freq, low_width,
        high_width, low_overlap, high_overlap, filename, restricted_n_trials,
        n_folds,
        segment_ival, save_base_name, orig_params):
    starttime = time.time()
    marker_def = {'1': [1], '2': [2], '4': [4]}
    
    load_sensor_names = None#['2LA','4Z', '5R'] # None means all
    
    set_loader = BBCIDataset(filename,
                                    load_sensor_names=load_sensor_names)
    
    sensor_names_after_cleaning = None # none means all
    csp_experiment = CSPExperiment(set_loader, 
            sensor_names=sensor_names_after_cleaning,
            cleaner=cleaner,
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
            n_folds=n_folds, 
            n_top_bottom_csp_filters=5, # this number times two will be number of csp filters per filterband before feature selection 
            n_selected_filterbands=None, # how many filterbands to select?
            n_selected_features=20, # how many Features to select with the feature selection?
            forward_steps=2,  # feature selection param
            backward_steps=1, # feature selection param
            stop_when_no_improvement=False, # feature selection param
            only_last_fold=True, # Split into number of folds, but only run the last fold (i.e. last fold as test fold)?
            restricted_n_trials=restricted_n_trials, # restrict to certain number of _clean_ trials?
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
    