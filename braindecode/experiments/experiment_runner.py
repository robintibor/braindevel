import sys
from glob import glob
import yaml
from numpy.random import RandomState 
import time
import os
import pickle
import lasagne
from pylearn2.config import yaml_parse
from pprint import pprint
import numpy as np
import re
from braindecode.scripts.print_results import ResultPrinter
from braindecode.experiments.experiment import Experiment, ExperimentCrossValidation
from braindecode.results.results import Result
from braindecode.datasets.grasp_lift import (KaggleGraspLiftSet,
    create_submission_csv_for_one_subject, AllSubjectsKaggleGraspLiftSet,
    create_submission_csv_for_all_subject_model)
from braindecode.veganlasagne.layers import (get_n_sample_preds,
    get_model_input_window, get_input_time_length)
from braindecode.veganlasagne.stopping import MaxEpochs
from braindecode.datahandling.splitters import FixedTrialSplitter,\
    SeveralSetsSplitter
import logging
from braindecode.util import dict_equal
log = logging.getLogger(__name__)

class ExperimentsRunner:
    def __init__(self, test=False, start_id=None, stop_id=None, 
            quiet=False, dry_run=False, cross_validation=False,
            shuffle=False, debug=False, only_first_n_sets=False,
            batch_test=False, skip_existing=False):
        self._start_id = start_id
        self._stop_id = stop_id
        self._test = test
        self._quiet = quiet
        self._dry_run = dry_run
        self._cross_validation = cross_validation
        self._shuffle = shuffle
        self._debug = debug
        self._only_first_n_sets = only_first_n_sets
        self._batch_test=batch_test
        self._skip_existing = skip_existing
        
    def run(self, all_train_strs):
        if (self._quiet):
            self._log_only_warnings()
        self._all_train_strs = all_train_strs
        if self._skip_existing:
            self._skip_already_done_experiments()
        log.info("Running {:d} experiments".format(self._get_stop_id() + 1 - 
            self._get_start_id()))
        
        self._create_base_save_paths_for_all_experiments()
        self._run_all_experiments()
    
    def _log_only_warnings(self):
        logging.getLogger("pylearn2").setLevel(logging.WARN)
        logging.getLogger("braindecode").setLevel(logging.WARN)
    
    def _skip_already_done_experiments(self):
        log.info("Check if some experiments were already run...")
        clean_all_train_strs = []
        # Go through all experiments, all result folders
        # and check if experiments already exist
        
        # First collect all folder paths and load results
        # in order not to load results twice
        # Possible optimization: First get all save paths
        # Then only load results once for respective save paths
        all_folder_paths = []
        for i_experiment in range(len(self._all_train_strs)):
            folder_path = self._create_save_folder_path(i_experiment)
            all_folder_paths.append(folder_path)
        
        unique_folder_paths = set(all_folder_paths)
        folder_path_to_results = dict()
        for folder_path in unique_folder_paths:
            existing_result_files = glob(folder_path + "*[0-9].result.pkl")
            results = [np.load(f) for f in existing_result_files]
            folder_path_to_results[folder_path] = results
        
        
        for i_experiment in range(len(self._all_train_strs)):
            train_str = self._all_train_strs[i_experiment]
            train_dict = self._load_without_layers(train_str)
            original_params = train_dict['original_params']
            folder_path = all_folder_paths[i_experiment]
            results = folder_path_to_results[folder_path]
            experiment_already_run = False
            for r in results:
                if dict_equal(r.parameters, original_params):
                    experiment_already_run = True
                    log.warn("Already ran id {:d} {:s}".format(i_experiment,
                        str(original_params)))
            if not experiment_already_run:
                clean_all_train_strs.append(train_str)
    
        self._all_train_strs = clean_all_train_strs
    
    def _create_base_save_paths_for_all_experiments(self):
        self._base_save_paths = []
        self._folder_paths = [] # will be set inside function for later result printing
        for i in range(len(self._all_train_strs)):
            save_path = self._create_base_save_path(i)
            self._base_save_paths.append(save_path)

    def _create_base_save_path(self, experiment_index):
        folder_path = self._create_save_folder_path(experiment_index) 
        self._folder_paths.append(folder_path) # store for result printing
        result_nr = experiment_index + 1
        # try not to overwrite existing models, instead
        # use higher numbers
        existing_result_files = glob(folder_path + "*[0-9].result.pkl")
        if (len(existing_result_files) > 0):
            # model nrs are last part of file name before .pkl
            existing_result_nrs = [int(file_name.split('/')[-1][:-len('.result.pkl')])\
                for file_name in existing_result_files]
            highest_result_nr = max(existing_result_nrs)
            result_nr = highest_result_nr + result_nr
        return os.path.join(folder_path, str(result_nr))
    
    def _create_save_folder_path(self, experiment_index):
        train_str = self._all_train_strs[experiment_index]
        folder_path = self._load_without_layers(train_str)['save_path']
        if (self._test):
            folder_path += '/test/'
        if (self._cross_validation):
            folder_path += '/cross-validation/'
        if (self._shuffle):
            folder_path += '/shuffle/'
        if (self._debug):
            folder_path += '/debug/'
        if (self._only_first_n_sets is not False):
            folder_path += '/first{:d}/'.format(self._only_first_n_sets)
            
        return folder_path
    
    def _get_model_save_path(self, experiment_index):
        return self._base_save_paths[experiment_index] + ".pkl"

    def _get_result_save_path(self, experiment_index):
        return self._base_save_paths[experiment_index] + ".result.pkl"
        
    @staticmethod
    def _load_without_layers(train_str):
        def do_not_load_constructor(loader, node):
            return None
        yaml.add_constructor(u'!DoNotLoad', do_not_load_constructor)
        # replace layers by layers with a tag not to load it after...
        # have to replace any different tag there might be ...
        train_str = re.sub("layers:[\s]+!obj:[^{]*", "layers: !DoNotLoad ", train_str)
        train_str = train_str.replace("layers: [", "layers: !DoNotLoad [")

        return yaml_parse.load(train_str)

    def _run_all_experiments(self):
        
        for i in range(self._get_start_id(),  self._get_stop_id() + 1):
            self._run_experiment(i)           
            
        if (not self._dry_run and (not self._quiet)):
            self._print_results()
    
    def _get_start_id(self):
        # Should have created experiments before
        assert(self._all_train_strs is not None)
        return 0 if self._start_id is None else self._start_id 
        
    def _get_stop_id(self):
        # Should have created experiments before
        assert(self._all_train_strs is not None)
        num_experiments = len(self._all_train_strs)
        return num_experiments - 1 if self._stop_id is None else self._stop_id

    def _run_experiment(self, i):
        train_str = self._all_train_strs[i]
        log.info("Now running {:d} of {:d}".format(i + 1, self._get_stop_id() + 1))
        self._run_experiments_with_string(i, train_str)
    
    def _run_experiments_with_string(self, experiment_index, train_str):
        lasagne.random.set_rng(RandomState(9859295))
        # Save train string now, will be overwritten later after 
        # input dimensions determined, save now for debug in
        # case of crash
        if not self._dry_run:
            self._save_train_string(train_str, experiment_index)
        starttime = time.time()
        
        train_dict = self._load_without_layers(train_str)
        log.info("With params...")
        if not self._quiet:
            pprint(train_dict['original_params'])
        if self._dry_run:
            # Do not do the loading or training...
            # Only go until here to show the train params
            return
        
        if self._batch_test:
        # TODO: put into function
        # load layers, load data with dimensions of the layer
        # create experiment with max epochs 2, run
            from braindecode.datasets.random import RandomSet
            train_str = train_str.replace('in_cols', '1')
            train_str = train_str.replace('in_sensors', '32')
            train_dict =  yaml_parse.load(train_str)
            layers = load_layers_from_dict(train_dict)
            final_layer = layers[-1]
            n_chans = layers[0].shape[1]
            n_classes = final_layer.output_shape[1]
            n_samples = 500000
            # set n sample perds in case of cnt model
            if (np.any([hasattr(l, 'n_stride') for l in layers])):
                n_sample_preds =  get_n_sample_preds(final_layer)
                log.info("Setting n_sample preds automatically to {:d}".format(
                    n_sample_preds))
                for monitor in train_dict['exp_args']['monitors']:
                    if hasattr(monitor, 'n_sample_preds'):
                        monitor.n_sample_preds = n_sample_preds
                train_dict['exp_args']['iterator'].n_sample_preds = n_sample_preds
                log.info("Input window length is {:d}".format(
                    get_model_input_window(final_layer)))
                # make at least batches
                n_samples = int(n_sample_preds * 1.5 * 200)
            dataset = RandomSet(topo_shape=[n_samples, n_chans, 1, 1], 
                y_shape=[n_samples, n_classes]) 
            dataset.load()
            splitter = FixedTrialSplitter(n_train_trials=int(n_samples*0.8), 
                valid_set_fraction=0.1)
            train_dict['exp_args']['preprocessor'] = None
            train_dict['exp_args']['stop_criterion'] = MaxEpochs(1)
            train_dict['exp_args']['iterator'].batch_size = 1
            # TODO: set stop criterion to max epochs =1
            #  change batch_size in iterator
            exp = Experiment(final_layer, dataset, splitter,
                **train_dict['exp_args'])
            exp.setup()
            exp.run_until_early_stop()
            datasets = exp.dataset_provider.get_train_valid_test(exp.dataset)
            for batch_size in range(32,200,5):
                train_dict['exp_args']['stop_criterion'].num_epochs += 2
                log.info("Running with batch size {:d}".format(batch_size))
                train_dict['exp_args']['iterator'].batch_size = batch_size
                exp.run_until_stop(datasets, remember_best=False)
            return
            
            
        dataset = train_dict['dataset'] 
        dataset.load()
        iterator = train_dict['exp_args']['iterator']
        splitter = train_dict['dataset_splitter']
        train_set = splitter.split_into_train_valid_test(dataset)['train']
        batch_gen = iterator.get_batches(train_set, shuffle=True)
        dummy_batch_topo = batch_gen.next()[0]
        del train_set

        assert 'in_sensors' in train_str
        #not for cnt net assert 'in_rows' in train_str
        assert 'in_cols' in train_str
        
        train_str = train_str.replace('in_sensors',
            str(dummy_batch_topo.shape[1]))
        train_str = train_str.replace('in_rows',
            str(dummy_batch_topo.shape[2]))
        train_str = train_str.replace('in_cols', 
            str(dummy_batch_topo.shape[3]))
        
        self._save_train_string(train_str, experiment_index)
        
        
        # reset rng for actual loading of layers, so you can reproduce it 
        # when you load the file later
        lasagne.random.set_rng(RandomState(9859295))
        train_dict =  yaml_parse.load(train_str)
            
        layers = load_layers_from_dict(train_dict)
        final_layer = layers[-1]
        assert len(np.setdiff1d(layers, 
            lasagne.layers.get_all_layers(final_layer))) == 0, ("All layers "
            "should be used, unused {:s}".format(str(np.setdiff1d(layers, 
            lasagne.layers.get_all_layers(final_layer)))))
        # Set n sample preds in case of cnt model
        if (np.any([hasattr(l, 'n_stride') for l in layers])):
            # Can this be moved up and duplication in if clause( batch test,
            # more above) be removed?
            n_sample_preds =  get_n_sample_preds(final_layer)
            log.info("Setting n_sample preds automatically to {:d}".format(
                n_sample_preds))
            for monitor in train_dict['exp_args']['monitors']:
                if hasattr(monitor, 'n_sample_preds'):
                    monitor.n_sample_preds = n_sample_preds
            train_dict['exp_args']['iterator'].n_sample_preds = n_sample_preds
            log.info("Input window length is {:d}".format(
                get_model_input_window(final_layer)))
        
        if not self._cross_validation:
            exp = Experiment(final_layer, dataset, splitter,
                **train_dict['exp_args'])
            exp.setup()
            exp.run()
            endtime = time.time()
            result_or_results = Result(parameters=train_dict['original_params'],
                templates={}, 
                training_time=endtime - starttime, 
                monitor_channels=exp.monitor_chans, 
                predictions=[0,3,1,2,3,4],
                targets=[3,4,1,2,3,4])
            model = exp.final_layer
        else: # cross validation
            # default 5 folds for now
            n_folds = train_dict['num_cv_folds']
            exp_cv = ExperimentCrossValidation(final_layer, 
                dataset, exp_args=train_dict['exp_args'], n_folds=n_folds,
                shuffle=self._shuffle)
            exp_cv.run()
            endtime = time.time()
            result_or_results = []
            for i_fold in xrange(n_folds):
                res = Result(parameters=train_dict['original_params'],
                templates={}, 
                training_time=endtime - starttime, 
                monitor_channels=exp_cv.all_monitor_chans[i_fold], 
                predictions=[0,3,1,2,3,4],
                targets=[3,4,1,2,3,4])
                result_or_results.append(res)
            model = exp_cv.all_layers
            
        if not os.path.exists(self._folder_paths[experiment_index]):
            os.makedirs(self._folder_paths[experiment_index])
        
        result_file_name = self._get_result_save_path(experiment_index)
        
        log.info("Saving result...")
        with open(result_file_name, 'w') as resultfile:
            pickle.dump(result_or_results, resultfile)

        log.info("Saving model...")
        model_file_name = self._get_model_save_path(experiment_index)
        # Let's save model
        # set recursion limit very high to avoid problems saving
        # see 
        #https://github.com/lisa-lab/pylearn2/blob/74fd21b77f24620de442768cd15f22ad06c7fa2c/pylearn2/utils/serial.py#L102-L124
        # and http://stackoverflow.com/questions/2134706/hitting-maximum-recursion-depth-using-pythons-pickle-cpickle
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(50000)
        with open(model_file_name, 'w') as modelfile:
            pickle.dump(model, modelfile)
        sys.setrecursionlimit(old_limit)
        
        param_file_name = model_file_name.replace('.pkl', '.npy')
        np.save(param_file_name, lasagne.layers.get_all_param_values(model))
        
        # Possibly make kaggle submission file
        if isinstance(dataset, KaggleGraspLiftSet) and splitter.use_test_as_valid:
            experiment_save_id = int(
                self._base_save_paths[experiment_index].split("/")[-1])
            create_submission_csv_for_one_subject(self._folder_paths[experiment_index],
                exp.dataset, iterator,
                train_dict['exp_args']['preprocessor'], 
                final_layer, experiment_save_id)
        elif isinstance(dataset, AllSubjectsKaggleGraspLiftSet) and splitter.use_test_as_valid:
            experiment_save_id = int(
                self._base_save_paths[experiment_index].split("/")[-1])
            create_submission_csv_for_all_subject_model(
                self._folder_paths[experiment_index],
                exp.dataset, exp.dataset_provider, iterator,
                final_layer, experiment_save_id)
        elif isinstance(splitter, SeveralSetsSplitter):
            pass # nothing to do in this case
        elif hasattr(splitter, 'use_test_as_valid') and splitter.use_test_as_valid:
            raise ValueError("Splitter has use test as valid set, but unknown dataset type" 
                "" + str(dataset.__class__.__name__))
            
    
        
         
    def _save_train_string(self, train_string, experiment_index):
        file_name = self._base_save_paths[experiment_index] + ".yaml"
        # create folder if necessary
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        yaml_train_file = open(file_name, 'w')
        yaml_train_file.write(train_string)
        yaml_train_file.close()

    def _print_results(self):
        for folder_path in np.unique(self._folder_paths):
            res_printer = ResultPrinter(folder_path)
            res_printer.print_results()
            print("\n")


def load_layers_from_dict(train_dict):
    """Layers can  be a list or an object that returns a list."""
    layers_obj = train_dict['layers']
    if hasattr(layers_obj, '__len__'):
        return layers_obj
    else:
        return layers_obj.get_layers()

def create_experiment(yaml_filename):
    """Utility function to create experiment from yaml file"""
    train_dict = yaml_parse.load(open(yaml_filename, 'r'))
    layers = load_layers_from_dict(train_dict)
    final_layer = layers[-1]
    dataset = train_dict['dataset'] 
    splitter = train_dict['dataset_splitter']
    if (np.any([hasattr(l, 'n_stride') for l in layers])):
        n_sample_preds =  get_n_sample_preds(final_layer)
        # for backwards compatibility input time length also
        input_time_length = get_input_time_length(final_layer)
        log.info("Setting n_sample preds automatically to {:d}".format(
            n_sample_preds))
        for monitor in train_dict['exp_args']['monitors']:
            if hasattr(monitor, 'n_sample_preds'):
                monitor.n_sample_preds = n_sample_preds
            if hasattr(monitor, 'input_time_length'):
                monitor.input_time_length = input_time_length
                
        train_dict['exp_args']['iterator'].n_sample_preds = n_sample_preds
        log.info("Input window length is {:d}".format(
            get_model_input_window(final_layer)))
    # add early stop chan, encessary for backwards compatibility
    exp_args = train_dict['exp_args']
    exp_args['remember_best_chan'] = train_dict['exp_args'].pop('remember_best_chan',
        'valid_misclass')
    exp_args['run_after_early_stop'] = train_dict['exp_args'].pop('run_after_early_stop',
        True)
    exp = Experiment(final_layer, dataset, splitter,
                    **exp_args)
    assert len(np.setdiff1d(layers, 
        lasagne.layers.get_all_layers(final_layer))) == 0, ("All layers "
        "should be used, unused {:s}".format(str(np.setdiff1d(layers, 
        lasagne.layers.get_all_layers(final_layer)))))
    return exp
