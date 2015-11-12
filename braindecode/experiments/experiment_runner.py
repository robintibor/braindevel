import logging
from braindecode.datasets.grasp_lift import KaggleGraspLiftSet,\
    create_submission_csv
log = logging.getLogger(__name__)
from glob import glob
import yaml
from numpy.random import RandomState 
import time
import os
from braindecode.experiments.experiment import Experiment, ExperimentCrossValidation
from braindecode.results.results import Result
import pickle
from braindecode.scripts.print_results import ResultPrinter
import lasagne
from pylearn2.config import yaml_parse
from pprint import pprint
import numpy as np

class ExperimentsRunner:
    def __init__(self, test=False, start_id=None, stop_id=None, 
            quiet=False, dry_run=False, cross_validation=False,
            shuffle=False):
        self._start_id = start_id
        self._stop_id = stop_id
        self._test = test
        self._quiet = quiet
        self._dry_run = dry_run
        self._cross_validation=cross_validation
        self._shuffle=shuffle
        
    def run(self, all_train_strs):
        if (self._quiet):
            self._log_only_warnings()
        self._all_train_strs = all_train_strs
        log.info("Running {:d} experiments".format(len(all_train_strs)))
        self._create_base_save_paths_for_all_experiments()
        self._run_all_experiments()
    
    def _log_only_warnings(self):
        logging.getLogger("pylearn2").setLevel(logging.WARN)
        logging.getLogger("braindecode").setLevel(logging.WARN)
    
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
        train_str = train_str.replace('layers: ', 'layers: !DoNotLoad ')
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
        if not(self._dry_run):
            self._run_experiments_with_string(i, train_str)
    
    def _run_experiments_with_string(self, experiment_index, train_str):
        lasagne.random.set_rng(RandomState(9859295))
        starttime = time.time()
        
        train_dict = self._load_without_layers(train_str)
        log.info("With params...")
        if not self._quiet:
            pprint(train_dict['original_params'])
        dataset = train_dict['dataset'] 
        dataset.load()
        iterator = train_dict['exp_args']['iterator']
        splitter = train_dict['dataset_splitter']
        train_set = splitter.split_into_train_valid_test(dataset)['train']
        batch_gen = iterator.get_batches(train_set, shuffle=True)
        dummy_batch_topo = batch_gen.next()[0]
        dataset_splitter = train_dict['dataset_splitter']

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
            
        layers = train_dict['layers']
        final_layer = layers[-1]
        
        if not self._cross_validation:
            exp = Experiment(final_layer, dataset, dataset_splitter,
                **train_dict['exp_args'])
            exp.setup()
            exp.run()
            endtime = time.time()
            log.info("Saving result...")
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
        with open(result_file_name, 'w') as resultfile:
            pickle.dump(result_or_results, resultfile)

        log.info("Saving model...")
        model_file_name = self._get_model_save_path(experiment_index)
        # Let's save model
        with open(model_file_name, 'w') as modelfile:
            pickle.dump(model, modelfile)
            
        if isinstance(dataset, KaggleGraspLiftSet) and splitter.use_test_as_valid:
            create_submission_csv(self._folder_paths[experiment_index],
                exp.dataset, iterator,
                train_dict['exp_args']['preprocessor'], 
                final_layer, experiment_index + 1)
    
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
