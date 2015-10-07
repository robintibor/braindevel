import logging
from glob import glob
import yaml
from pprint import pprint
from numpy.random import RandomState 
import time
import os
from braindecode.datasets.preprocessing import RestrictToTwoClasses,\
    OnlineAxiswiseStandardize
from braindecode.datasets.dataset_splitters import DatasetSingleFoldSplitter,\
    PreprocessedSplitter
from braindecode.datasets.batch_iteration import get_balanced_batches
from braindecode.veganlasagne.monitors import LossMonitor, MisclassMonitor,\
    RuntimeMonitor
import numpy as np
from braindecode.experiments.experiment import Experiment
from braindecode.datasets.bbci_pylearn_dataset import BBCIPylearnCleanDataset
from braindecode.mywyrm.processing import highpass_cnt
from braindecode.mywyrm.clean import BBCISetNoCleaner
from braindecode.veganlasagne.stopping import NoDecrease, Or, MaxEpochs
from braindecode.results.results import Result
import pickle
from braindecode.scripts.print_results import ResultPrinter
import lasagne

class ExperimentsRunner:
    def __init__(self, test=False, start_id=None, stop_id=None, 
        quiet=False, dry_run=False,
        print_results=True):
        self._start_id = start_id
        self._stop_id = stop_id
        self._test = test
        self._quiet = quiet
        self._dry_run = dry_run
        self._do_print_results = print_results
        self._logger = logging.getLogger(__name__)
        
    def run(self, all_train_strs):
        self._all_train_strs = all_train_strs
        print("Running {:d} experiments".format(len(all_train_strs)))
        self._create_base_save_paths_for_all_experiments()
        self._run_all_experiments()
        
    
    def _create_base_save_paths_for_all_experiments(self):
        self._base_save_paths = []
        for i in range(len(self._all_train_strs)):
            save_path = self._create_base_save_path(i)
            self._base_save_paths.append(save_path)

    def _create_base_save_path(self, experiment_index):
        folder_path = self._create_save_folder_path() 
        self._folder_path = folder_path # store for result printing
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
        # Todo: use os.join to prevent mistakes? i.e. than always
        # folder path is a folder, no matter if slash or not at end
        return folder_path + str(result_nr)
    
    def _create_save_folder_path(self):
        train_str = self._all_train_strs[0]
        folder_path = self._load_without_layers(train_str)['save_path']
        if (self._test):
            folder_path += '/test/'
        return folder_path
    
    def _get_model_save_path(self, experiment_index):
        return self._base_save_paths[experiment_index] + ".pkl"

    def _get_result_save_path(self, experiment_index):
        return self._base_save_paths[experiment_index] + ".result.pkl"
        
    def _load_without_layers(self, train_str):
        def do_not_load_constructor(loader, node):
            return None

        yaml.add_constructor(u'!DoNotLoad', do_not_load_constructor)
    
        train_str = train_str.replace('layers: ', 'layers: !DoNotLoad ')
    
        return yaml.load(train_str)

    def _run_all_experiments(self):
        if (self._quiet):
            logging.getLogger("pylearn2").setLevel(logging.WARN)
            logging.getLogger("braindecode").setLevel(logging.WARN)
        
        for i in range(self._get_start_id(),  self._get_stop_id() + 1):
            self._run_experiment(i)           
            
        # lets mkae it faster for tests by not printing... 
        if (not self._dry_run and self._do_print_results):
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
        print("Now running {:d} of {:d}".format(i + 1,
            len(self._all_train_strs)))
        if not(self._dry_run):
            self._run_experiments_with_string(i, train_str)
    
    def _run_experiments_with_string(self, experiment_index, train_str):
        lasagne.random.set_rng(RandomState(9859295))
        self._save_train_string(train_str, experiment_index)
        starttime = time.time()
        
        train_dict = self._load_without_layers(train_str)
        raw_dataset = train_dict['dataset'] 
        raw_dataset.load()
        
        # for now format y back to classes
        raw_dataset.y = np.argmax(raw_dataset.y, axis=1).astype(np.int32)
        
        dataset_provider = PreprocessedSplitter(
            dataset_splitter=DatasetSingleFoldSplitter(raw_dataset, num_folds=10, 
                test_fold_nr=9),
                  preprocessor=OnlineAxiswiseStandardize(axis=('c', 1)))
        
        assert 'in_sensors' in train_str
        assert 'in_rows' in train_str
        assert 'in_cols' in train_str
        
        train_str = train_str.replace('in_sensors',
            str(raw_dataset.get_topological_view().shape[1]))
        train_str = train_str.replace('in_rows',
            str(raw_dataset.get_topological_view().shape[2]))
        train_str = train_str.replace('in_cols', 
            str(raw_dataset.get_topological_view().shape[3]))
        
        train_dict = yaml.load(train_str)
        
        layers = train_dict['layers']
        final_layer = layers[-1]
        
        exp = Experiment()
        exp.setup(final_layer, dataset_provider,
                  loss_var_func=lasagne.objectives.categorical_crossentropy, 
                  updates_var_func=lasagne.updates.adam,
                  batch_iter_func=get_balanced_batches,
                  monitors=[LossMonitor(), MisclassMonitor(), RuntimeMonitor()],
                  stop_criterion=Or(stopping_criteria=[
                    NoDecrease('valid_loss', num_epochs=150, min_decrease=0),
                    MaxEpochs(num_epochs=10)]))
        exp.run()
        
        endtime = time.time()
        result = Result(parameters=train_dict['original_params'],
            templates={}, 
                     training_time=endtime - starttime, 
                     monitor_channels=exp.monitor_chans, 
                     predictions=[0,3,1,2,3,4],
                     targets=[3,4,1,2,3,4])
        
        if not os.path.exists(self._folder_path):
            os.makedirs(self._folder_path)
        
        result_file_name = self._get_result_save_path(experiment_index)
        with open(result_file_name, 'w') as resultfile:
            pickle.dump(result, resultfile)
        
        """
        result_path = self._get_result_save_path(experiment_index)
        # Overwrite shouldn't happen I think?
        serial.save(result_path, result, on_overwrite="ignore")"""

    def _save_train_string(self, train_string, experiment_index):
        file_name = self._base_save_paths[experiment_index] + ".yaml"
        # create folder if necessary
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        yaml_train_file = open(file_name, 'w')
        yaml_train_file.write(train_string)
        yaml_train_file.close()
        
    def _create_result_for_model(self, model, experiment_index, training_time):
        result = Result(
            parameters=self._all_experiments[experiment_index],
            templates=self._templates,
            training_time=training_time,
            monitor_channels=model.monitor.channels,
            predictions=model.info_predictions,
            targets = model.info_targets)
        return result

    def _print_results(self):
        res_printer = ResultPrinter(self._folder_path)
        res_printer.print_results()

    # TODO: remove
    def _store_parameter_info_to_file(self, experiment_index, training_time):
        model_path = self._get_model_save_path(experiment_index)
        model_or_models = serial.load(model_path)
        if (self._cross_validation or self._transfer_learning):
            for model in model_or_models:
                self._store_parameter_info_to_model(model, experiment_index,
                    training_time)
        else:
            model = model_or_models
            self._store_parameter_info_to_model(model, experiment_index, 
                training_time)
        serial.save(model_path, model_or_models, on_overwrite="ignore")
    
    def _store_parameter_info_to_model(self, model, experiment_index,
        training_time):
        parameters = self._all_experiments[experiment_index]
        model.info_templates = self._templates
        model.info_parameters = parameters
        model.training_time = training_time
