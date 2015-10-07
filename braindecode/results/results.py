from pylearn2.utils import serial
from glob import glob
import os
import numpy as np
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import itertools

class Result:
    """ Empty class for holding result values"""
    def __init__(self, parameters, templates, training_time,
        monitor_channels, predictions, targets):
        self.__dict__.update(locals())
        del self.self
        
    def get_misclasses(self):
        misclasses = {}
        monitor_channels = self.monitor_channels
        for key in monitor_channels:
            if '_misclass' in key:
                misclass_by_epoch = monitor_channels[key]
                keyname = key.replace('_misclass', '')
                misclasses[keyname] = misclass_by_epoch
        return misclasses

def save_result(model_or_models, result_save_path, training_time, parameters={}, 
        templates={}):
    result = None
    if (isinstance(model_or_models, list)):
        result = []
        for model in model_or_models:
            fold_result = _create_result_for_model(model, 
                parameters, templates, training_time)
            result.append(fold_result)
    else:
        model = model_or_models
        result = _create_result_for_model(model, parameters, templates,
            training_time)
    # Overwrite shouldn't happen I think?
    serial.save(result_save_path, result, on_overwrite="ignore")

def _create_result_for_model(model, parameters, templates, training_time):
    result = Result(parameters, templates, training_time, 
        model.monitor_channels, model.info_predictions, model.info_targets)
    return result

class ResultPool():
    def load_results(self, folder_name, start=None, stop=None):
        self._determine_file_names(folder_name, start, stop)
        self._load_result_objects()
        self._collect_results()
        self._collect_parameters()
        self._split_parameters_into_constant_and_varying_parameters()

    def _determine_file_names(self, folder_name, start, stop):
        # get all result_obj file names, should always have digit and then .pkl at end
        # (results have result.pkl instead)
        self._result_file_names = glob(os.path.join(folder_name, '*.result.pkl'))
        # sort numerically by last part of filename (before .pkl extension)
        # i.e. sort bla/12.pkl and bla/1.pkl
        # -> bla/bla/1.pkl, bla/12.pkl
        self._result_file_names.sort(
            key=lambda s:int(s.split('.result.pkl')[0].split('/')[-1]))
        
        if start is not None:
            self._result_file_names = self._result_file_names[start:]
        if stop is not None:
            self._result_file_names = self._result_file_names[:stop]
            

    def _load_result_objects(self):
        self._result_objects = []
        for file_name in self._result_file_names:
            results = serial.load(file_name)
            self._result_objects.append(results)
        if (len(self._result_objects) > 0 and 
                isinstance(self._result_objects[0], list)):
            self._cross_validation = True
        else:
            self._cross_validation = False

    def _collect_results(self):
        self._misclasses  = {}
        self._training_times = []
        result_obj = None
        debug_i = 1
        for result_or_results in self._result_objects:
            if (isinstance(result_or_results, list)):
                result_objs = result_or_results
                misclasses = self._collect_cross_validation_results(result_objs)
                # all result objects should have same parameters/templates
                # so just  take first result object for parameters/templates
                result_obj = result_objs[0]
            else:
                result_obj = result_or_results
                misclasses = self._collect_result(result_obj)
            # Append misclasses from this experiment
            for key in misclasses:
                misclasses_so_far = self._misclasses.get(key, [])
                misclasses_so_far.append(misclasses[key])
                self._misclasses[key] = misclasses_so_far
            try:
                self._training_times.append(result_obj.training_time)
            except:
                print "No Info for {:d}".format(debug_i)
                self._training_times.append(-1)
            debug_i += 1
        # Convert to numpy array and add empty dimension for train_test results
        # making np.mean on axis=1 work for cross val and train_test
        for key in self._misclasses:
            this_misclasses = np.array(self._misclasses[key])
            if (this_misclasses.ndim == 1):
                this_misclasses = np.expand_dims(this_misclasses, 1)
            self._misclasses[key] = this_misclasses

    def _collect_cross_validation_results(self, result_objects):
        """ Collect results from single cross validation"""
        misclasses = {}
        all_misclasses = [self._collect_result(res) for res in result_objects]
        # init keys using first fold (should all be same)
        for key in all_misclasses[0]:
            misclasses[key] = [] 
        for fold_misclass in all_misclasses:
            for key in fold_misclass:
                misclasses[key].append(fold_misclass[key])
        return misclasses

    def _collect_result(self, result_obj):
        misclasses = result_obj.get_misclasses()
        return misclasses
    
    def _collect_parameters(self):
        self._parameters = []
        self._templates = []
        debug_i = 1
        for result_or_results in self._result_objects:
            if (isinstance(result_or_results, list)):
                # all result objects should have same parameters/templates
                # so just  take first result object for parameters/templates
                result_obj = result_or_results[0]
            else:
                result_obj = result_or_results
            try:
                self._templates.append(result_obj.templates)
                self._parameters.append(result_obj.parameters)
            except:
                print "No Info for {:d}".format(debug_i)
                self._templates.append({})
                self._parameters.append({})
            debug_i += 1
        
    def _split_parameters_into_constant_and_varying_parameters(self):
        params = deepcopy(self._parameters)
        varying_params_keys = []
        constant_params = params[0]
        # go through parameters, if parameters with different values
        # appear remove from constant params and add to varying params
        # do same if new parameter appears that does not exist in constant params
        for param_dict in params:
            for param_name, value in param_dict.iteritems():
                if (constant_params.has_key(param_name) and
                    constant_params[param_name] != value):
                    constant_params.pop(param_name)
                    varying_params_keys.append(param_name)
                elif ((not constant_params.has_key(param_name)) and
                      (param_name not in varying_params_keys)):
                    varying_params_keys.append(param_name)
            # also check if all constant params are in this dict, otherwise add them
            missing =  set(constant_params.keys()) - set(param_dict.keys())
            missing = missing - set(varying_params_keys)
            for missing_key in missing:
                varying_params_keys.append(missing_key)
                constant_params.pop(missing_key)
        self._constant_params = constant_params
        # create varying param dicts by removing constant parameters
        # from all/original parameters
        varying_params = deepcopy(self._parameters)
        for paramdict in varying_params:
            for constant_key in self._constant_params:
                # maybe not event presentas a key, thats why ",None"
                paramdict.pop(constant_key, None) 
            for varying_key in varying_params_keys:
                if (not paramdict.has_key(varying_key)):
                    paramdict[varying_key] = "-"
        self._varying_params = varying_params
        
    def have_varying_datasets(self):
        # now all varying paramsshould have same keys
        # so just use first dict to see if there are different filesets
        return ('dataset_filename' in self._varying_params[0].keys() or 
            'filename' in self._varying_params[0].keys() or
            'trainer_filename' in self._varying_params[0].keys())

    def have_varying_leave_out(self):
        return 'transfer_leave_out' in self._varying_params[0].keys()

    def get_misclasses(self):
        """ Get misclassifications as array of dicts (one dict per experiment)"""
        num_experiments = len(self._parameters)
        misclass_array = []
        for i in range(0, num_experiments):
            experiment_misclass = {}
            for key in self._misclasses:
                experiment_misclass[key] = self._misclasses[key][i]
            misclass_array.append(experiment_misclass)
        return misclass_array 
    
    def num_experiments(self):
        return len(self._result_file_names)

    def constant_params(self):
        return self._constant_params

    def varying_params(self):
        return self._varying_params
    
    def result_file_names(self):
        return self._result_file_names
    
    def training_times(self):
        return self._training_times

    def template(self):
        "All templates should be same, so return first one"
        return self._templates[0]

    def result_objects(self):
        return self._result_objects
    
    @staticmethod
    def load_result_objects_for_folder(result_folder):
        resultpool = ResultPool()
        resultpool.load_results(result_folder)
        return resultpool.result_objects()

class DatasetAveragedResults:
    def extract_results(self, result_pool):
        self._result_pool = result_pool
        experiments_same_params = self._extract_experiments_with_same_params()
        results = self.create_results(experiments_same_params)
        self._results = results
        
    def create_results(self, experiments_same_params, ):
        results = []
        for experiment_ids in experiments_same_params:
            result = self._create_results_one_param_set(experiment_ids)
            results.append(result)
        return results
        
    def _extract_experiments_with_same_params(self):
        """ Extract experiment ids of experiments that have the same parameters
        except for the dataset filename or the leave out from transfer."""
        params_without_dataset = deepcopy(self._result_pool.varying_params())
        for params in params_without_dataset:
            params.pop('filename', None)
            params.pop('dataset_filename', None)
            params.pop('transfer_leave_out', None)
            params.pop('test_filename', None)
            params.pop('trainer_filename', None)
        params_to_experiment = {}
        for experiment_i in range(len(params_without_dataset)):
            params = params_without_dataset[experiment_i]
            if str(params) in params_to_experiment:
                params_to_experiment[str(params)].append(experiment_i)
            else:
                params_to_experiment[str(params)] = [experiment_i]
        same_param_ids = params_to_experiment.values()
        # sort so that list of lists is sorted 
        # by lowest experiment id in each list => 
        # appear in same oarder as in original table
        return sorted(same_param_ids, key=np.min)
    
    def _create_results_one_param_set(self, experiment_ids):
        """ Create result for one sequence of experiment ids with same parameters"""
        results = []
        varying_params = self._result_pool.varying_params()
        misclasses = self._result_pool.get_misclasses()
        training_times = self._result_pool.training_times()
        result_objects = self._result_pool.result_objects()
        for experiment_id in experiment_ids:
            this_result =  {'parameters': varying_params[experiment_id],
                'misclasses': misclasses[experiment_id], 
                'training_time': training_times[experiment_id], 
                'result_objects': result_objects[experiment_id]}
            results.append(this_result)
        return results

    def results(self):
        return self._results

    @staticmethod 
    def load_result_objects_for(result_folder, result_nr):
        resultpool = ResultPool()
        resultpool.load_results(result_folder)
        dataset_averaged_pool = DatasetAveragedResults()
        dataset_averaged_pool.extract_results(resultpool)
        results = dataset_averaged_pool.results()
        wanted_results = results[result_nr]
        result_objects = [res['result_objects'] for res in wanted_results]
        return result_objects

def compute_confusion_matrix(result_objects):
    try:
        targets = [fold_res.targets for dataset_result_obj in result_objects for fold_res in dataset_result_obj]
    except:
        targets = [dataset_result_obj.targets for dataset_result_obj in result_objects]
    test_targets = [target['test'] for target in targets]
    test_labels = [np.argmax(test_target, axis=1) for test_target in test_targets]
    test_labels_flat = list(itertools.chain(*test_labels))
    try:
        predictions = [fold_res.predictions for dataset_result_obj in result_objects for fold_res in dataset_result_obj]
    except:
        predictions = [dataset_result_obj.predictions for dataset_result_obj in result_objects]
    test_predictions = [prediction['test'] for prediction in predictions]
    test_predicted_labels = [np.argmax(test_prediction, axis=1) for test_prediction in test_predictions]
    test_predicted_labels_flat = list(itertools.chain(*test_predicted_labels))
    confusion_mat = confusion_matrix(y_true=test_labels_flat, 
                                     y_pred=test_predicted_labels_flat)
    return confusion_mat

def compute_confusion_matrix_csp(result_objects):
    test_labels = [r.multi_class.test_labels for r in result_objects]
    # have to "chain" both folds and datasets thats why two times itertools chain
    # to flatten the list
    # TODELAY: in this case i guess number of folds always same so maybe just wrap with
    # np array and then flatten? instead of itertools chain?
    test_labels_flat = list(itertools.chain(*itertools.chain(*test_labels)))
    test_predicted_labels = [r.multi_class.test_predicted_labels for r in result_objects]
    test_predicted_labels_flat = list(itertools.chain(*itertools.chain(*test_predicted_labels)))
    confusion_mat = confusion_matrix(y_true=test_labels_flat, 
                                     y_pred=test_predicted_labels_flat)
    return confusion_mat