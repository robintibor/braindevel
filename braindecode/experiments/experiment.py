import lasagne
from numpy.random import RandomState
import theano
import theano.tensor as T
from collections import OrderedDict
from braindecode.veganlasagne.remember import RememberBest
from braindecode.veganlasagne.stopping import Or, MaxEpochs, ChanBelow
import logging
import numpy as np
from pylearn2.config import yaml_parse
from pylearn2.utils.timing import log_timing
from copy import deepcopy
from braindecode.datahandling.splitters import (SingleFoldSplitter,
    PreprocessedSplitter, FixedTrialSplitter)
from braindecode.veganlasagne.monitors import MonitorManager, MisclassMonitor,\
    LossMonitor, RuntimeMonitor
from braindecode.datahandling.batch_iteration import BalancedBatchIterator
from braindecode.veganlasagne.layers import get_n_sample_preds,\
    get_input_time_length, get_model_input_window
from braindecode.veganlasagne.layer_util import layers_to_str
log = logging.getLogger(__name__)

class ExperimentCrossValidation():
    def __init__(self, final_layer, dataset, exp_args, n_folds, shuffle):
        self.final_layer = final_layer
        self.dataset = dataset
        self.n_folds = n_folds
        self.exp_args = exp_args
        self.shuffle = shuffle
        
    def setup(self):
        lasagne.random.set_rng(RandomState(9859295))

    def run(self):
        self.all_layers = []
        self.all_monitor_chans = []
        for i_fold in range(self.n_folds):
            log.info("Running fold {:d} of {:d}".format(i_fold + 1,
                self.n_folds))
            this_layers = deepcopy(self.final_layer)
            this_exp_args = deepcopy(self.exp_args)
            ## make sure dataset is loaded... 
            self.dataset.ensure_is_loaded()
            dataset_splitter = SingleFoldSplitter(
                n_folds=self.n_folds, i_test_fold=i_fold,
                shuffle=self.shuffle)
            exp = Experiment(this_layers, self.dataset, dataset_splitter, 
                **this_exp_args)
            exp.setup()
            exp.run()
            self.all_layers.append(deepcopy(exp.final_layer))
            self.all_monitor_chans.append(deepcopy(exp.monitor_chans))

def create_default_experiment(final_layer, dataset, n_epochs=100,
        **overwrite_args):
    n_trials = len(dataset.X)
    splitter = FixedTrialSplitter(n_train_trials=n_trials // 2, 
        valid_set_fraction=0.2)
    monitors = [MisclassMonitor(), LossMonitor(),RuntimeMonitor()]
    stop_criterion = MaxEpochs(n_epochs)
    
    exp_args = dict(splitter=splitter,
        preprocessor=None, iterator=BalancedBatchIterator(batch_size=45),
        loss_expression=lasagne.objectives.categorical_crossentropy,
        updates_expression=lasagne.updates.adam,
        updates_modifier=None,
        monitors=monitors, 
        stop_criterion=stop_criterion,
        remember_best_chan='valid_misclass',
        run_after_early_stop=True,
        batch_modifier=None)
    exp_args.update(**overwrite_args)
    
    
    exp = Experiment(final_layer, dataset, **exp_args)
    return exp
    
class Experiment(object):
    def __init__(self, final_layer, dataset, splitter, preprocessor,
            iterator, loss_expression, updates_expression, updates_modifier,
            monitors, stop_criterion, remember_best_chan, run_after_early_stop,
            batch_modifier=None):
        self.final_layer = final_layer
        self.dataset = dataset
        self.dataset_provider = PreprocessedSplitter(splitter, preprocessor)
        self.preprocessor=preprocessor
        self.iterator = iterator
        self.loss_expression = loss_expression
        self.updates_expression = updates_expression
        self.updates_modifier = updates_modifier
        self.monitors = monitors
        self.stop_criterion = stop_criterion
        self.monitor_manager = MonitorManager(monitors)
        self.remember_extension = RememberBest(remember_best_chan)
        self.run_after_early_stop = run_after_early_stop
        self.batch_modifier = batch_modifier
    
    def setup(self, target_var=None):
        lasagne.random.set_rng(RandomState(9859295))
        self.dataset.ensure_is_loaded()
        self.print_layer_sizes()
        log.info("Create theano functions...")
        self.create_theano_functions(target_var)
        # reset remember best extension in case you rerun some experiment
        self.remember_extension = RememberBest(
            self.remember_extension.chan_name)
        log.info("Done.")

    def print_layer_sizes(self):
        log.info("Layers...")
        # start on newline so everything starts from left end of terminal, 
        # including input layer string
        log.info('\n' + layers_to_str(self.final_layer))
    
    def create_theano_functions(self, target_var, deterministic_training=False):
        if target_var is None:
            # get a dummy batch and determine target size
            # use test set since it is smaller
            # maybe memory is freed quicker
            test_set = self.dataset_provider.get_train_valid_test(self.dataset)['test']
            batches = self.iterator.get_batches(test_set, shuffle=False)
            dummy_batch = batches.next()
            dummy_y = dummy_batch[1]
            del test_set
            # for two dims assume we have int targets..
            # maybe could remove these clauses also
            # and just keep else clause
            if dummy_y.ndim == 1:
                print("targets")
                target_var = T.ivector('targets')
            elif dummy_y.ndim == 2:
                target_var = T.imatrix('targets')
            else:
                # tensor with as many dimensions as y
                target_type = T.TensorType(
                    dtype=dummy_y.dtype,
                    broadcastable=[False]*len(self.dataset.y.shape))
                target_var = target_type()
        
        prediction = lasagne.layers.get_output(self.final_layer,
            deterministic=deterministic_training)
        
        # test as in during testing not as in "test set"
        test_prediction = lasagne.layers.get_output(self.final_layer, 
            deterministic=True)
        # Loss function might need layers or not...
        try:
            loss = self.loss_expression(prediction, target_var).mean()
            test_loss = self.loss_expression(test_prediction, target_var).mean()
        except TypeError:
            loss = self.loss_expression(prediction, target_var, self.final_layer).mean()
            test_loss = self.loss_expression(test_prediction, target_var, self.final_layer).mean()
            
        # create parameter update expressions
        params = lasagne.layers.get_all_params(self.final_layer, trainable=True)
        updates = self.updates_expression(loss, params)
        if self.updates_modifier is not None:
            # put norm constraints on all layer, for now fixed to max kernel norm
            # 2 and max col norm 0.5
            updates = self.updates_modifier.modify(updates, self.final_layer)
        input_var = lasagne.layers.get_all_layers(self.final_layer)[0].input_var
        # Store all parameters, including update params like adam params,
        # needed for resetting to best model after early stop
        all_layer_params = lasagne.layers.get_all_params(self.final_layer)
        self.all_params = all_layer_params
        # now params from adam would still be missing... add them ...
        all_update_params = updates.keys()
        for param in all_update_params:
            if param not in self.all_params:
                self.all_params.append(param)

        self.train_func = theano.function([input_var, target_var], updates=updates)
        self.monitor_manager.create_theano_functions(input_var, target_var,
            test_prediction, test_loss)
        
    def run(self):
        log.info("Run until first stop...")
        self.run_until_early_stop()
        # always setup for second stop, in order to get best model
        # even if not running after early stop...
        log.info("Setup for second stop...")
        self.setup_after_stop_training()
        if self.run_after_early_stop:
            log.info("Run until second stop...")
            self.run_until_second_stop()

    def run_until_early_stop(self):
        log.info("Split/Preprocess datasets...")
        datasets = self.dataset_provider.get_train_valid_test(self.dataset)
        log.info("...Done")
        self.create_monitors(datasets)
        self.run_until_stop(datasets, remember_best=True)

    def run_until_stop(self, datasets, remember_best):
        self.monitor_epoch(datasets)
        self.print_epoch()
        if remember_best:
            self.remember_extension.remember_epoch(self.monitor_chans,
                self.all_params)
            
        self.iterator.reset_rng()
        while not self.stop_criterion.should_stop(self.monitor_chans):
            self.run_one_epoch(datasets, remember_best)

    def run_one_epoch(self, datasets, remember_best):
        batch_generator = self.iterator.get_batches(datasets['train'], shuffle=True)
        with log_timing(log, None, final_msg='Time updates following epoch:'):
            for inputs, targets in batch_generator:
                if self.batch_modifier is not None:
                    inputs, targets = self.batch_modifier.process(inputs,
                        targets)
                self.train_func(inputs, targets)
        
        self.monitor_epoch(datasets)
        self.print_epoch()
        if remember_best:
            self.remember_extension.remember_epoch(self.monitor_chans,
                self.all_params)

    def setup_after_stop_training(self):
        self.remember_extension.reset_to_best_model(self.monitor_chans,
                self.all_params)
        loss_to_reach = self.monitor_chans['train_loss'][-1]
        self.stop_criterion = Or(stop_criteria=[
            MaxEpochs(num_epochs=self.remember_extension.best_epoch * 2),
            ChanBelow(chan_name='valid_loss', target_value=loss_to_reach)])
        log.info("Train loss to reach {:.5f}".format(loss_to_reach))
    
    def run_until_second_stop(self):
        datasets = self.dataset_provider.get_train_merged_valid_test(
            self.dataset)
        self.run_until_stop(datasets, remember_best=False)

    def create_monitors(self, datasets):
        self.monitor_chans = OrderedDict()
        self.last_epoch_time = None
        for monitor in self.monitors:
            monitor.setup(self.monitor_chans, datasets)
            
    def monitor_epoch(self, all_datasets):
        self.monitor_manager.monitor_epoch(self.monitor_chans, all_datasets, 
            self.iterator)

    def print_epoch(self):
        # -1 due to doing one monitor at start of training
        i_epoch = len(self.monitor_chans.values()[0]) - 1 
        log.info("Epoch {:d}".format(i_epoch))
        for chan_name in self.monitor_chans:
            log.info("{:25s} {:.5f}".format(chan_name,
                self.monitor_chans[chan_name][-1]))
        log.info("")

def load_layers_from_dict(train_dict):
    """Layers can  be a list or an object that returns a list."""
    layers_obj = train_dict['layers']
    if hasattr(layers_obj, '__len__'):
        return layers_obj
    else:
        return layers_obj.get_layers()

def create_experiment(yaml_filename):
    """Utility function to create experiment from yaml file"""
    # for reproducibility for layer weights
    lasagne.random.set_rng(RandomState(9859295))
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
