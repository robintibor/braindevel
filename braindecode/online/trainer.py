import lasagne
import numpy as np
import theano
import os
from glob import glob
import datetime
import theano.tensor as T
from lasagne.updates import adam
from numpy.random import RandomState
from braindecode.veganlasagne.layers import (get_input_time_length,
    get_n_sample_preds, get_input_var)
from braindecode.datahandling.batch_iteration import (create_batch, 
    get_start_end_blocks_for_trial)
from braindecode.util import FuncAndArgs
import logging
from pylearn2.utils.timing import log_timing
from braindecode.datahandling.preprocessing import exponential_running_standardize
from braindecode.experiments.load import set_param_values_backwards_compatible
log = logging.getLogger(__name__)

class BatchWiseCntTrainer(object):
    def __init__(self, exp, n_updates_per_break, batch_size, learning_rate,
                n_min_trials, trial_start_offset, break_start_offset,
                break_stop_offset,
                train_param_values,
                deterministic_training=False, add_breaks=True):
        self.cnt_model = exp.final_layer
        self.exp = exp
        self.n_updates_per_break = n_updates_per_break
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_min_trials = n_min_trials
        self.trial_start_offset = trial_start_offset
        self.break_start_offset = break_start_offset
        self.break_stop_offset = break_stop_offset
        self.train_param_values = train_param_values
        self.deterministic_training = deterministic_training
        self.add_breaks = add_breaks
        
    def set_predicting_model(self, model):
        """ Needed to keep trained and used params in sync, i.e.
        Update the params of the epo model used for prediction
        with those params of the trained cnt model."""
        self.predicting_model = model
        
    def set_data_processor(self, data_processor):
        self.data_processor = data_processor

    def set_marker_buffer(self, marker_buffer):
        self.marker_buffer = marker_buffer
        
    def initialize(self):
        """ Initialize data containers and theano functions for training."""
        self.rng = RandomState(30948348)
        self.data_batches = []
        self.y_batches = []
        self.input_time_length = get_input_time_length(self.cnt_model)
        self.n_sample_preds = get_n_sample_preds(self.cnt_model)
        self.n_classes = self.cnt_model.output_shape[1]
        # create train function
        log.info("Compile train function...")
        self._create_train_function()
        log.info("Done compiling train function.")
        
    def _create_train_function(self):
        # Maybe replace self.exp.final_layer by self.cnt_model?
        # not clear to me why I am using self.exp.final_layer here 
        targets = T.ivector()
        input_var = get_input_var(self.exp.final_layer)
        updates_expression = FuncAndArgs(adam, learning_rate=self.learning_rate)
        prediction = lasagne.layers.get_output(self.exp.final_layer,
            deterministic=self.deterministic_training, input_var=input_var,
            inputs=input_var)
        # Loss function might need layers or not...
        try:
            loss = self.exp.loss_expression(prediction, targets).mean()
        except TypeError:
            loss = self.exp.loss_expression(prediction, targets,
                self.exp.final_layer).mean()
        # create parameter update expressions
        params = lasagne.layers.get_all_params(self.exp.final_layer,
            trainable=True)
        updates = updates_expression(loss, params)
        if self.exp.updates_modifier is not None:
            # put norm constraints on all layer, for now fixed to max kernel norm
            # 2 and max col norm 0.5
            updates = self.exp.updates_modifier.modify(updates,
                self.exp.final_layer)
            
        # store only the parameters for training,
        # assumes parameters for layers already set
        self.train_params = []
        all_update_params = updates.keys()
        for update_param in all_update_params:
            if update_param not in params:
                self.train_params.append(update_param)
        
        self.train_func = theano.function([input_var, targets], updates=updates)
        
        # Set optimizer/train parameter values if not done
        if self.train_param_values is not None:
            log.info("Setting train parameter values")
            for param, val in zip(self.train_params, self.train_param_values):
                param.set_value(val)
            log.info("...Done setting parameter train values")
        else:
            log.info("Not setting train parameter values, optimization values "
            "start from scratch (model params may be loaded anyways.)")
            
    def add_data_from_today(self, data_processor):
        # Check if old data exists, if yes add it
        now = datetime.datetime.now()
        day_string = now.strftime('%Y-%m-%d')
        data_folder = 'data/online/{:s}'.format(day_string)
        # sort should sort timewise for our timeformat...
        data_files = sorted(glob(os.path.join(data_folder, '*.npy')))
        if len(data_files) > 0:
            log.info("Loading {:d} data files for adaptation:\n{:s}".format(
                len(data_files), str(data_files)))
            for filename in data_files:
                log.info("Add data from {:s}...".format(filename))
                samples_markers = np.load(filename)
                samples = samples_markers[:,:-1]
                markers = np.int32(samples_markers[:,-1])
                self.add_training_blocks_from_old_data(samples, markers,
                    data_processor)
            log.info("Done loading, now have {:d} trials (including breaks)".format(
                len(self.data_batches)))
        else:
            log.info("No data files found to load for adaptation in {:s}".format(
                data_folder))

    def add_training_blocks_from_old_data(self, old_samples,
            old_markers, data_processor):
        # first standardize data
        old_samples = exponential_running_standardize(old_samples, 
            factor_new=data_processor.factor_new, init_block_size=1000, 
            eps=data_processor.eps)
        trial_starts, trial_stops = self.get_trial_start_stop_indices(
                old_markers)
        log.info("Adding {:d} trials".format(len(trial_starts)))
        for trial_start, trial_stop in zip(trial_starts, trial_stops):
            self.add_blocks(trial_start + self.trial_start_offset, 
                trial_stop, old_samples, old_markers)
        # now lets add breaks
        log.info("Adding {:d} breaks".format(len(trial_starts) - 1))
        for break_start, break_stop in zip(trial_stops[:-1], trial_starts[1:]):
            self.add_break(break_start, break_stop, old_samples, old_markers)

    def process_markers(self, markers):
        # Check if a trial has ended with last samples
        # need marker samples with some overlap
        # so we do not miss trial boundaries inbetween two sample blocks
        marker_samples_with_overlap = np.copy(
            self.marker_buffer[-len(markers)-2:])
        trial_has_ended = np.sum(np.diff(marker_samples_with_overlap) < 0) > 0
        if trial_has_ended:
            trial_starts, trial_stops = self.get_trial_start_stop_indices(
                self.marker_buffer)
            trial_start = trial_starts[-1]
            trial_stop = trial_stops[-1]
            log.info("Trial has ended for class {:d}".format(
                self.marker_buffer[trial_start]))
            assert trial_start < trial_stop, ("trial start {:d} should be "
                "before trial stop {:d}, markers: {:s}").format(trial_start, 
                    trial_stop, str(marker_samples_with_overlap))
            self.add_blocks(trial_start + self.trial_start_offset, trial_stop,
                self.data_processor.sample_buffer,
                self.marker_buffer)
            log.info("Now {:d} trials (including breaks)".format(
                len(self.data_batches)))
            
            with log_timing(log, None, final_msg='Time for training:'):
                self.train()
        trial_has_started = np.sum(np.diff(marker_samples_with_overlap) > 0) > 0
        if trial_has_started:
            trial_end_in_marker_buffer = np.sum(np.diff(self.marker_buffer) < 0) > 0
            if trial_end_in_marker_buffer:
                # +1 necessary since diff removes one index
                trial_start = np.flatnonzero(np.diff(self.marker_buffer) > 0)[-1] + 1
                trial_stop = np.flatnonzero(np.diff(self.marker_buffer) < 0)[-1] + 1
                assert trial_start > trial_stop, ("If trial has just started "
                    "expect this to be after stop of last trial")
                self.add_break(break_start=trial_stop, break_stop=trial_start,
                    all_samples=self.data_processor.sample_buffer,
                    all_markers=self.marker_buffer)
            #log.info("Break added, now at {:d} batches".format(len(self.data_batches)))
                
    def add_break(self, break_start, break_stop, all_samples, all_markers):
        if self.add_breaks:
            all_markers = np.copy(all_markers)
            assert np.all(all_markers[break_start:break_stop] == 0)
            assert all_markers[break_start - 1] != 0
            assert all_markers[break_stop] != 0
            # keep n_classes for 1-based matlab indexing logic in markers
            all_markers[break_start:break_stop] = self.n_classes
            self.add_blocks(break_start + self.break_start_offset, 
                break_stop + self.break_stop_offset, all_samples,
                all_markers)
        else:
            pass #Ignore break that was supposed to be added

    def get_trial_start_stop_indices(self, markers):
        # + 1 as diff "removes" one index, i.e. diff will be above zero
            # at the index 1 before the increase=> the trial start
        trial_starts = np.flatnonzero(np.diff(markers) > 0) + 1
        # diff removing index, so this index is last sample of trial
        # but stop indices in python are exclusive so +1
        trial_stops = np.flatnonzero(np.diff(markers) < 0) + 1

        if trial_starts[0] >= trial_stops[0]:
            # cut out first trial which only has end marker
            trial_stops = trial_stops[1:]
        if trial_starts[-1] >= trial_stops[-1]:
            # cut out last trial which only has start marker
            trial_starts = trial_starts[:-1]
        
        assert(len(trial_starts) == len(trial_stops))
        assert(np.all(trial_starts <= trial_stops))
        return trial_starts, trial_stops
    
    def add_blocks(self, trial_start, trial_stop, all_samples, all_markers):
        """Trial start offset as parameter to give different offsets
        for break and normal trial."""
        # n_sample_preds is how many predictions done for
        # one forward pass of the network -> how many crops predicted
        # together in one forward pass for given input time length of 
        # the ConvNet
        # -> crop size is how many samples are needed for one prediction
        crop_size = self.input_time_length - self.n_sample_preds + 1
        if trial_start + self.n_sample_preds > trial_stop:
            log.info("Too little data in this trial to train in it, only "
                "{:d} predictable samples, need atleast {:d}".format(
                     trial_stop - trial_start, self.n_sample_preds))
            return # Too little data in this trial to train on it...
        needed_sample_start = trial_start - crop_size + 1
        # not sure if copy necessary, but why not :)
        needed_samples = np.copy(all_samples[needed_sample_start:trial_stop])
        trial_markers = all_markers[needed_sample_start:trial_stop]
        # trial start can't be at zero atm or else we would have to take more data
        assert (len(np.unique(trial_markers[(crop_size - 1):])) == 1), (
            ("Trial should have exactly one class, markers: {:s} "
                "trial start: {:d}, trial_stop: {:d}").format(
                np.unique(trial_markers[(crop_size - 1):]), # crop_size -1 is index of first prediction
                needed_sample_start, trial_stop))
        self.add_trial_topo_trial_y(needed_samples, trial_markers)
        
    def add_trial_topo_trial_y(self, needed_samples, trial_markers):
        """ needed_samples are samples needed for predicting entire trial,
        i.e. they typically include a part before the first sample of the trial."""
        crop_size = self.input_time_length - self.n_sample_preds + 1
        assert (len(np.unique(trial_markers[(crop_size - 1):])) == 1), (
            ("Trial should have exactly one class, markers: {:s} ").format(
                np.unique(trial_markers[(crop_size - 1):])))
        trial_topo = needed_samples[:,:,np.newaxis,np.newaxis]
        trial_y = np.copy(trial_markers) - 1 # -1 as zero is non-trial marker
        trial_len = len(trial_topo)
        start_end_blocks = get_start_end_blocks_for_trial(crop_size-1,
            trial_len-1, self.input_time_length, self.n_sample_preds)
        assert start_end_blocks[0][0] == 0, "First block should start at first sample"
        batch = create_batch(trial_topo, trial_y, start_end_blocks,
            self.n_sample_preds)
        self.data_batches.append(batch[0])
        self.y_batches.append(batch[1])
        
    def train(self):
        n_trials = len(self.data_batches)
        if n_trials >= self.n_min_trials:
            log.info("Training model...")
            # Remember values as backup in case of NaNs
            model_param_vals_before = lasagne.layers.get_all_param_values(self.exp.final_layer)
            train_param_vals_before = [p.get_value() for p in self.train_params]
            all_blocks = np.concatenate(self.data_batches, axis=0)
            all_y_blocks = np.concatenate(self.y_batches, axis=0)
            # reshape to per block
            # assuming right now targets are simply labels
            # not one-hot encoded
            all_y_blocks = np.reshape(all_y_blocks, (-1, self.n_sample_preds))
            
            # make classes balanced
            # hopefully this is correct?! any sample shd be fine, -10 is a random decision
            labels_per_block = all_y_blocks[:,-10]
            unique_labels = sorted(np.unique(labels_per_block))
            if not np.array_equal(range(len(unique_labels)), 
                unique_labels):
                missing_classes = np.setdiff1d(range(len(unique_labels)),
                    unique_labels)
                log.info(("Do not have labels for all classes yet, "
                    "missing: {:s}, Skipping training...".format(
                        str(missing_classes))))
                return
            class_probs = np.zeros(len(unique_labels))
            for i_class in unique_labels:
                freq = np.mean(labels_per_block == i_class)
                prob = 1.0/ (len(unique_labels) * freq)
                class_probs[i_class] = prob
            block_probs = np.zeros(len(labels_per_block))
            for i_class in unique_labels:
                block_probs[labels_per_block == i_class] = class_probs[i_class]
            block_probs = block_probs / np.sum(block_probs)
            
            assert len(all_blocks) == len(all_y_blocks)
            for _ in xrange(self.n_updates_per_break):
                i_blocks = self.rng.choice(len(all_y_blocks),
                    size=self.batch_size, p=block_probs)
                this_y = np.concatenate(all_y_blocks[i_blocks], axis=0)
                this_topo = all_blocks[i_blocks]
                self.train_func(this_topo, this_y)

            # Check for Nans and if necessary reset to old values
            if np.any([np.any(np.isnan(p.get_value())) for p in self.train_params]):
                log.warn("Reset train parameters due to NaNs")
                for p, old_val in zip(self.train_params, train_param_vals_before):
                    p.set_value(old_val)
            all_layers_trained = lasagne.layers.get_all_layers(self.exp.final_layer)
            if np.any([np.any(np.isnan(p_val))
                    for p_val in lasagne.layers.get_all_param_values(all_layers_trained)]):
                log.warn("Reset model params due to NaNs")
                set_param_values_backwards_compatible(self.exp.final_layer, model_param_vals_before)
            assert not np.any([np.any(np.isnan(p.get_value())) for p in self.train_params])
            assert not np.any([np.any(np.isnan(p_val))
                    for p_val in lasagne.layers.get_all_param_values(all_layers_trained)])
            
            # Copy over new values to model used for prediction
            all_layers_used = lasagne.layers.get_all_layers(self.predicting_model)
            set_param_values_backwards_compatible(all_layers_used,
                lasagne.layers.get_all_param_values(all_layers_trained))
        else:
            log.info("Not training model yet, only have {:d} of {:d} trials ".format(
                n_trials, self.n_min_trials))

class NoTrainer(object):
    def process_markers(self, samples):
        pass
        
    def set_predicting_model(self, model):
        pass
    
    def set_data_processor(self, data_processor):
        pass

    def set_marker_buffer(self, marker_buffer):
        pass
        
    def initialize(self):
        pass
        
    def train(self):
        pass
