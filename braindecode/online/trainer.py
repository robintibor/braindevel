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
from lasagne.objectives import categorical_crossentropy
from braindecode.online.objectives import masked_loss_func
log = logging.getLogger(__name__)

class BatchWiseCntTrainer(object):
    def __init__(self, exp, n_updates_per_break, batch_size, learning_rate,
                n_min_trials, trial_start_offset, break_start_offset,
                break_stop_offset,
                train_param_values,
                deterministic_training=False, add_breaks=True,
                min_break_samples=0, min_trial_samples=0):
        self.cnt_model = exp.final_layer
        self.__dict__.update(locals())
        del self.self
        
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
        loss_fn = masked_loss_func(categorical_crossentropy)
        loss = loss_fn(prediction, targets).mean()
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
            assert self.marker_buffer[trial_start - 1] == 0, (
                "Expect a 0 marker before trial start, instead {:d}".format(
                    self.marker_buffer[trial_start - 1]))
            assert self.marker_buffer[trial_start] != 0, (
                "Expect a nonzero marker at trial start instead {:d}".format(
                    self.marker_buffer[trial_start]))
            assert self.marker_buffer[trial_stop-1] != 0, (
                "Expect a nonzero marker at trial end instead {:d}".format(
                self.marker_buffer[trial_stop]))
            assert self.marker_buffer[trial_start] == self.marker_buffer[trial_stop-1], (
                "Expect a same marker at trial start and end instead {:d} / {:d}".format(
                    self.marker_buffer[trial_start],
                    self.marker_buffer[trial_stop]))
            self.add_trial(trial_start, trial_stop,
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
                assert self.marker_buffer[trial_start - 1] == 0, (
                    "Expect a 0 marker before trial start, instead {:d}".format(
                    self.marker_buffer[trial_start - 1]))
                assert self.marker_buffer[trial_start] != 0, (
                    "Expect a nonzero marker at trial start instead {:d}".format(
                    self.marker_buffer[trial_start]))
                self.add_break(break_start=trial_stop, break_stop=trial_start,
                    all_samples=self.data_processor.sample_buffer,
                    all_markers=self.marker_buffer)
            #log.info("Break added, now at {:d} batches".format(len(self.data_batches)))

    def add_trial(self, trial_start, trial_stop, all_samples, all_markers):
        # Add trial by determining needed signal/samples and markers
        # In case the model predicts more samples concurrently
        # than the number of trials in this sample
        # prepad the markers with -1 and signal with zeros
        assert (len(np.unique(all_markers[trial_start:trial_stop])) == 1), (
            "All markers should be the same in one trial, instead got:"
            "{:s}".format(str(np.unique(all_markers[trial_start:trial_stop]))))
        # determine markers and in samples for default case
        pred_start = trial_start + self.trial_start_offset
        if (pred_start < trial_stop) and (trial_stop - trial_start >= self.min_trial_samples):
            assert (len(np.unique(all_markers[pred_start:trial_stop])) == 1), (
                "All predicted markers should be the same in one trial, instead got:"
                "{:s}".format(str(np.unique(all_markers[trial_start:trial_stop]))))
            self.add_trial_or_break(pred_start, trial_stop, all_samples,
                all_markers)
        elif pred_start >= trial_stop:
            log.warning("Prediction start {:d} is past trial stop {:d}".format(
                    pred_start, trial_stop) + ", not adding trial")
        else:
            assert trial_stop - trial_start < self.min_trial_samples
            log.warn("Trial only {:d} samples, want {:d} samples, not using.".format(
                trial_stop -trial_start, self.min_trial_samples))

    def add_break(self, break_start, break_stop, all_samples, all_markers):
        if self.add_breaks:
            all_markers = np.copy(all_markers)
            assert np.all(all_markers[break_start:break_stop] == 0), (
                "all markers in break should be 0, instead have markers:"
                "{:s}\nbreak start: {:d}\nbreak stop: {:d}\nmarker sequence: {:s}".format(
                    str(np.unique(all_markers[break_start:break_stop]
                    )), break_start, break_stop,
                    str(all_markers[break_start-1:break_stop+1])))
            assert all_markers[break_start - 1] != 0
            assert all_markers[break_stop] != 0
            pred_start = break_start + self.break_start_offset
            pred_stop = break_stop + self.break_stop_offset
            if (pred_start < pred_stop) and (break_stop - break_start >= self.min_break_samples):
                # keep n_classes for 1-based matlab indexing logic in markers
                all_markers[pred_start:pred_stop] = self.n_classes
                self.add_trial_or_break(pred_start, pred_stop, all_samples, all_markers)
            elif pred_start >= pred_stop:
                log.warning(
                    "Prediction start {:d} is past prediction stop {:d}".format(
                        pred_start, pred_stop) + ", not adding break")
            else:
                assert break_stop - break_start < self.min_break_samples
                log.warn("Break only {:d} samples, want {:d} samples, not using.".format(
                    break_stop - break_start, self.min_break_samples))
                
        else:
            pass #Ignore break that was supposed to be added

    def add_trial_or_break(self, pred_start, pred_stop, all_samples, all_markers):
        """Assumes all markers already changed the class for break."""
        crop_size = self.input_time_length - self.n_sample_preds + 1
        in_sample_start = pred_start - crop_size + 1
        # Later functions need one marker per input sample
        # (so also need markers at start that will not actually be used, 
        # which are only there
        # for the receptive field of ConvNet)
        # These functions will then cut out correct markers.
        # We want to make sure that no unwanted markers are used, so
        # we only extract the markers that will actually be predicted
        # and pad with 0s (which will be converted to -1 (!) and 
        # should not be used later, except
        # trial too small and we go into the if clause below)
        assert len(all_markers) == len(all_samples)
        assert pred_stop < len(all_markers)
        needed_samples = all_samples[in_sample_start:pred_stop]
        needed_markers = np.copy(all_markers[pred_start:pred_stop])
        needed_markers = np.concatenate((np.zeros(crop_size - 1,
            dtype=needed_markers.dtype), needed_markers))
        assert len(needed_samples) == len(needed_markers), (
            "{:d} markers and {:d} samples (should be same)".format(
                len(needed_samples), len(needed_markers)))
        n_expected_samples = pred_stop - pred_start + crop_size - 1
        # this assertion here for regression reasons, failed before
        assert len(needed_markers) == n_expected_samples, (
            "Extracted {:d} markers, but should have {:d}".format(
                len(needed_markers), n_expected_samples))
        # handle case where trial is too small
        if pred_stop - pred_start < self.n_sample_preds:
            log.warn("Trial/break has only {:d} predicted samples in it, "
                "less than the "
                "{:d} concurrently processed samples of the model!".format(
                    pred_stop - pred_start,
                    self.n_sample_preds))
            # add -1 markers that will not be used during training for the
            # data before
            n_pad_samples = self.n_sample_preds - (pred_stop - pred_start)
            pad_markers = np.zeros(n_pad_samples, dtype=all_markers.dtype)
            needed_markers = np.concatenate((pad_markers, needed_markers))
            pad_samples = np.zeros_like(all_samples[0:n_pad_samples])
            needed_samples = np.concatenate((pad_samples, needed_samples))
            pred_start = pred_start - n_pad_samples

        assert pred_stop - pred_start >= self.n_sample_preds
        n_expected_samples = pred_stop - pred_start + crop_size - 1
        assert len(needed_markers) == n_expected_samples, (
            "Extracted {:d} markers, but should have {:d}".format(
                len(needed_markers), n_expected_samples))
        assert len(needed_samples) == n_expected_samples, (
            "Extracted {:d} samples, but should have {:d}".format(
                len(needed_samples), n_expected_samples))
        self.add_trial_topo_trial_y(needed_samples, needed_markers)

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
        
        assert (len(trial_starts) == len(trial_stops)), (
            "Have {:d} trial starts, but {:d} trial stops (should be equal)".format(
                len(trial_starts), len(trial_stops)))
        assert(np.all(trial_starts <= trial_stops))
        return trial_starts, trial_stops

    def add_trial_topo_trial_y(self, needed_samples, trial_markers):
        """ needed_samples are samples needed for predicting entire trial,
        i.e. they typically include a part before the first sample of the trial."""
        crop_size = self.input_time_length - self.n_sample_preds + 1
        assert (len(np.unique(trial_markers[(crop_size - 1):])) == 1) or (
            (len(np.unique(trial_markers[(crop_size - 1):])) == 2) and (
                0 in trial_markers[(crop_size - 1):])), (
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
            # hopefully this is correct?! any sample shd be fine, -1 is a random decision
            labels_per_block = all_y_blocks[:,-1]
            
            # Rebalance by calculating frequencies of classes in data
            # and then rebalancing by sampling with inverse probability
            unique_labels = sorted(np.unique(labels_per_block))
            class_probs = {}
            for i_class in unique_labels:
                freq = np.mean(labels_per_block == i_class)
                prob = 1.0/ (len(unique_labels) * freq)
                class_probs[i_class] = prob
            block_probs = np.zeros(len(labels_per_block))
            for i_class in unique_labels:
                block_probs[labels_per_block == i_class] = class_probs[i_class]
            # Renormalize probabilities
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
