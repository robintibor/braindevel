import lasagne
import numpy as np
import theano.tensor as T
from lasagne.updates import adam
from numpy.random import RandomState
from braindecode.veganlasagne.layers import (get_input_time_length,
    get_n_sample_preds)
from braindecode.datahandling.batch_iteration import (create_batch, 
    get_start_end_blocks_for_trial)
from braindecode.util import FuncAndArgs
import logging
from pylearn2.utils.timing import log_timing
log = logging.getLogger(__name__)

class BatchWiseCntTrainer(object):
    def __init__(self, exp, n_updates_per_break, batch_size, learning_rate,
                n_min_trials, trial_start_offset):
        self.cnt_model = exp.final_layer
        self.exp = exp
        self.n_updates_per_break = n_updates_per_break
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_min_trials = n_min_trials
        self.trial_start_offset = trial_start_offset
        
    def process_samples(self, samples):
        marker_samples_with_overlap = np.copy(
            self.marker_buffer[-len(samples)-2:])
        trial_has_ended = np.sum(np.diff(marker_samples_with_overlap) < 0) > 0
        if trial_has_ended:
            log.info("Trial has ended")
            # + 1 as diff "removes" one index, i.e. diff will be above zero
            # at the index 1 before the increase=> the trial start
            trial_start = np.flatnonzero(np.diff(self.marker_buffer) > 0)[-1] + 1
            trial_end = np.flatnonzero(np.diff(self.marker_buffer) < 0)[-1]
            assert trial_start < trial_end, ("trial start {:d} should be "
                "before trial end {:d}, markers: {:s}").format(trial_start, 
                    trial_end, str(marker_samples_with_overlap))
            self.add_blocks(trial_start, trial_end)
            
            with log_timing(log, None, final_msg='Time for training:'):
                self.train()
        
    def set_model(self, model):
        self.model = model
        self.input_time_length = get_input_time_length(self.cnt_model)
        self.n_sample_preds = get_n_sample_preds(self.cnt_model)
        
    def set_data_processor(self, data_processor):
        self.data_processor = data_processor

    def set_marker_buffer(self, marker_buffer):
        self.marker_buffer = marker_buffer
        
    def initialize(self):
        self.rng = RandomState(30948348)
        self.data_batches = []
        self.y_batches = []
        # create train function
        targets = T.ivector()
        self.exp.updates_expression = FuncAndArgs(adam,
            learning_rate=self.learning_rate)
        log.info("Compile train function...")
        self.exp.create_theano_functions(targets)
        log.info("Done compiling train function.")
    
    def add_blocks(self, trial_start, trial_end):
        samples_per_pred = self.input_time_length - self.n_sample_preds + 1
        pred_start = trial_start + self.trial_start_offset
        if pred_start + self.n_sample_preds - 1 > trial_end:
            return # Too little data in this trial to train on it...
        needed_sample_start = pred_start - samples_per_pred + 1
        trial_topo = np.copy(self.data_processor.get_samples(needed_sample_start, 
            trial_end))
        trial_topo = trial_topo[:,:,np.newaxis,np.newaxis]
        all_markers = self.marker_buffer[needed_sample_start:trial_end]
        assert (len(np.unique(all_markers[(samples_per_pred - 1):])) == 1), (
            ("Trial should have exactly one class, markers: {:s} "
                "trial start: {:d}, trial_end: {:d}").format(
                np.unique(all_markers[(samples_per_pred - 1):]),
                needed_sample_start, trial_end))
        trial_y = np.copy(all_markers) - 1 # -1 as zero is non-trial marker
        # trial start can't be at zeor atm or else we would have to take more data
        trial_len = len(trial_topo)
        start_end_blocks = get_start_end_blocks_for_trial(samples_per_pred-1,
            trial_len-1, self.input_time_length, self.n_sample_preds)
        batch = create_batch(trial_topo, trial_y, start_end_blocks,
            self.n_sample_preds)
        self.data_batches.append(batch[0])
        self.y_batches.append(batch[1])
        
    def train(self):
        n_trials = len(self.data_batches)
        if n_trials >= self.n_min_trials:
            log.info("Training model...")
            all_blocks = np.concatenate(self.data_batches, axis=0)
            all_y_blocks = np.concatenate(self.y_batches, axis=0)
            # reshape to per block
            # assuming right now targets are simply labels
            # not one-hot encoded
            all_y_blocks = np.reshape(all_y_blocks, (-1, self.n_sample_preds))
            assert len(all_blocks) == len(all_y_blocks)
            for _ in xrange(self.n_updates_per_break):
                i_blocks = self.rng.choice(len(all_y_blocks), size=self.batch_size)
                this_y = np.concatenate(all_y_blocks[i_blocks], axis=0)
                this_topo = all_blocks[i_blocks]
                self.exp.train_func(this_topo, this_y)
            # Copy over new values
            all_layers_trained = lasagne.layers.get_all_layers(self.exp.final_layer)
            all_layers_used = lasagne.layers.get_all_layers(self.model)
            lasagne.layers.set_all_param_values(all_layers_used,
                lasagne.layers.get_all_param_values(all_layers_trained))
        else:
            log.info("Not training model yet, only have {:d} of {:d} trials ".format(
                n_trials, self.n_min_trials))

class NoTrainer(object):
    def process_samples(self, samples):
        pass
        
    def set_model(self, model):
        pass
    
    def set_data_processor(self, data_processor):
        pass

    def set_marker_buffer(self, marker_buffer):
        pass
        
    def initialize(self):
        pass
    
    def add_blocks(self, trial_start, trial_end):
        pass
        
    def train(self):
        pass