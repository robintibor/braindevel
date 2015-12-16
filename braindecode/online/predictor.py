import numpy as np

from braindecode.online.ring_buffer import RingBuffer
import lasagne
import theano
from braindecode.veganlasagne.layers import get_input_time_length
from braindecode.datahandling.preprocessing import exponential_running_mean,\
    exponential_running_var_from_demeaned

class OnlinePredictor(object):
    def __init__(self, model, prediction_frequency):
        self.model = model
        self.prediction_frequency = prediction_frequency
        
    def initialize(self, n_chans):
        self.running_mean = None
        self.running_var = None
        self.i_sample = 0
        n_samples_in_buffer = 10000
        self.sample_buffer = RingBuffer(np.ones((n_samples_in_buffer, n_chans),
            dtype=np.float32))
        self.i_last_sample_prediction = 0
        self.last_prediction = None
        self.n_samples_pred_window = get_input_time_length(self.model)
        output = lasagne.layers.get_output(self.model, deterministic=True)
        inputs = lasagne.layers.get_all_layers(self.model)[0].input_var
        pred_fn = theano.function([inputs], output)
        self.pred_fn = pred_fn
    
    def update_and_standardize(self, samples):
        eps = 1e-7
        self.i_sample += len(samples)
        if self.running_mean is not None:
            assert self.running_var is not None
            next_means = exponential_running_mean(samples, factor_new=0.001,
                start_mean=self.running_mean)
            demeaned = samples - next_means
            next_vars = exponential_running_var_from_demeaned(demeaned,
                factor_new=0.001, start_var=self.running_var)
            standardized = demeaned / np.maximum(eps, np.sqrt(next_vars))
            self.running_mean = next_means[-1]
            self.running_var = next_vars[-1]
            return standardized
        else:
            self.running_mean = np.mean(samples, axis=0)
            self.running_var = np.var(samples, axis=0)
            return (samples - self.running_mean) / np.maximum(eps,
                np.sqrt(self.running_var))
        
    def receive_sample_block(self, samples):
        """Expect samples in timexchan format"""
        standardized_samples = self.update_and_standardize(samples)
        self.sample_buffer.extend(standardized_samples)
        if self.should_do_next_prediction():
            self.predict()
            
    def should_do_next_prediction(self):
        return (self.i_sample > self.n_samples_pred_window and 
            self.i_sample > (self.i_last_sample_prediction + 
                self.prediction_frequency))
    
    def predict(self):
        n_samples_after_pred = min(self.i_sample - self.n_samples_pred_window,
            self.i_sample - self.i_last_sample_prediction - self.prediction_frequency)
        assert n_samples_after_pred < self.prediction_frequency, ("Other case "
            "not implemented yet")
        start = -self.n_samples_pred_window - n_samples_after_pred
        end = -n_samples_after_pred
        if end == 0:
            end = None
        topo = self.sample_buffer[start:end].T[np.newaxis,:,:,np.newaxis]
        self.last_prediction = self.pred_fn(topo)
        self.i_last_sample_prediction = self.i_sample - n_samples_after_pred
    
    def pop_last_prediction_and_sample_ind(self):
        last_pred = self.last_prediction
        self.last_prediction = None
        return last_pred, self.i_last_sample_prediction

    def has_new_prediction(self):
        return self.last_prediction is not None
             
            
        