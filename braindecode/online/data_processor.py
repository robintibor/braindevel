from braindecode.online.ring_buffer import RingBuffer
import numpy as np
from braindecode.datahandling.preprocessing import exponential_running_mean,\
    exponential_running_var_from_demeaned


class StandardizeProcessor(object):
    def __init__(self, factor_new, eps=1e-4, 
            n_samples_in_buffer=10000):
        self.factor_new = factor_new
        self.eps = eps
        self.n_samples_in_buffer = n_samples_in_buffer
    
    def initialize(self, n_chans):
        self.running_mean = None
        self.running_var = None
        self.sample_buffer = RingBuffer(np.ones((
            self.n_samples_in_buffer, n_chans), dtype=np.float32))
        
    def process_samples(self, samples):
        standardized_samples = self.update_and_standardize(samples)
        self.sample_buffer.extend(standardized_samples)
        
    def update_and_standardize(self, samples):
        if self.running_mean is not None:
            assert self.running_var is not None
            next_means = exponential_running_mean(samples,
                factor_new=self.factor_new, start_mean=self.running_mean)
            demeaned = samples - next_means
            next_vars = exponential_running_var_from_demeaned(demeaned,
                factor_new=self.factor_new, start_var=self.running_var)
            standardized = demeaned / np.maximum(self.eps, np.sqrt(next_vars))
            self.running_mean = next_means[-1]
            self.running_var = next_vars[-1]
            return standardized
        else:
            self.running_mean = np.mean(samples, axis=0)
            self.running_var = np.var(samples, axis=0)
            return (samples - self.running_mean) / np.maximum(self.eps,
                np.sqrt(self.running_var))
            
    def get_samples(self, start, end):
        return self.sample_buffer[start:end]

        
    
    
        