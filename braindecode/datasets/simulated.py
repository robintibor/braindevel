from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np
from numpy.random import RandomState
from braindecode.analysis.data_generation import create_sine_signal

class SimulatedData(DenseDesignMatrix):
    reloadable=False
    def __init__(self, pipeline_func_and_args, shape):
        self.pipeline_func_and_args = pipeline_func_and_args
        self.shape = shape
    
    def ensure_is_loaded(self):
        if not hasattr(self, 'X'):
            self.load()
        
    def load(self):
        topo, y = pipeline_data(self.pipeline_func_and_args, self.shape)
        super(SimulatedData, self).__init__(topo_view=topo, y=y)


def pipeline_data(pipeline_func_and_args, shape):
    inputs = np.zeros(shape, dtype=np.float32)
    n_trials = shape[0]
    rng = RandomState(4576546)
    y = np.round(rng.rand(n_trials)).astype(np.int32)
    for func_and_args in pipeline_func_and_args:
        func = func_and_args[0]
        kwargs = func_and_args[1]
        merge_args = func_and_args[2]
        try:
            new_inputs = func(shape=inputs.shape, **kwargs)
        except TypeError:
            new_inputs = func(topo=inputs, **kwargs)

        new_inputs = new_inputs * merge_args.pop('factor', 1)
        if merge_args.pop('only_positive_class', False):
            new_inputs = new_inputs * y[:,np.newaxis,np.newaxis,np.newaxis]
        if merge_args['operator'] == 'add':
            inputs += new_inputs
        elif merge_args['operator'] == 'multiply':
            inputs *= new_inputs
        elif merge_args['operator'] == 'replace':
            inputs = new_inputs
        else:
            raise ValueError("Unknown operator {:s}".format(merge_args['operator']))
            
    return inputs.astype(np.float32), y

def create_sine_trials(shape, sampling_freq, freq):
    n_trials = shape[0]
    n_samples = shape[2]
    sine_signal = create_sine_signal(samples=n_samples, freq=freq,sampling_freq=sampling_freq)
    sine_matrix = np.tile(sine_signal, [n_trials,1,1])[:,:,:,np.newaxis]
    return sine_matrix
    
def create_gaussian_noise(shape, rng=None):
    if rng is None:
        rng = RandomState(329082938)
    return rng.randn(*shape)

def standardize_trialwise(topo):
    return (topo - np.mean(topo, axis=(2), keepdims=True)) / np.std(topo,axis=(2), keepdims=True)