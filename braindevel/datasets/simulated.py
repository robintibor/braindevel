from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np
from lasagne.layers import get_all_layers
from numpy.random import RandomState
from braindevel.analysis.data_generation import create_sine_signal,\
    randomly_shifted_sines
from lasagne.layers.input import InputLayer

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


def pipeline_data(pipeline_func_and_args, shape, seed=4576546):
    inputs = np.zeros(shape, dtype=np.float32)
    n_trials = shape[0]
    rng = RandomState(seed)
    y = np.round(rng.rand(n_trials)).astype(np.int32)
    for func_and_args in pipeline_func_and_args:
        func = func_and_args[0]
        merge_func = func_and_args[1]
        try:
            new_inputs = func(shape=inputs.shape)
        except TypeError:
            try:
                new_inputs = func(topo=inputs)
            except TypeError:
                new_inputs = func(topo=inputs, y=y)
                

        inputs = merge_func(inputs, new_inputs)
    return inputs.astype(np.float32), y

def chan_factors(layer, factors):
    # put into chan_dim
    factors_arr = np.array(factors)[np.newaxis,:,np.newaxis,np.newaxis]
    return transform(layer, lambda topo: topo * factors_arr)

def zero_negative_class(topo, y):
    topo = topo.copy()
    topo[y == 0] = 0
    return topo

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

def same_shifted_sines_on_all_chans(shape, freq=10, sampling_freq=100,
        rng=None):
    number = shape[0]
    samples = shape[2]
    if rng is None:
        rng = RandomState(38473847)
    sines = randomly_shifted_sines(number, samples,freq=freq,
        sampling_freq=sampling_freq, rng=rng)
    # repeat along channel and final axis
    # to get correct shape
    sines = np.tile(sines[:,np.newaxis,:,np.newaxis],
                    (1, shape[1], 1, shape[3]))
    return sines

class TransformLayer(object):
    def __init__(self, incoming, func):
        if hasattr(incoming, '__len__'):
            self.input_layers = incoming
        else:
            self.input_layer = incoming
        self.func = func
        
    def transform(self, topo, y):
        try: 
            return self.func(topo=topo)
        except TypeError:
            try:
                return self.func(topo=topo, y=y)
            except TypeError:
                return self.func(shape=topo.shape)

# shortcut
transform = TransformLayer
        


def get_data(layer_or_layers, seed=4576546):
    """
    Computes the output for simulated data network.
    Parameters
    ----------
    layer_or_layers : Layer or list
        the :class:`TransformLayer` instance for which to compute the output
        data, or a list of :class:`TransformLayer` instances.
    Returns
    -------
    output : nd-array, nd_array 
        Topo output and y output
    """
    all_layers = get_all_layers(layer_or_layers)
    # initialize layer-to-output mapping from all input layers
    # with zeros
    all_outputs = dict((layer, np.zeros(layer.shape, dtype=np.float32))
                       for layer in all_layers
                       if isinstance(layer, InputLayer))
    
    rng = RandomState(seed)
    n_trials = all_layers[0].shape[0]
    y = np.round(rng.rand(n_trials)).astype(np.int32)
    # update layer-to-output mapping by propagating the inputs
    for layer in all_layers:
        if layer not in all_outputs:
            try:
                try:
                    layer_inputs = [all_outputs[input_layer]
                                for input_layer in layer.input_layers]
                except AttributeError:
                    layer_inputs = all_outputs[layer.input_layer]
            except KeyError:
                # one of the input_layer attributes must have been `None`
                raise ValueError("get_output() was called without giving an "
                                 "input expression for the free-floating "
                                 "layer %r. Please call it with a dictionary "
                                 "mapping this layer to an input expression."
                                 % layer)
            outputs = layer.transform(topo=layer_inputs, y=y)
            all_outputs[layer] = outputs
    # return the output(s) of the requested layer(s) only
    try:
        return [all_outputs[layer].astype(np.float32) 
            for layer in layer_or_layers], y
    except TypeError:
        return all_outputs[layer_or_layers].astype(np.float32), y
    
def set_first_channel_to_class_signal(topo,y):
    topo = topo.copy()
    assert np.array_equal(np.unique(y), [0,1])
    # transform form 0/1 to -1/1
    topo[:,0] = 2 * y[:,np.newaxis, np.newaxis] -1
    return topo

def add_topo(topo):
    summed_topo = topo[0]
    for other_topo in topo[1:]:
        summed_topo += other_topo
    return summed_topo

def create_gaussian_distractor(topo, rng=None):
    noise_shape = list(topo.shape)
    # same noise on all chans
    noise_shape[1] = 1
    if rng is None:
        rng = RandomState(329082938)
    return rng.randn(*noise_shape)

def weight(transform_layer, factor):
    return  transform(transform_layer, lambda topo: topo * factor)