import theano.tensor as T
import numpy as np
import lasagne
from braindecode.veganlasagne.layers import transform_to_normal_net,\
    get_used_input_length, get_input_shape, get_model_input_window
from braindecode.experiments.experiment import create_experiment

def load_model(filename):
    model = np.load(filename)
    all_layers = lasagne.layers.get_all_layers(model)
    for l in all_layers:
        if hasattr(l, 'convolve'):
            # I guess this is for backward compatibility?
            l.flip_filters = True
            l.convolution = T.nnet.conv2d
            l.n = 2
        # fix final reshape layer
        if hasattr(l, 'remove_invalids') and not hasattr(l, 'flatten'):
            l.flatten = True
            
    return model

def load_exp_and_model(basename):
    """ Loads experiment and model for analysis, sets invalid fillv alues to NaN."""
    model = load_model(basename + '.pkl')
    exp = create_experiment(basename + '.yaml')
    all_layers = lasagne.layers.get_all_layers(model)
    # mark nans to be sure you are doing correct transformations
    # also necessary for transformations to cnt and time activations
    for l in all_layers:
        if hasattr(l, 'invalid_fill_value'):
            l.invalid_fill_value = np.nan
    return exp, model

