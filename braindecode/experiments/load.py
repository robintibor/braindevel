import theano.tensor as T
import numpy as np
import lasagne

def load_model(filename):
    model = np.load(filename)
    all_layers = lasagne.layers.get_all_layers(model)
    for l in all_layers:
        if hasattr(l, 'convolve'):
            l.flip_filters = True
            l.convolution = T.nnet.conv2d
            l.n = 2
        # fix final reshape layer
        if hasattr(l, 'remove_invalids') and not hasattr(l, 'flatten'):
            l.flatten = True
            
    return model