import theano.tensor as T
import theano
from braindevel.veganlasagne.layers import SeparableConv2DLayer, unfold_filters
from numpy.random import RandomState
import numpy as np
import lasagne
from lasagne.layers import Conv2DLayer, InputLayer

def test_unfold_filters():
    channel_weights = T.fmatrix()
    kernel_weights = T.ftensor3()
    
    unfolded_filters = unfold_filters(channel_weights, kernel_weights)
    
    unfold_fn = theano.function([channel_weights, kernel_weights], unfolded_filters)
    rng = RandomState(38734)
    chan_weights = rng.rand(9,3).astype(np.float32)
    kernel_weights = rng.rand(9,4,2).astype(np.float32)
    unfolded = unfold_fn(chan_weights, kernel_weights)
    
    for i_filt in xrange(chan_weights.shape[0]):
        for i_chan in xrange(chan_weights.shape[1]):
            assert np.allclose(
                chan_weights[i_filt,i_chan] * kernel_weights[i_filt],
                unfolded[i_filt,i_chan])


def test_separable_conv():
    """ Test separable conv by comparing to normal conv..."""
    # Create unfolded filters
    channel_weights = T.fmatrix()
    kernel_weights = T.ftensor3()
    unfolded_filters = unfold_filters(channel_weights, kernel_weights)
    unfold_fn = theano.function([channel_weights, kernel_weights], unfolded_filters)
    rng = RandomState(38734)
    chan_weights = rng.rand(9,3).astype(np.float32)
    kernel_weights = rng.rand(9,4,2).astype(np.float32)
    unfolded = unfold_fn(chan_weights, kernel_weights)
    
    # Create reference function
    in_layer = InputLayer([None,3,5,8])
    conv_layer = Conv2DLayer(in_layer,num_filters=9, filter_size=[4, 2])
    conv_layer.W.set_value(unfolded)
    reference_out = lasagne.layers.get_output(conv_layer, deterministic=True)
    reference_out_fn = theano.function([in_layer.input_var], reference_out)
    
    # Create separable conv function, flatten weights to do that
    flattened_W = np.concatenate((chan_weights, 
        kernel_weights.reshape(kernel_weights.shape[0], -1)), 
        axis=1)
    in_layer = InputLayer([None,3,5,8])
    conv_layer = SeparableConv2DLayer(in_layer,num_filters=9, filter_size=[4, 2])
    conv_layer.W.set_value(flattened_W)
    out = lasagne.layers.get_output(conv_layer, deterministic=True)
    out_fn = theano.function([in_layer.input_var], out)
    
    # Compare results on random test data
    rng = RandomState(3876)
    test_data = rng.rand(7,3,5,8).astype(np.float32)
    assert np.allclose(reference_out_fn(test_data), out_fn(test_data))