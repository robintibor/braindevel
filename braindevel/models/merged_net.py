import numpy as np
import lasagne
from lasagne.layers.merge import ConcatLayer
from lasagne.layers.dense import DenseLayer
from lasagne.nonlinearities import softmax
from lasagne.layers.conv import Conv2DLayer
from jinja2.runtime import identity
from braindevel.veganlasagne.layers import FinalReshapeLayer,\
    get_n_sample_preds
from braindevel.veganlasagne.batch_norm import batch_norm

class MergedNet(object):
    def __init__(self, networks, n_features_per_net, n_classes,
            batch_norm_before_merge=False,
            nonlin_before_merge=identity):
        self.networks = networks
        self.n_features_per_net = n_features_per_net
        self.n_classes = n_classes
        self.batch_norm_before_merge = batch_norm_before_merge
        self.nonlin_before_merge = nonlin_before_merge
        
    def get_layers(self):
        # make into list if nonlin only one
        if not hasattr(self.nonlin_before_merge, '__len__'):
            nonlins_before_merge = ((self.nonlin_before_merge,) *
                len(self.networks))
        else:
            nonlins_before_merge = self.nonlin_before_merge
        layers_per_net = [net.get_layers() for net in self.networks]
        # Check that all have same number of sample preds
        n_sample_preds = get_n_sample_preds(layers_per_net[0][-1])
        for layers in layers_per_net:
            assert get_n_sample_preds(layers[-1]) == n_sample_preds
        # remove dense softmax replace by dense linear
        reduced_layers = [replace_dense_softmax_by_dense_linear(all_l, n_f, 
            nonlin_before_merge=nonlin,
            batch_norm_before_merge=self.batch_norm_before_merge) 
              for all_l, n_f, nonlin in zip(layers_per_net, self.n_features_per_net,
                  nonlins_before_merge)]
        # hopefully still works with new method below:)
        use_same_input_layer(reduced_layers)
            
        final_layers = [layers[-1] for layers in reduced_layers]
        l_merged = ConcatLayer(final_layers)

        l_merged = DenseLayer(l_merged,num_units=self.n_classes,
            nonlinearity=softmax)
        return lasagne.layers.get_all_layers(l_merged)

def use_same_input_layer(networks):
    '''
    Make multiple networks use sanme input layer
    :param networks: List of either final layers or all layers per network
    '''
    networks = list(networks)
    # make into all layers if not 
    # merge to all get same input
    for i_net, network in enumerate(networks):
        if not hasattr(network, '__len__'):
            networks[i_net] = lasagne.layers.get_all_layers(network)
        
    input_first_net = networks[0][0]
    for layers in networks: # check if input shape is equal
        input_net = layers[0]
        assert np.array_equal(input_first_net.shape, input_net.shape)
        # for any layers that have reference to this input layer,
        # set reference to common input layer
        for l in layers:
            if hasattr(l, 'input_layer'):
                if l.input_layer == input_net:
                    l.input_layer = input_first_net
            elif hasattr(l, 'input_layers'):
                for i_in in len(l.input_layers):
                    if l.input_layers[i_in] == input_net:
                        l.input_layers[i_in] = input_first_net
            else:
                assert l.__class__.__name__ == "InputLayer"

def replace_dense_softmax_by_dense_linear(all_layers, n_features,
        nonlin_before_merge, batch_norm_before_merge):
    """Replace dense/conv (n_classes) -> reshape -> softmax
    by         dense/conv (n_features) -> reshape"""
    
    reshape_layer = [l for l in all_layers if l.__class__.__name__ == 'FinalReshapeLayer']

    assert len(reshape_layer) == 1
    reshape_layer = reshape_layer[0]

    input_to_reshape = reshape_layer.input_layer
    # We expect a linear conv2d as "final dense" before the reshape...
    assert input_to_reshape.__class__.__name__ == 'Conv2DLayer', (
        "expect conv before reshape")
    assert input_to_reshape.nonlinearity.func_name == 'linear'

    # recreate with different number of filters
    assert input_to_reshape.stride == (1,1)
    new_input_to_reshape = Conv2DLayer(input_to_reshape.input_layer,
           num_filters=n_features,
            filter_size=input_to_reshape.filter_size, nonlinearity=nonlin_before_merge,
            name='final_dense')
    if batch_norm_before_merge:
        new_input_to_reshape = batch_norm(new_input_to_reshape, 
            alpha=0.1,epsilon=0.01)

    new_reshape_l = FinalReshapeLayer(new_input_to_reshape)
    return lasagne.layers.get_all_layers(new_reshape_l)
