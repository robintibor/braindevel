import numpy as np
import lasagne
from lasagne.layers.merge import ConcatLayer
from lasagne.layers.dense import DenseLayer
from lasagne.nonlinearities import softmax
from lasagne.layers.conv import Conv2DLayer
from jinja2.runtime import identity
from braindecode.veganlasagne.layers import FinalReshapeLayer,\
    get_n_sample_preds

class MergedNet(object):
    def __init__(self, networks, n_features_per_net, n_classes):
        self.networks = networks
        self.n_features_per_net = n_features_per_net
        self.n_classes = n_classes
        
    def get_layers(self):
        layers_per_net = [net.get_layers() for net in self.networks]
        # Check that all have same number of sample preds
        n_sample_preds = get_n_sample_preds(layers_per_net[0][-1])
        for layers in layers_per_net:
            assert get_n_sample_preds(layers[-1]) == n_sample_preds
        # remove dense softmax replace by dense linear
        reduced_layers = [replace_dense_softmax_by_dense_linear(all_l, n_f) 
                          for all_l, n_f in zip(layers_per_net, self.n_features_per_net)]
        # merge to all get same input
        input_first_net = reduced_layers[0][0]
        for layers in reduced_layers:
            input_net = layers[0]
            assert np.array_equal(input_first_net.shape, input_net.shape)
            # check any that have reference to this input layer,
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
            
        final_layers = [layers[-1] for layers in reduced_layers]
        l_merged = ConcatLayer(final_layers)

        l_merged = DenseLayer(l_merged,num_units=self.n_classes,
            nonlinearity=softmax)
        return lasagne.layers.get_all_layers(l_merged)

def replace_dense_softmax_by_dense_linear(all_layers, n_features):
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
            filter_size=input_to_reshape.filter_size, nonlinearity=identity,
            name='final_dense')

    new_reshape_l = FinalReshapeLayer(new_input_to_reshape)
    return lasagne.layers.get_all_layers(new_reshape_l)