import numpy as np
import lasagne
from copy import deepcopy

def combine_temporal_spatial_weights(temporal_weights, spatial_weights):
    # Compute combined the weights. 
    # We have to reverse them with ::-1 to change convolution to cross-correlation


    temporal_weights = temporal_weights[:,:,::-1,::-1]
    spat_filt_weights = spatial_weights[:,:,::-1,::-1]

    combined_weights = np.tensordot(spat_filt_weights, temporal_weights, axes=(1,0))

    combined_weights = combined_weights.squeeze()
    return combined_weights

def transform_to_combined_weights(model, i_temp_layer=2):
    new_model = deepcopy(model)
    all_layers = lasagne.layers.get_all_layers(new_model)
    temp_layer = all_layers[i_temp_layer]
    spat_layer = all_layers[i_temp_layer+1]
    after_conv_layer = all_layers[i_temp_layer+2]
    temp_weights = temp_layer.W.get_value()
    spat_filt_weights = spat_layer.W.get_value()
    combined_weights = combine_temporal_spatial_weights(temp_weights,
        spat_filt_weights)

    temp_biases = temp_layer.b.get_value()
    # Take care to multiply along correct axis (second, not first)
    temp_biases_weighted = temp_biases[np.newaxis, :, np.newaxis, np.newaxis] * spat_filt_weights
    spat_biases = spat_layer.b.get_value()
    combined_biases = np.sum(temp_biases_weighted, axis=(1,2,3)) + spat_biases

    combined_layer = lasagne.layers.Conv2DLayer(all_layers[i_temp_layer - 2],
                                               num_filters = combined_weights.shape[0],
                                   filter_size=[combined_weights.shape[2], 1],
                                               nonlinearity=spat_layer.nonlinearity)
    combined_layer.W.set_value(combined_weights[:,:,::-1,np.newaxis])
    combined_layer.b.set_value(combined_biases)

    after_conv_layer.input_layer = combined_layer
    new_model = all_layers[-1]
    return new_model