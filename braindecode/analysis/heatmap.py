from theano.tensor.signal import downsample
import theano.tensor as T
import numpy as np
import theano
import lasagne

inputs = T.ftensor3()
output = downsample.max_pool_2d(inputs,ds=(2,2),mode='sum')#, ignore_border=True)
pool_fun = theano.function([inputs], output)

def back_relevance_dense_layer(out_relevances, in_activations, weights):
    """In_activations not used atm. would be needed for z or z+ rule"""
    squared_weights = weights * weights
    denominators_per_j = np.sum(squared_weights, axis=0)  # + np.maximum(softmax_biases, 0)
    scaled_weights = squared_weights / denominators_per_j
    input_relevances = np.sum(scaled_weights * out_relevances, axis=1)
    return input_relevances

def back_relevance_conv(out_relevances, in_activations, conv_weights, rule):
    #... do it differently go back form out relevances since they have to be redistributed
    conv_weights = conv_weights.astype(np.float32)
    out_relevances = out_relevances.astype(np.float32)
    in_activations = in_activations.astype(np.float32)
    assert rule in ['w_sqr', 'z_plus']
    kernel_size = conv_weights.shape[2:]
    in_relevances = np.zeros(in_activations.shape)
    for out_filt in xrange(out_relevances.shape[0]):
        relevant_weights = conv_weights[out_filt, :,::-1,::-1] # reverse filter
        if rule == 'w_sqr':
            adapted_weights = relevant_weights * relevant_weights
        for out_x in xrange(out_relevances.shape[1]):
            for out_y in xrange(out_relevances.shape[2]):
                if rule == 'z_plus':
                    adapted_weights = relevant_weights * (relevant_weights > 0)
                    relevant_input = in_activations[:,out_x:out_x+kernel_size[0],
                              out_y:out_y+kernel_size[1]]
                    adapted_weights = adapted_weights * relevant_input
                scaled_weights = adapted_weights / float(np.sum(adapted_weights))
                if np.sum(adapted_weights) == 0:
                    scaled_weights[:] = 1 / float(np.prod(scaled_weights.shape))
                relevance_to_add = scaled_weights * out_relevances[out_filt, out_x, out_y] 
                in_relevances[:,out_x:out_x+kernel_size[0],
                              out_y:out_y+kernel_size[1]] += relevance_to_add
    return in_relevances

def back_relevance_pool(out_relevances, in_activations, pool_size, pool_stride):
    assert pool_size == pool_stride, "At the moment only support pool size and stride same"
    # have to upsample relevances and compute sums of activations per pool region
    upsampled_relevances = np.repeat(np.repeat(out_relevances,pool_size[0],  axis=1), 
                                     pool_size[1], axis=2)
    region_pooled_activation = pool_fun(in_activations)
    upsampled_sum_activation = np.repeat(np.repeat(region_pooled_activation, pool_size[0],  axis=1), 
                                         pool_size[1], axis=2)
    scaled_activations = in_activations / upsampled_sum_activation
    # fix those activations where there was a 0 pooling output.. just distribute equally
    scaled_activations[np.isnan(scaled_activations)] = 1 / float(np.prod(pool_size))
    in_relevances = scaled_activations * upsampled_relevances
    return in_relevances

def compute_heatmap(out_relevances, all_activations_per_layer, all_layers):
    # stop before first layer...as it should be input layer...
    # and we alwasys need activations from before
    for i_layer in xrange(len(all_layers)-1, 0,-1):
        # We have out relevance for that layer, now 
        layer = all_layers[i_layer]
        in_activations = all_activations_per_layer[i_layer-1]
        if isinstance(layer, lasagne.layers.DenseLayer):
            dense_weights = layer.W.get_value()
            out_relevances = back_relevance_dense_layer(out_relevances,
                 in_activations, dense_weights)
        elif hasattr(layer, 'pool_size'):
            assert layer.stride == layer.pool_size, (
                "Only works with stride equal size at the moment")
            out_relevances = back_relevance_pool(out_relevances, in_activations,
                 pool_size=layer.pool_size, pool_stride=layer.stride)
        elif hasattr(layer, 'filter_size'):
            conv_weights = layer.W.get_value()
            out_relevances = back_relevance_conv(out_relevances, in_activations,conv_weights)
        else:
            raise ValueError("Trying to propagate through unknown layer")
        if out_relevances.shape != in_activations.shape:
            out_relevances = out_relevances.reshape(in_activations.shape)
    return out_relevances