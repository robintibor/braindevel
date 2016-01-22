from theano.tensor.signal import downsample
import theano.tensor as T
import numpy as np
import theano
import lasagne
pool_fun = None

def back_relevance_dense_layer(out_relevances, in_activations, weights, rule):
    assert rule in ['w_sqr', 'z_plus']
    # for tests where i put int numbers
    weights = np.array(weights, dtype=np.float32)
    out_relevances = np.array(out_relevances, dtype=np.float32)
    in_activations = np.array(in_activations, dtype=np.float32)
    # weights are features x output_units => input_units x output_units
    # in_activations are input_units
    if rule == 'w_sqr':
        W_adapted = weights * weights
    if rule == 'z_plus':
        if in_activations.ndim > 1:
            in_activations = in_activations.flatten()
        W_plus = weights * (weights > 0)
        W_adapted = W_plus * in_activations[:, np.newaxis]
    W_scaled = W_adapted / np.sum(W_adapted, axis=0)
    W_scaled[:,np.sum(W_adapted, axis=0) == 0] = 1.0 / W_adapted.shape[0]
    input_relevances = W_scaled * out_relevances
    input_relevances = np.sum(input_relevances, axis=1)
    return input_relevances


def back_relevance_dense_layer_as_in_paper(out_relevances, in_activations, weights, rule):
    """Just for reference.. leads to same results (except not checking for NaNs yet)..."""
    assert rule in ['w_sqr', 'z_plus']
    # for tests where i put int numbers
    weights = np.array(weights, dtype=np.float32)
    out_relevances = np.array(out_relevances, dtype=np.float32)
    in_activations = np.array(in_activations, dtype=np.float32)
    # weights are features x output_units => input_units x output_units
    # in_activations are input_units
    if rule == 'w_sqr':
        W_adapted = weights * weights
        N = W_adapted / np.sum(W_adapted, axis=0)
        return np.dot(N, out_relevances)
    if rule == 'z_plus':
        V = weights * (weights > 0)
        Z = np.dot(V.T, in_activations)
        return in_activations * np.dot(V, (out_relevances / Z))

def back_relevance_conv(out_relevances, in_activations, conv_weights, rule):
    assert rule in ['w_sqr', 'z_plus']
    # for tests where i put int numbers
    conv_weights = np.array(conv_weights, dtype=np.float32)
    out_relevances = np.array(out_relevances, dtype=np.float32)
    in_activations = np.array(in_activations, dtype=np.float32)
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
    # FIrst create pool function if it doesn't exist
    global pool_fun
    if pool_fun is None:
        inputs = T.ftensor3()
        output = downsample.max_pool_2d(inputs,ds=(2,2),mode='sum')#, ignore_border=True)
        pool_fun = theano.function([inputs], output)
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

def compute_heatmap(out_relevances, all_activations_per_layer, all_layers, 
        all_rules):
    for rule in all_rules:
        assert rule in ['w_sqr', 'z_plus', None]
    # stop before first layer...as it should be input layer...
    # and we alwasys need activations from before
    for i_layer in xrange(len(all_layers)-1, 0,-1):
        # We have out relevance for that layer, now 
        layer = all_layers[i_layer]
        in_activations = all_activations_per_layer[i_layer-1]
        rule = all_rules[i_layer]
        if isinstance(layer, lasagne.layers.DenseLayer):
            dense_weights = layer.W.get_value()
            out_relevances = back_relevance_dense_layer(out_relevances,
                 in_activations, dense_weights, rule)
        elif hasattr(layer, 'pool_size'):
            assert layer.stride == layer.pool_size, (
                "Only works with stride equal size at the moment")
            out_relevances = back_relevance_pool(out_relevances, in_activations,
                 pool_size=layer.pool_size, pool_stride=layer.stride)
        elif hasattr(layer, 'filter_size'):
            conv_weights = layer.W.get_value()
            out_relevances = back_relevance_conv(out_relevances, in_activations,
                conv_weights, rule)
        else:
            raise ValueError("Trying to propagate through unknown layer")
        if out_relevances.shape != in_activations.shape:
            out_relevances = out_relevances.reshape(in_activations.shape)
    return out_relevances