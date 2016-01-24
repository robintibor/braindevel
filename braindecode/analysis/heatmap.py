import theano.tensor as T
import numpy as np
import theano
import lasagne
from theano.tensor.nnet import conv2d
from braindecode.veganlasagne.layers import get_input_shape

def create_heatmap_fn(all_layers, rules, min_in=None, max_in=None):
    # only using single trial, so one less dim than input shape
    if len(get_input_shape(all_layers[-1])) == 2:
        input_trial = T.fvector()
    elif len(get_input_shape(all_layers[-1])) == 4:
        input_trial = T.ftensor3()
    assert len(all_layers[-1].output_shape) == 2
    # again, just single trial:
    out_relevances = T.fvector()
    heatmap = create_heatmap(out_relevances, input_trial, all_layers, rules,
                            min_in, max_in)
    heatmap_fn = theano.function([out_relevances, input_trial], heatmap)         
    return heatmap_fn

def create_heatmap(out_relevances, input_trial, all_layers, 
        all_rules, min_in=None, max_in=None):
    """Theano expression for computing the heatmap.
    Expects a single input trial not a batch. Similarly a single output relevance"""
    
    for rule in all_rules:
        assert rule in ['w_sqr', 'z_plus', 'z_b', None]
    assert len(all_rules) == len(all_layers)
    
    if input_trial.ndim == 1:
        input_var = input_trial.dimshuffle('x',0)
    elif input_trial.ndim == 3:
        input_var = input_trial.dimshuffle('x', 0,1,2)
    activations_per_layer = lasagne.layers.get_output(all_layers, 
         input_var,deterministic=True, input_var=input_var)
    
    assert len(activations_per_layer) == len(all_layers)
    
    # stop before first layer...as it should be input layer...
    # and we always need activations from before
    for i_layer in xrange(len(all_layers)-1, 0,-1):
        # We have out relevance for that layer, now 
        layer = all_layers[i_layer]
        in_activations = activations_per_layer[i_layer-1][0]
        rule = all_rules[i_layer]
        if isinstance(layer, lasagne.layers.DenseLayer):
            if in_activations.ndim > 1:
                in_act_flat = in_activations.flatten()
            out_relevances = relevance_dense(out_relevances,
                 in_act_flat, layer.W, rule, min_in, max_in)
        elif hasattr(layer, 'pool_size'):
            out_relevances = relevance_pool(out_relevances, in_activations,
                 pool_size=layer.pool_size, pool_stride=layer.stride)
        elif hasattr(layer, 'filter_size'):
            out_relevances = relevance_conv(out_relevances, in_activations,
                layer.W, rule, min_in, max_in)
        else:
            raise ValueError("Trying to propagate through unknown layer")
        if out_relevances.shape != in_activations.shape:
            out_relevances = out_relevances.reshape(in_activations.shape)
    return out_relevances

def compute_heatmap(out_relevances, all_activations_per_layer, all_layers, 
        all_rules, min_in=None, max_in=None):
    """Numpy version to compute relevance heatmap."""
    
    for rule in all_rules:
        assert rule in ['w_sqr', 'z_plus', 'z_b', None]
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
                conv_weights, rule, min_in, max_in)
        else:
            raise ValueError("Trying to propagate through unknown layer")
        if out_relevances.shape != in_activations.shape:
            out_relevances = out_relevances.reshape(in_activations.shape)
    return out_relevances

def relevance_conv(out_relevances, inputs, weights, rule, min_in=None,
        max_in=None):
    assert rule in ['w_sqr', 'z_plus', 'z_b']
    if rule == 'w_sqr':
        return relevance_conv_w_sqr(out_relevances, weights)
    elif rule == 'z_plus':
        return relevance_conv_z_plus(out_relevances, inputs, weights)
    elif rule == 'z_b':
        assert min_in is not None
        assert max_in is not None
        assert min_in <= 0
        assert max_in >= 0
        return relevance_conv_z_b(out_relevances, inputs, weights,
            min_in, max_in)

def relevance_conv_w_sqr(out_relevances, weights):
    weights_sqr = weights * weights
    weights_norm = T.sum(weights_sqr, axis=(1,2,3), keepdims=True)
    # prevent division by zero
    weights_norm = weights_norm + (T.eq(weights_norm, 0) * 1)
    weights_scaled = weights_sqr / weights_norm
    # upconv
    in_relevances = conv2d(out_relevances.dimshuffle('x',0,1,2), 
                           weights_scaled.dimshuffle(1,0,2,3)[:,:,::-1,::-1],
                           border_mode='full')[0]
    return in_relevances

def relevance_conv_z_plus(out_relevances, inputs, weights):
    weights_plus = weights * T.gt(weights, 0)
    norms_for_relevances = conv2d(inputs.dimshuffle('x',0,1,2), weights_plus)[0]
    # prevent division by 0...
    norms_for_relevances += T.eq(norms_for_relevances, 0) * 1
    normed_relevances = out_relevances / norms_for_relevances
    # upconv
    in_relevances = conv2d(normed_relevances.dimshuffle('x',0,1,2), 
                           weights_plus.dimshuffle(1,0,2,3)[:,:,::-1,::-1], border_mode='full')[0]
    in_relevances_proper = in_relevances * inputs
    return in_relevances_proper


def relevance_conv_z_b(out_relevances, inputs, weights, min_in, max_in):
    assert min_in <= 0
    assert max_in >= 0
    weights_b = T.lt(weights, 0) * weights * -max_in
    weights_b += T.gt(weights, 0) * weights * -min_in

    norms_for_relevances = conv2d(inputs.dimshuffle('x',0,1,2), weights)[0]
    norms_for_relevances += T.sum(weights_b)
    # prevent division by 0...
    norms_for_relevances += T.eq(norms_for_relevances, 0) * 1
    normed_relevances = out_relevances / norms_for_relevances
    # upconv data
    in_relevances_data = conv2d(normed_relevances.dimshuffle('x',0,1,2), 
                           weights.dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
                           border_mode='full')[0]
    in_relevances_data *= inputs
    # upconv weight offsets to enforce positivity
    in_relevances_b = conv2d(normed_relevances.dimshuffle('x',0,1,2), 
                           weights_b.dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
                           border_mode='full')[0]
    in_relevances = in_relevances_data + in_relevances_b
    return in_relevances


def relevance_dense(out_relevances, in_activations, weights, rule,
    min_in=None, max_in=None):
    """Mostly copied from paper Supplementary pseudocode:
    http://arxiv.org/abs/1512.02479"""
    assert rule in ['w_sqr', 'z_plus', 'z_b']
    # weights are features x output_units => input_units x output_units
    # in_activations are input_units
    if rule == 'w_sqr':
        W_adapted = weights * weights
        Z = T.sum(W_adapted, axis=0, keepdims=True)
        # prevent division by 0...
        Z += T.eq(Z, 0) * 1
        N = W_adapted / Z
        return T.dot(N, out_relevances)
    elif rule == 'z_plus':
        V = weights * T.gt(weights, 0)
        Z = T.dot(V.T, in_activations)
        # prevent division by 0...
        Z += T.eq(Z, 0) * 1
        return in_activations * T.dot(V, (out_relevances / Z))
    elif rule == 'z_b':
        assert min_in is not None
        assert max_in is not None
        assert min_in <= 0
        assert max_in >= 0
        U = weights * T.lt(weights, 0)
        V = weights * T.gt(weights, 0)
        R_Norm = T.dot(weights.T, in_activations)
        # we only expect two global min in and max in values
        # therefore no need to do dot product here just multiply and um
        R_Norm -= T.sum(V * min_in, axis=0)
        R_Norm -= T.sum(U * max_in, axis=0)
        # prevent division by 0...
        R_Norm += T.eq(R_Norm, 0) * 1
        N = out_relevances / R_Norm
        in_relevances = in_activations * (T.dot(weights, N))
        in_relevances -= min_in * (T.dot(V, N))
        in_relevances -= max_in * (T.dot(U, N))
        return in_relevances

def create_back_conv_w_sqr_fn():
    out_relevances = T.ftensor3()
    weights = T.ftensor4()
    in_relevances = relevance_conv_w_sqr(out_relevances, weights)
    back_relevance_conv_fn = theano.function([out_relevances, weights],
                                         in_relevances)
    return back_relevance_conv_fn

def create_back_conv_z_plus_fn():
    inputs = T.ftensor3()
    weights = T.ftensor4()
    out_relevances = T.ftensor3()
    in_relevances = relevance_conv_z_plus(out_relevances, inputs, weights)
    back_relevance_conv_fn = theano.function([out_relevances, inputs, weights],
                                         in_relevances)
    return back_relevance_conv_fn

def create_back_conv_z_b_fn(min_in, max_in):
    inputs = T.ftensor3()
    weights = T.ftensor4()
    out_relevances = T.ftensor3()
    in_relevances = relevance_conv_z_b(out_relevances, inputs, weights,
        min_in, max_in)
    back_relevance_conv_fn = theano.function([out_relevances, inputs, weights],
                                         in_relevances)
    return back_relevance_conv_fn

def create_back_dense_fn(rule, min_in=None, max_in=None):
    inputs = T.vector()
    weights = T.fmatrix()
    out_relevances = T.fvector()
    in_relevances = relevance_dense(out_relevances, inputs, weights,
        rule, min_in, max_in)
    if rule == 'w_sqr':
        back_relevance_dense_fn = theano.function([out_relevances, weights],
                                         in_relevances)
    else:
        back_relevance_dense_fn = theano.function([out_relevances, inputs,
            weights], in_relevances)
    return back_relevance_dense_fn

def relevance_pool(out_relevances, inputs, pool_size, pool_stride):
    pool_ones_shape = [1, out_relevances.shape[0], pool_size[0], pool_size[1]]
    pool_ones = T.ones(pool_ones_shape, dtype=np.float32)
    norms_for_relevances = conv2d(inputs.dimshuffle('x',0,1,2), 
                           pool_ones, subsample=pool_stride, 
                           border_mode='valid')[0]
    # prevent division by 0...
    # the relevance which had norm zero will not be redistributed anyways..
    # so it doesnt matter which normlaization factor you choose ehre,
    # only thing is to prevent NaNs...
    # however this means heatmapping is no longer completely preserving
    # 
    norms_for_relevances += T.eq(norms_for_relevances, 0) * 1
    normed_relevances = out_relevances / norms_for_relevances
    # stride has to be taken into account, see 
    # http://stackoverflow.com/a/28752057/1469195
    upsampled_relevances = T.zeros((normed_relevances.shape[0], 
        normed_relevances.shape[1] * pool_stride[0] - pool_stride[0] + 1, 
        normed_relevances.shape[2] * pool_stride[1] - pool_stride[1] + 1, 
        ), dtype=np.float32)
    upsampled_relevances = T.set_subtensor(
        upsampled_relevances[:, ::pool_stride[0], ::pool_stride[1]], 
        normed_relevances)
    
    in_relevances = conv2d(upsampled_relevances.dimshuffle('x',0,1,2), 
                           pool_ones, subsample=(1,1),
                           border_mode='full')[0]
    in_relevances = in_relevances * inputs
    return in_relevances

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

def back_relevance_conv(out_relevances, in_activations, conv_weights, rule,
        min_in=None, max_in=None):
    assert rule in ['w_sqr', 'z_plus', 'z_b']
    if rule == 'z_b':
        assert min_in is not None
        assert max_in is not None
        assert min_in <= 0
        assert max_in >= 0
    # for tests if you want to use int numbers
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
                if rule == 'z_b':
                    relevant_input = in_activations[:,out_x:out_x+kernel_size[0],
                              out_y:out_y+kernel_size[1]]
                    adapted_weights = relevant_weights * relevant_input
                    # will be positive as min in is negative....
                    offset_negative_in = (-relevant_weights *
                        (relevant_weights > 0) * min_in)
                    # will be positive as max in is positive....
                    offset_positive_in = (-relevant_weights *
                        (relevant_weights < 0) * max_in)
                    adapted_weights += offset_negative_in + offset_positive_in
                scaled_weights = adapted_weights / float(np.sum(adapted_weights))
                if np.sum(adapted_weights) == 0:
                    scaled_weights[:] = 1 / float(np.prod(scaled_weights.shape))
                relevance_to_add = scaled_weights * out_relevances[out_filt, out_x, out_y] 
                in_relevances[:,out_x:out_x+kernel_size[0],
                              out_y:out_y+kernel_size[1]] += relevance_to_add
    return in_relevances

def back_relevance_pool(out_relevances, in_activations, pool_size,
        pool_stride):
    # for tests if you want to use int numbers
    out_relevances = np.array(out_relevances, dtype=np.float32)
    in_activations = np.array(in_activations, dtype=np.float32)
    assert in_activations.shape[0] == out_relevances.shape[0]

    in_relevances = np.zeros(in_activations.shape, dtype=np.float32)
    for out_filt in xrange(out_relevances.shape[0]):
        for out_x in xrange(out_relevances.shape[1]):
            for out_y in xrange(out_relevances.shape[2]):
                in_x = out_x * pool_stride[0]
                in_y = out_y * pool_stride[1]
                in_x_stop = in_x+pool_size[0]
                in_y_stop = in_y+pool_size[1]
                relevant_inputs = in_activations[out_filt, in_x:in_x_stop,
                    in_y:in_y_stop]
                scaled_inputs = relevant_inputs / np.sum(relevant_inputs)
                if np.sum(relevant_inputs) == 0:
                    scaled_inputs = relevant_inputs + 1/relevant_inputs.size
                scaled_relevance = scaled_inputs * out_relevances[out_filt,
                    out_x, out_y]
                in_relevances[out_filt, in_x:in_x_stop, 
                    in_y:in_y_stop] += scaled_relevance
    return in_relevances
