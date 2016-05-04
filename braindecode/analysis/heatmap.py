import theano.tensor as T
import numpy as np
import theano
import lasagne
from theano.tensor.nnet import conv2d
from braindecode.veganlasagne.layers import get_input_shape, \
    StrideReshapeLayer
from braindecode.veganlasagne.batch_norm import BatchNormLayer
import sklearn.preprocessing

def create_heatmap_fn(all_layers, rules, min_in=None, max_in=None,
        return_all=False, use_output_as_relevance=False):
    # only using single trial, so one less dim than input shape
    if len(get_input_shape(all_layers[-1])) == 2:
        input_trials = T.fmatrix()
    elif len(get_input_shape(all_layers[-1])) == 4:
        input_trials = T.ftensor4()
    if use_output_as_relevance:
        out_relevances = lasagne.layers.get_output(all_layers[-1], 
            inputs=input_trials, deterministic=True, input_var=input_trials)
    else:
        if len(all_layers[-1].output_shape) == 2:
            out_relevances = T.fmatrix()
        elif len(all_layers[-1].output_shape) == 4:
            out_relevances = T.ftensor4()
    heatmap = create_heatmap(out_relevances, input_trials, all_layers, rules,
                            min_in, max_in, return_all)
    if use_output_as_relevance:
        heatmap_fn = theano.function([input_trials], heatmap)
    else:
        heatmap_fn = theano.function([out_relevances, input_trials], heatmap)         
    return heatmap_fn

def create_heatmap(out_relevances, input_trials, all_layers, 
        all_rules, min_in=None, max_in=None, return_all=False):
    """Theano expression for computing the heatmap.
    Expects a single input trial not a batch. Similarly a single output relevance"""
    # First make all rules correct in case just rule for first layer
    # and remaining layers given
    
    if len(all_rules) == 2 and (not len(all_layers) == 2):
        # then expect that there is rule for layers until first weights
        # and after
        real_all_rules = []
        i_layer = 0
        first_weight_found = False
        while not first_weight_found:
            real_all_rules.append(all_rules[0])
            layer = all_layers[i_layer]
            if hasattr(layer, 'W'):
                first_weight_found = True
            i_layer += 1
        # make remaning layers use 2nd rule
        real_all_rules.extend([all_rules[1]] * (len(all_layers) - i_layer))
        all_rules = real_all_rules
        
    for rule in all_rules:
        assert rule in ['w_sqr', 'z_plus', 'z_b', 'adapt_z_b', None]
    assert len(all_rules) == len(all_layers), ("number of rules "
        "{:d} number layers {:d}".format(len(all_rules), len(all_layers)))
    
    activations_per_layer = lasagne.layers.get_output(all_layers, 
         input_trials, deterministic=True, input_var=input_trials)
    
    assert len(activations_per_layer) == len(all_layers)
    
    all_heatmaps = []
    # stop before first layer...as it should be input layer...
    # and we always need activations from before
    for i_layer in xrange(len(all_layers)-1, 0,-1):
        # We have out relevance for that layer, now 
        layer = all_layers[i_layer]
        in_activations = activations_per_layer[i_layer-1]
        rule = all_rules[i_layer]
        if isinstance(layer, lasagne.layers.DenseLayer):
            if in_activations.ndim > 2:
                in_act_flat = in_activations.flatten(2)
            out_relevances = relevance_dense(out_relevances,
                 in_act_flat, layer.W, rule, min_in, max_in)
        elif hasattr(layer, 'pool_size'):
            out_relevances = relevance_pool(out_relevances, in_activations,
                 pool_size=layer.pool_size, pool_stride=layer.stride)
        elif hasattr(layer, 'filter_size'):
            out_relevances = relevance_conv(out_relevances, in_activations,
                layer.W, rule, min_in, max_in)
        elif isinstance(layer, StrideReshapeLayer):
            out_relevances = relevance_back_by_grad(out_relevances,
                in_activations, layer)
        elif isinstance(layer, lasagne.layers.DimshuffleLayer):
            pattern = layer.pattern
            reverse_pattern = [pattern.index(i) for i in range(len(pattern))]
            out_relevances = out_relevances.dimshuffle(reverse_pattern)
        elif (isinstance(layer, lasagne.layers.DropoutLayer) or
            isinstance(layer, lasagne.layers.NonlinearityLayer) or
            isinstance(layer, lasagne.layers.FlattenLayer)or
            isinstance(layer, BatchNormLayer)):
            pass
        else:
            raise ValueError("Trying to propagate through unknown layer "
                "{:s}".format(layer.__class__.__name__))
        if out_relevances.shape != in_activations.shape:
            out_relevances = out_relevances.reshape(in_activations.shape)
        all_heatmaps.append(out_relevances)
    if return_all:
        return all_heatmaps[::-1]
    else:
        return all_heatmaps[-1]

def relevance_conv(out_relevances, inputs, weights, rule, min_in=None,
        max_in=None):
    assert rule in ['w_sqr', 'z_plus', 'z_b', 'adapt_z_b']
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
    elif rule == 'adapt_z_b':
        # clip to zero both min and max to prevent mistakes...
        min_in = T.min(inputs)
        min_in = T.minimum(0, min_in)
        max_in = T.max(inputs)
        max_in = T.maximum(0, max_in)
        return relevance_conv_z_b(out_relevances, inputs, weights,
            min_in, max_in)

def relevance_conv_w_sqr(out_relevances, weights, biases=None):
    weights_sqr = weights * weights
    weights_norm = T.sum(weights_sqr, axis=(1,2,3), keepdims=True)
    # prevent division by zero
    weights_norm = weights_norm + (T.eq(weights_norm, 0) * 1)
    weights_scaled = weights_sqr / weights_norm
    # upconv
    in_relevances = conv2d(out_relevances, 
                           weights_scaled.dimshuffle(1,0,2,3)[:,:,::-1,::-1],
                           border_mode='full')
    return in_relevances

def relevance_conv_z_plus(out_relevances, inputs, weights):
    # hack for negative inputs
    #inputs = T.abs_(inputs)
    weights_plus = weights * T.gt(weights, 0)
    norms_for_relevances = conv2d(inputs, weights_plus)
    # prevent division by 0...
    # adds 1 to every entry that is 0 -> sets 0s to 1
    relevances_are_0 = T.eq(norms_for_relevances, 0)
    norms_for_relevances += relevances_are_0 * 1
    
    normed_relevances = out_relevances / norms_for_relevances
    # upconv
    in_relevances = conv2d(normed_relevances, 
                           weights_plus.dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
                           border_mode='full')
   
    in_relevances_proper = in_relevances * inputs
    
    # Correct for those parts where all inputs of a relevance were
    # zero, spread relevance equally them
    pool_ones = T.ones(weights_plus.shape, dtype=np.float32)
    # mean across channel, 0, 1 dims (hope this is correct?)
    pool_fractions = pool_ones / T.prod(weights_plus.shape[1:]).astype(
        theano.config.floatX)
    in_relevances_from_0 = conv2d(out_relevances * relevances_are_0, 
                           pool_fractions.dimshuffle(1,0,2,3), 
                           subsample=(1,1),
                           border_mode='full')
     
    in_relevances_proper += in_relevances_from_0
    
    
    return in_relevances_proper

def relevance_conv_a_b(out_relevances, inputs, weights):
    return "TODO"

def relevance_conv_z_b(out_relevances, inputs, weights, min_in, max_in):
    #assert min_in <= 0
    #assert max_in >= 0
    weights_b = T.lt(weights, 0) * weights * -max_in
    weights_b += T.gt(weights, 0) * weights * -min_in

    norms_for_relevances = conv2d(inputs, weights)
    norms_for_relevances += T.sum(weights_b, axis=(1,2,3)).dimshuffle(
        'x',0,'x','x')
    # prevent division by 0...
    norms_for_relevances += T.eq(norms_for_relevances, 0) * 1
    normed_relevances = out_relevances / norms_for_relevances
    # upconv data
    in_relevances_data = conv2d(normed_relevances, 
                           weights.dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
                           border_mode='full')
    in_relevances_data *= inputs
    # upconv weight offsets to enforce positivity
    in_relevances_b = conv2d(normed_relevances, 
                           weights_b.dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
                           border_mode='full')
    in_relevances = in_relevances_data + in_relevances_b
    return in_relevances

def relevance_dense(out_relevances, in_activations, weights, rule,
    min_in=None, max_in=None):
    """Party copied from paper supplementary pseudocode:
    http://arxiv.org/abs/1512.02479
    Partly adapted for propagating multiple trials at once"""
    assert rule in ['w_sqr', 'z_plus', 'z_b']
    # weights are features x output_units => input_units x output_units
    # in_activations are trials x input_units
    if rule == 'w_sqr':
        W_adapted = weights * weights
        Z = T.sum(W_adapted, axis=0, keepdims=True)
        # prevent division by 0...
        Z += T.eq(Z, 0) * 1
        N = W_adapted / Z
        return T.dot(N, out_relevances.T).T
    elif rule == 'z_plus':
        weights_plus = weights * T.gt(weights, 0)
        norms_for_relevances = T.dot(in_activations, weights_plus)
        # prevent division by zero
        norms_for_relevances += T.eq(norms_for_relevances, 0) * 1
        normed_relevances = out_relevances / norms_for_relevances
        in_relevances = T.dot(normed_relevances, weights_plus.T)
        in_relevances = in_relevances * in_activations
        return in_relevances
    elif rule == 'z_b':
        assert min_in is not None
        assert max_in is not None
        assert min_in <= 0
        assert max_in >= 0
        
        weights_plus = weights * T.gt(weights, 0)
        weights_minus = weights * T.lt(weights, 0)
        Z_I_J = weights.dimshuffle('x',0,1) * in_activations.dimshuffle(
            0,1,'x')
        Z_I_J -= weights_plus.dimshuffle('x',0,1) * min_in
        Z_I_J -= weights_minus.dimshuffle('x',0,1) * max_in
        Z_I_J = Z_I_J / T.sum(Z_I_J, axis=1, keepdims=True)
        in_relevances = Z_I_J * out_relevances.dimshuffle(0,'x',1)
        in_relevances = T.sum(in_relevances, axis=2)
        return in_relevances
        
        
        
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

def relevance_pool(out_relevances, inputs, pool_size, pool_stride):
    # channels x channels x pool_0 x pool_1
    pool_ones_shape = [out_relevances.shape[1], out_relevances.shape[1],
        pool_size[0], pool_size[1]]
    # modification: make inputs positive
    inputs = T.abs_(inputs)
    pool_ones = T.ones(pool_ones_shape, dtype=np.float32)
    # only within a channel spread values of that channel...
    # therefore set all values of indices like
    # filt_i, channel_j with j!=i to zero!
    pool_ones = pool_ones * T.eye(out_relevances.shape[1],
                              out_relevances.shape[1]).dimshuffle(
                                 0,1,'x','x')
    norms_for_relevances = conv2d(inputs, 
                           pool_ones, subsample=pool_stride, 
                           border_mode='valid')
    # prevent division by 0...
    # the relevance which had norm zero will not be redistributed anyways..
    # so it doesnt matter which normalization factor you choose here,
    # only thing is to prevent NaNs...
    # however this means heatmapping is no longer completely preserving
    # 
    norms_for_relevances += T.eq(norms_for_relevances, 0) * 1
    normed_relevances = out_relevances / norms_for_relevances
    # stride has to be taken into account, see 
    # http://stackoverflow.com/a/28752057/1469195
    upsampled_relevances = T.zeros((normed_relevances.shape[0],
        normed_relevances.shape[1], 
        normed_relevances.shape[2] * pool_stride[0] - pool_stride[0] + 1, 
        normed_relevances.shape[3] * pool_stride[1] - pool_stride[1] + 1, 
        ), dtype=np.float32)
    upsampled_relevances = T.set_subtensor(
        upsampled_relevances[:, :, :pool_stride[0], ::pool_stride[1]], 
        normed_relevances)
    
    in_relevances = conv2d(upsampled_relevances,
                           pool_ones, subsample=(1,1),
                           border_mode='full')
    in_relevances = in_relevances * inputs
    return in_relevances

def relevance_back_by_grad(out_relevances, in_activations, layer):
    out = lasagne.layers.get_output(layer, deterministic=True,
        inputs={layer.input_layer: in_activations})
    in_relevance = T.grad(None, in_activations,
        known_grads={out:out_relevances})
    return in_relevance

def create_back_conv_w_sqr_fn():
    out_relevances = T.ftensor4()
    weights = T.ftensor4()
    in_relevances = relevance_conv_w_sqr(out_relevances, weights)
    back_relevance_conv_fn = theano.function([out_relevances, weights],
                                         in_relevances)
    return back_relevance_conv_fn

def create_back_conv_z_plus_fn():
    inputs = T.ftensor4()
    weights = T.ftensor4()
    out_relevances = T.ftensor4()
    in_relevances = relevance_conv_z_plus(out_relevances, inputs, weights)
    back_relevance_conv_fn = theano.function([out_relevances, inputs, weights],
                                         in_relevances)
    return back_relevance_conv_fn

def create_back_conv_z_b_fn(min_in, max_in):
    inputs = T.ftensor4()
    weights = T.ftensor4()
    out_relevances = T.ftensor4()
    in_relevances = relevance_conv_z_b(out_relevances, inputs, weights,
        min_in, max_in)
    back_relevance_conv_fn = theano.function([out_relevances, inputs, weights],
                                         in_relevances)
    return back_relevance_conv_fn

def create_back_dense_fn(rule, min_in=None, max_in=None):
    inputs = T.fmatrix()
    weights = T.fmatrix()
    out_relevances = T.fmatrix()
    in_relevances = relevance_dense(out_relevances, inputs, weights,
        rule, min_in, max_in)
    if rule == 'w_sqr':
        back_relevance_dense_fn = theano.function([out_relevances, weights],
                                         in_relevances)
    else:
        back_relevance_dense_fn = theano.function([out_relevances, inputs,
            weights], in_relevances)
    return back_relevance_dense_fn

def compute_all_epo_relevances(dataset, pred_fn, heatmap_fn,n_classes):
    topo = dataset.get_topological_view()
    ys = sklearn.preprocessing.OneHotEncoder(n_classes, 
     sparse=False).fit_transform(dataset.y[:,np.newaxis])

    all_relevances = []
    for i_trial in xrange(len(topo)):
        trial_topo = topo[i_trial:i_trial+1]

        pred = pred_fn(trial_topo)[0]
        # only take correct part of prediction
        pred = pred * ys[i_trial]
        trial_relevances = heatmap_fn(pred.astype(np.float32),
            trial_topo[0])
        all_relevances.append(trial_relevances)
    all_relevances = np.array(all_relevances)
    return all_relevances
