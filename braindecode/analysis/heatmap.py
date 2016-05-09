import theano.tensor as T
import numpy as np
import theano
import lasagne
from theano.tensor.nnet import conv2d
import sklearn.preprocessing
import matplotlib.pyplot as plt
from numpy.random import RandomState
from braindecode.veganlasagne.layers import get_input_shape, \
    StrideReshapeLayer
from braindecode.veganlasagne.batch_norm import BatchNormLayer
import logging
log = logging.getLogger(__name__)

def create_heatmap_fn(all_layers, rules, pool_by_grad, min_in=None, max_in=None,
        return_all=False, use_output_as_relevance=False, a=None, b=None,
        biases=False):
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
                            pool_by_grad=pool_by_grad,
                            min_in=min_in, max_in=max_in, return_all=return_all,
                            a=a,b=b,
                            biases=biases)
    if use_output_as_relevance:
        heatmap_fn = theano.function([input_trials], heatmap)
    else:
        heatmap_fn = theano.function([out_relevances, input_trials], heatmap)         
    return heatmap_fn

def create_heatmap(out_relevances, input_trials, all_layers, 
        all_rules, pool_by_grad, min_in=None, max_in=None, return_all=False,
        a=None, b=None, biases=False):
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
        assert rule in ['w_sqr', 'z_plus', 'z_b', 'adapt_z_b', 'sign_stable', 
            'a_b','z', 'a_b_sign_switch', 'a_b_abs', None]
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
            else:
                in_act_flat = in_activations
            bias = None
            if biases == True:
                bias = layer.b
            out_relevances = relevance_dense(out_relevances,
                 in_act_flat, layer.W, rule, min_in, max_in, a=a,b=b,
                 bias=bias)
        elif hasattr(layer, 'pool_size'):
            if pool_by_grad:
                out_relevances = relevance_back_by_grad(out_relevances,
                    in_activations, layer)
            else:
                out_relevances = relevance_pool(out_relevances, in_activations,
                    pool_size=layer.pool_size, pool_stride=layer.stride)
        elif hasattr(layer, 'filter_size'):
            
            bias = None
            if biases == True:
                bias = layer.b
            out_relevances = relevance_conv(out_relevances, in_activations,
                layer.W, rule=rule, min_in=min_in, max_in=max_in,
                a=a,b=b,bias=bias)
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

def relevance_conv(out_relevances, inputs, weights, rule, bias=None, min_in=None,
        max_in=None, a=None, b=None):
    assert rule in ['w_sqr', 'z', 'z_plus', 'z_b', 'adapt_z_b', 'sign_stable',
        'a_b', 'a_b_sign_switch', 'a_b_abs']
    if rule == 'w_sqr':
        return relevance_conv_w_sqr(out_relevances, weights, bias=bias)
    elif rule == 'z_plus':
        return relevance_conv_z_plus(out_relevances, inputs, weights, bias=bias)
    elif rule == 'z_b':
        assert min_in is not None
        assert max_in is not None
        assert min_in <= 0
        assert max_in >= 0
        return relevance_conv_z_b(out_relevances, inputs, weights,
            min_in, max_in, bias=bias)
    elif rule == 'adapt_z_b':
        # clip to zero both min and max to prevent mistakes...
        min_in = T.min(inputs)
        min_in = T.minimum(0, min_in)
        max_in = T.max(inputs)
        max_in = T.maximum(0, max_in)
        return relevance_conv_z_b(out_relevances, inputs, weights,
            min_in, max_in, bias=bias)
    elif rule == 'sign_stable':
        return relevance_conv_stable_sign(inputs, weights, out_relevances,
            bias=bias)
    elif rule == 'a_b':
        return relevance_conv_a_b(inputs, weights, out_relevances, 
            a=a,b=b, bias=bias)
    elif rule == 'z':
        return relevance_conv_z(out_relevances, inputs, weights, 
            bias=bias)
    elif rule == 'a_b_sign_switch':
        return relevance_conv_a_b_sign_switch(inputs, weights, out_relevances, 
            a=a,b=b, bias=bias)
    elif rule == 'a_b_abs':
        return relevance_conv_a_b_abs(inputs, weights, out_relevances, 
            a=a,b=b, bias=bias)
        

def relevance_conv_w_sqr(out_relevances, weights, bias=None):
    if bias is not None:
        log.warning("Bias not respected for conv w_sqr")
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

def relevance_conv_z_plus(out_relevances, inputs, weights, bias=None):
    if bias is not None:
        log.warning("Bias not respected for conv z_plus")
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


def relevance_conv_z(out_relevances, inputs, weights, bias=None):
    norms_for_relevances = conv2d(inputs, weights)
    if bias is not None:
        norms_for_relevances +=  bias.dimshuffle('x',0,'x','x')
    # stabilize
    # prevent division by 0 and division by small numbers
    eps = 1e-4
    norms_for_relevances += (T.sgn(norms_for_relevances) * eps)
    norms_for_relevances += (T.eq(norms_for_relevances, 0) * eps)
    
    normed_relevances = out_relevances / norms_for_relevances
    # upconv
    in_relevances = conv2d(normed_relevances, 
                           weights.dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
                           border_mode='full')
   
    in_relevances_proper = in_relevances * inputs
    
    if bias is not None:
        bias_relevance = bias.dimshuffle('x',0,'x','x') * normed_relevances
        # Divide bias by weight size before convolving back
        # mean across channel, 0, 1 dims (hope this is correct?)
        fraction_bias = bias_relevance / T.prod(weights.shape[1:]).astype(
            theano.config.floatX)
        bias_rel_in = conv2d(fraction_bias, 
          T.ones_like(weights).dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
          border_mode='full')
        in_relevances_proper +=  bias_rel_in
    
    return in_relevances_proper


def relevance_conv_z_b(out_relevances, inputs, weights, min_in, max_in, bias=None):
    # min in /max in can be symbolic or number, so no way to check
    # any assertions here  
    if bias is not None:
        log.warning("Bias not respected for conv z_b")
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


def _forward_positive_z(inputs, weights, bias=None):
    inputs_plus = inputs * T.gt(inputs, 0)
    weights_plus = weights * T.gt(weights, 0)
    inputs_minus = inputs * T.lt(inputs, 0)
    weights_minus = weights * T.lt(weights, 0)
    
    plus_part_a = conv2d(inputs_plus, weights_plus)
    plus_part_b = conv2d(inputs_minus, weights_minus)
    together = plus_part_a + plus_part_b
    if bias is not None:
        bias_plus = bias * T.gt(bias, 0)
        together +=  bias_plus.dimshuffle('x',0,'x','x')
    
    return together
    
def _forward_negative_z(inputs, weights, bias=None):
    inputs_plus = inputs * T.gt(inputs, 0)
    weights_plus = weights * T.gt(weights, 0)
    inputs_minus = inputs * T.lt(inputs, 0)
    weights_minus = weights * T.lt(weights, 0)
    negative_part_a = conv2d(inputs_plus, weights_minus)
    negative_part_b = conv2d(inputs_minus, weights_plus)
    together = negative_part_a + negative_part_b
    if bias is not None:
        bias_negative = bias * T.lt(bias, 0)
        together +=  bias_negative.dimshuffle('x',0,'x','x')
    
    return together

def _backward_positive_z(inputs, weights, normed_relevances, bias=None):
    inputs_plus = inputs * T.gt(inputs, 0)
    weights_plus = weights * T.gt(weights, 0)
    inputs_minus = inputs * T.lt(inputs, 0)
    weights_minus = weights * T.lt(weights, 0)
    # Compute weights+ * inputs+ and weights- * inputs-
    positive_part_a = conv2d(normed_relevances, 
          weights_plus.dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
          border_mode='full')
    positive_part_a *= inputs_plus
    positive_part_b = conv2d(normed_relevances, 
          weights_minus.dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
          border_mode='full')
    positive_part_b *= inputs_minus
    together = positive_part_a + positive_part_b
    if bias is not None:
        bias_plus = bias * T.gt(bias, 0)
        bias_relevance = bias_plus.dimshuffle('x',0,'x','x') * normed_relevances
        # Divide bias by weight size before convolving back
        # mean across channel, 0, 1 dims (hope this is correct?)
        fraction_bias = bias_relevance / T.prod(weights.shape[1:]).astype(
            theano.config.floatX)
        bias_rel_in = conv2d(fraction_bias, 
          T.ones_like(weights).dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
          border_mode='full') 
        together +=  bias_rel_in
    return together
    
def _backward_negative_z(inputs, weights, normed_relevances, bias=None):
    inputs_plus = inputs * T.gt(inputs, 0)
    weights_plus = weights * T.gt(weights, 0)
    inputs_minus = inputs * T.lt(inputs, 0)
    weights_minus = weights * T.lt(weights, 0)
    # Compute weights+ * inputs- and weights- * inputs+
    negative_part_a = conv2d(normed_relevances, 
          weights_plus.dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
          border_mode='full')
    negative_part_a *= inputs_minus
    negative_part_b = conv2d(normed_relevances, 
          weights_minus.dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
          border_mode='full')
    negative_part_b *= inputs_plus
        
    together = negative_part_a + negative_part_b
    if bias is not None:
        bias_negative = bias * T.lt(bias, 0)
        bias_relevance = bias_negative.dimshuffle('x',0,'x','x') * normed_relevances
        # Divide bias by weight size before convolving back
        # mean across channel, 0, 1 dims (hope this is correct?)
        fraction_bias = bias_relevance / T.prod(weights.shape[1:]).astype(
            theano.config.floatX)
        bias_rel_in = conv2d(fraction_bias, 
          T.ones_like(weights).dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
          border_mode='full') 
        together +=  bias_rel_in
    return together
    
    
def relevance_conv_stable_sign(inputs, weights, out_relevances, bias=None):
    negative_norm = _forward_negative_z(inputs, weights, bias=bias)
    positive_norm = _forward_positive_z(inputs, weights, bias=bias)
    
    # Set zeros to 1 to prevent division by 0
    positive_norm_nonzero = positive_norm + T.eq(positive_norm, 0)
    pos_minus_neg_nonzero = (positive_norm - negative_norm)
    pos_minus_neg_nonzero +=  T.eq(pos_minus_neg_nonzero, 0)
    positive_factor = (positive_norm - 2 * negative_norm) / (
        positive_norm_nonzero * pos_minus_neg_nonzero)
    
    positive_rel_normed = out_relevances * positive_factor
    negative_rel_normed = out_relevances / pos_minus_neg_nonzero
    
    in_rel_from_pos = _backward_positive_z(inputs, weights,
        positive_rel_normed, bias=bias)
    in_rel_from_neg = _backward_negative_z(inputs, weights, 
        negative_rel_normed, bias=bias)
    return in_rel_from_pos + in_rel_from_neg

def relevance_conv_a_b(inputs, weights, out_relevances, a,b, bias=None):
    assert a is not None
    assert b is not None
    assert a - b == 1
    positive_norm = _forward_positive_z(inputs, weights, bias=bias)
    negative_norm = _forward_negative_z(inputs, weights, bias=bias)
    # set 0s to 1
    positive_norm_nonzero = positive_norm + T.eq(positive_norm, 0)
    # set 0s to -1
    negative_norm_nonzero = negative_norm - T.eq(negative_norm, 0)
    
    positive_rel_normed = out_relevances / positive_norm_nonzero
    # relevances now already negative :))
    negative_rel_normed = out_relevances / negative_norm_nonzero
    
    in_rel_from_pos = _backward_positive_z(inputs, weights, 
        positive_rel_normed, bias=bias)
    in_rel_from_neg = _backward_negative_z(inputs, weights,
        negative_rel_normed, bias=bias)
    return a * in_rel_from_pos - b * in_rel_from_neg


def relevance_conv_a_b_sign_switch(inputs, weights, out_relevances, a,b, bias=None):
    assert a is not None
    assert b is not None
    assert a - b == 1
    # For each input, determine what 
    outputs = conv2d(inputs, weights)
    if bias is not None:
        outputs += bias.dimshuffle('x',0,'x','x')
        # do not use bias further, only to determine direction of outputs
        bias = None
    # stabilize
    # prevent division by 0 and division by small numbers
    eps = 1e-4
    outputs += (T.sgn(outputs) * eps)
    outputs += (T.eq(outputs, 0) * eps)
    positive_forward = _forward_positive_z(inputs, weights, bias)
    negative_forward = _forward_negative_z(inputs, weights, bias)
    rel_for_positive_outputs = out_relevances * T.gt(outputs, 0)
    rel_for_negative_outputs = out_relevances * T.lt(outputs, 0)
    
    positive_norm_with_trend = positive_forward * T.gt(outputs, 0)
    negative_norm_with_trend = negative_forward * T.lt(outputs, 0)
    # minus to make overall norm positive
    norm_with_trend = positive_norm_with_trend - negative_norm_with_trend
    # stabilize also 
    norm_with_trend += (T.eq(norm_with_trend, 0) * eps)
    
    in_positive_with_trend = _backward_positive_z(inputs, weights, 
        rel_for_positive_outputs  / norm_with_trend, bias)
    in_negative_with_trend = _backward_negative_z(inputs, weights, 
        rel_for_negative_outputs  / norm_with_trend, bias)
    
    # Minus in_negative since in_with_trend should not switch signs
    in_with_trend = in_positive_with_trend - in_negative_with_trend
    
    positive_norm_against_trend = positive_forward * T.lt(outputs, 0)
    negative_norm_against_trend = negative_forward * T.gt(outputs, 0)
    # minus to make overall norm positive
    norm_against_trend = positive_norm_against_trend - negative_norm_against_trend
    # stabilize also 
    norm_against_trend += (T.eq(norm_against_trend, 0) * eps)
    
    in_positive_against_trend = _backward_positive_z(inputs, weights, 
        rel_for_negative_outputs  / norm_against_trend, bias)
    in_negative_against_trend = _backward_negative_z(inputs, weights, 
        rel_for_positive_outputs  / norm_against_trend, bias)
    # Minus in_negative since switching signs is done below
    in_against_trend = in_positive_against_trend - in_negative_against_trend
    
    in_relevances = a * in_with_trend - b * in_against_trend
    return in_relevances

def relevance_conv_a_b_abs(inputs, weights, out_relevances, a,b, bias=None):
    assert a is not None
    assert b is not None
    assert a - b == 1
    weights_plus = weights * T.gt(weights, 0)
    weights_neg = weights * T.lt(weights, 0)
    
    plus_norm = conv2d(T.abs_(inputs), weights_plus)
    # stabilize, prevent division by 0
    eps=1e-4
    plus_norm += (T.eq(plus_norm,0) * eps)
    plus_rel_normed = out_relevances / plus_norm
    in_rel_plus = conv2d(plus_rel_normed, 
          weights_plus.dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
          border_mode='full')
    in_rel_plus *= T.abs_(inputs)
    
    # minuses to get positive outputs, since will be subtracted
    # at end of function
    neg_norm = -conv2d(T.abs_(inputs), weights_neg)
    neg_norm += (T.eq(neg_norm,0) * eps)
    neg_rel_normed = out_relevances / neg_norm
    in_rel_neg = -conv2d(neg_rel_normed, 
          weights_neg.dimshuffle(1,0,2,3)[:,:,::-1,::-1], 
          border_mode='full')
    in_rel_neg *= T.abs_(inputs)

    in_relevance = a * in_rel_plus - b * in_rel_neg
    return in_relevance

def relevance_dense(out_relevances, in_activations, weights, rule,
    min_in=None, max_in=None, a=None, b=None, bias=None):
    """Party copied from paper supplementary pseudocode:
    http://arxiv.org/abs/1512.02479"""
    assert rule in ['w_sqr', 'z', 'z_plus', 'z_b', 'adapt_z_b','sign_stable',
        'a_b', 'a_b_sign_switch', 'a_b_abs']
    # weights are features x output_units => input_units x output_units
    # in_activations are trials x input_units
    if rule == 'w_sqr':
        if bias is not None:
            log.warning("Bias not respected for dense w_sqr")
        W_sqr = weights * weights
        Z = T.sum(W_sqr, axis=0, keepdims=True)
        # prevent division by 0...
        Z += T.eq(Z, 0) * 1
        N = W_sqr / Z
        return T.dot(N, out_relevances.T).T
    elif rule == 'z':
        Z_I_J = weights.dimshuffle('x',0,1) * in_activations.dimshuffle(
            0,1,'x')
        if bias is not None:
            bias = bias.dimshuffle('x','x', 0)
            # redistribute bias proportionally across all inputs
            Z_I_J += (bias / weights.shape[0])
        norms_for_relevances = T.sum(Z_I_J, axis=1, keepdims=True)
        # stabilize, prevent division by 0 and small values
        eps = 1e-3
        norms_for_relevances += (T.sgn(norms_for_relevances) * eps)
        norms_for_relevances += (T.eq(norms_for_relevances, 0) * eps)
        Z_I_J = Z_I_J / norms_for_relevances
        in_relevances = Z_I_J * out_relevances.dimshuffle(0,'x',1)
        in_relevances = T.sum(in_relevances, axis=2)
        return in_relevances
    elif rule == 'z_plus':
        if bias is not None:
            log.warning("Bias not respected for dense z_plus")
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
        if bias is not None:
            log.warning("Bias not respected for dense z_b")
        
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
    elif rule == 'adapt_z_b':
        if bias is not None:
            log.warning("Bias not respected for dense adapt_z_b")
        # clip to zero both min and max to prevent mistakes...
        min_in = T.min(in_activations)
        min_in = T.minimum(0, min_in)
        max_in = T.max(in_activations)
        max_in = T.maximum(0, max_in)
        
        weights_plus = weights * T.gt(weights, 0)
        weights_minus = weights * T.lt(weights, 0)
        # TODO: change to simply add minimum of Z_I_J? much simpler?
        # rather than taking weights plus * min in etc?
        # if we are doing adapt anyways
        Z_I_J = weights.dimshuffle('x',0,1) * in_activations.dimshuffle(
            0,1,'x')
        Z_I_J -= weights_plus.dimshuffle('x',0,1) * min_in
        Z_I_J -= weights_minus.dimshuffle('x',0,1) * max_in
        Z_I_J = Z_I_J / T.sum(Z_I_J, axis=1, keepdims=True)
        in_relevances = Z_I_J * out_relevances.dimshuffle(0,'x',1)
        in_relevances = T.sum(in_relevances, axis=2)
        return in_relevances
    elif rule == 'sign_stable':
        Z_I_J = weights.dimshuffle('x',0,1) * in_activations.dimshuffle(
            0,1,'x')
        Z_I_J_plus = Z_I_J * T.gt(Z_I_J,0)
        Z_I_J_neg = Z_I_J * T.lt(Z_I_J,0)
        if bias is not None:
            bias = bias.dimshuffle('x','x', 0)
            # redistribute bias proportionally across all inputs
            Z_I_J_plus += (bias * T.gt(bias,0)) / weights.shape[0]
            Z_I_J_neg += (bias * T.lt(bias,0)) / weights.shape[0]
        plus_sum = Z_I_J_plus.sum(axis=1, keepdims=True)
        neg_sum = Z_I_J_neg.sum(axis=1, keepdims=True)
        plus_minus_neg_sum = plus_sum - neg_sum
        # set 0s to 1
        plus_minus_neg_sum += T.eq(plus_minus_neg_sum, 0)
        # set 0s to 1
        plus_non_neg_sum = plus_sum + T.eq(0,plus_sum)
        plus_factor = (plus_sum - 2 * neg_sum) / (
            plus_non_neg_sum * plus_minus_neg_sum)
        Z_I_J_plus_normed = plus_factor * Z_I_J_plus
        Z_I_J_neg_normed = Z_I_J_neg / plus_minus_neg_sum
        Z_I_J_normed = (Z_I_J_plus_normed + Z_I_J_neg_normed)
        in_relevances = out_relevances.dimshuffle(0,'x',1) * Z_I_J_normed
        in_relevances = T.sum(in_relevances, axis=2)
        return in_relevances
    elif rule == 'a_b':
        assert a is not None
        assert b is not None
        assert a - b == 1
        Z_I_J = weights.dimshuffle('x',0,1) * in_activations.dimshuffle(
            0,1,'x')
        Z_I_J_plus = Z_I_J * T.gt(Z_I_J,0)
        Z_I_J_neg = Z_I_J * T.lt(Z_I_J,0)
        if bias is not None:
            bias = bias.dimshuffle('x','x', 0)
            # redistribute bias proportionally across all inputs
            Z_I_J_plus += (bias * T.gt(bias,0)) / weights.shape[0]
            Z_I_J_neg += (bias * T.lt(bias,0)) / weights.shape[0]
        plus_sum = Z_I_J_plus.sum(axis=1, keepdims=True)
        neg_sum = Z_I_J_neg.sum(axis=1, keepdims=True)
        # Set 0s to 1 to avoid division by 0
        plus_sum = plus_sum + T.eq(0,plus_sum)
        neg_sum = neg_sum + T.eq(0,neg_sum)
        
        Z_I_J_plus_normed = Z_I_J_plus / plus_sum
        # Now already positive again (negative divided negative)
        Z_I_J_neg_normed = Z_I_J_neg / neg_sum
        Z_I_J_normed = (a * Z_I_J_plus_normed - b * Z_I_J_neg_normed)
        in_relevances = out_relevances.dimshuffle(0,'x',1) * Z_I_J_normed
        in_relevances = T.sum(in_relevances, axis=2)
        return in_relevances
    elif rule == 'a_b_sign_switch':
        assert a is not None
        assert b is not None
        assert a - b == 1
        Z_I_J = weights.dimshuffle('x',0,1) * in_activations.dimshuffle(
            0,1,'x')
        outputs = T.sum(Z_I_J, axis=1, keepdims=True)
        if bias is not None:
            # only add bias to outputs...
            bias = bias.dimshuffle('x', 0)
            outputs += bias
        # stabilize, prevent comparison to exactly 0
        eps = 1e-3
        outputs += (T.sgn(outputs) * eps)
        outputs += (T.eq(outputs, 0) * eps)
        Z_I_J_with_trend = T.abs_(Z_I_J) * (T.eq(T.sgn(outputs), T.sgn(Z_I_J)))
        norm_with_trend = T.sum(T.abs_(Z_I_J_with_trend), axis=1, 
            keepdims=True)
        # stabilize
        norm_with_trend += T.eq(norm_with_trend, 0)
        Z_I_J_with_trend_normed = Z_I_J_with_trend / norm_with_trend
        in_with_trend = (out_relevances.dimshuffle(0,'x', 1) * 
            Z_I_J_with_trend_normed)
        
        
        Z_I_J_against_trend = T.abs_(Z_I_J) * (T.neq(T.sgn(outputs), T.sgn(Z_I_J)))
        norm_against_trend = T.sum(T.abs_(Z_I_J_against_trend), axis=1, 
            keepdims=True)
        # stabilize
        norm_against_trend += T.eq(norm_against_trend, 0)
        Z_I_J_against_trend_normed = Z_I_J_against_trend / norm_against_trend
        in_against_trend = (out_relevances.dimshuffle(0,'x', 1) * 
            Z_I_J_against_trend_normed)
        
        in_relevances = a * in_with_trend - b * in_against_trend
        in_relevances = T.sum(in_relevances, axis=2)
        return in_relevances
    elif rule == 'a_b_abs':
        assert a is not None
        assert b is not None
        assert a - b == 1
        weights_plus = weights * T.gt(weights, 0)
        abs_inputs = T.abs_(in_activations).dimshuffle(0,1,'x')
        Z_I_J_plus = weights_plus.dimshuffle('x', 0,1) * abs_inputs
        plus_norm = T.sum(Z_I_J_plus, axis=1, keepdims=True)
        # stabilize
        plus_norm += T.eq(plus_norm, 0)
        Z_I_J_plus_normed = Z_I_J_plus / plus_norm
        in_plus = out_relevances.dimshuffle(0,'x', 1) * Z_I_J_plus_normed
        
        
        weights_neg = weights * T.lt(weights, 0)
        Z_I_J_neg = weights_neg.dimshuffle('x', 0,1) * abs_inputs
        # minus to get positive norms
        neg_norm = -T.sum(Z_I_J_neg, axis=1, keepdims=True)
        # stabilize
        neg_norm += T.eq(neg_norm, 0)
        # minus to get positive output, will be subtracted below
        Z_I_J_neg_normed = -Z_I_J_neg / neg_norm
        in_neg = out_relevances.dimshuffle(0,'x', 1) * Z_I_J_neg_normed
        in_relevances = a * in_plus - b * in_neg
        in_relevances = T.sum(in_relevances, axis=2)
        return in_relevances
        

def relevance_pool(out_relevances, inputs, pool_size, pool_stride):
    # channels x channels x pool_0 x pool_1
    pool_ones_shape = [out_relevances.shape[1], out_relevances.shape[1],
        pool_size[0], pool_size[1]]
    # modification: make inputs positive
    #inputs = T.abs_(inputs)
    # other variant: make inputs positive by offset
    offset = T.minimum(0, T.min(inputs, axis=(1,2,3), keepdims=True))
    inputs = inputs - offset
    pool_ones = T.ones(pool_ones_shape, dtype=np.float32)
    # only within a channel spread values of that channel...
    # therefore set all values of indices like
    # filt_i, channel_j with j!=i to zero!
    pool_ones = pool_ones * T.eye(out_relevances.shape[1],
                              out_relevances.shape[1]).dimshuffle(
                                 0,1,'x','x')
    norms_for_relevances = conv2d(inputs,
        pool_ones, subsample=pool_stride, border_mode='valid')
    # prevent division by 0...
    # the relevance which had norm zero will not be redistributed anyways..
    # so it doesn't matter which normalization factor you choose here,
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
        upsampled_relevances[:, :, ::pool_stride[0], ::pool_stride[1]], 
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


def show_heatmap_accuracy(heatmap_fn, pred_fn, X_train_flat_both,
                      y_train_both, n_trials=100, 
                         n_pixels=500):
    all_preds = pred_fn(X_train_flat_both)
    preds_for_correct_class = np.array([all_preds[i, y_train_both[i]] for i in xrange(len(X_train_flat_both))])

    wanted_trials = np.flatnonzero(preds_for_correct_class > 0.95)[:n_trials]
    input_trials = X_train_flat_both[wanted_trials]
    y_wanted_trials = y_train_both[wanted_trials]
    all_heatmaps = heatmap_fn(input_trials)


    n_pixels = 700
    # Pixels sorted by strength.. this proved better than randomly selecting
    # from positive pixels
    i_largest_pixels = np.argsort(input_trials, axis=-1)[:,::-1]
    """# shuffle individually for each trial, now put positive pixels in first
    rng = RandomState(39847834)
    for pixels in i_largest_pixels:
        first_part = pixels[:150]
        rng.shuffle(first_part)
        pixels[:150] = first_part"""
    random_acc_positive_pixels = []
    fake_input = input_trials.copy()
    for i_pixel in xrange(n_pixels):
        for i_trial, trial in enumerate(fake_input):
            rand_i = i_largest_pixels[i_trial, i_pixel]
            trial[rand_i] = 1 - trial[rand_i]
        wanted_preds = pred_fn(fake_input)
        preds_for_correct_class = np.array([wanted_preds[i, y_wanted_trials[i]]
                                            for i in xrange(len(wanted_preds))])
        random_acc_positive_pixels.append(np.mean(preds_for_correct_class))

    
        
    fake_input = input_trials.copy()
    all_heatmaps_flat = all_heatmaps.reshape(all_heatmaps.shape[0],-1)
    heatmap_vals = np.argsort(all_heatmaps_flat, axis=1)[:,::-1]
    heatmap_accuracy = []
    for i_pixel in xrange(n_pixels):
        for i_trial, trial in enumerate(fake_input):
            most_relevant_pixel = heatmap_vals[i_trial, i_pixel]
            trial[most_relevant_pixel] = 1 - trial[most_relevant_pixel]
        wanted_preds = pred_fn(fake_input)
        preds_for_correct_class = np.array([wanted_preds[i, y_wanted_trials[i]] 
            for i in xrange(len(wanted_preds))])
        heatmap_accuracy.append(np.mean(preds_for_correct_class))

        
    plt.figure()
    plt.plot(random_acc_positive_pixels)
    plt.plot(heatmap_accuracy)
    print("Mean difference first 100 positive pixels {:.2f}".format(
        np.mean(np.array(random_acc_positive_pixels[:100]) -
            np.array(heatmap_accuracy[:100]))))
    print("Mean difference first 250 positive pixels {:.2f}".format(
        np.mean(np.array(random_acc_positive_pixels[:250]) -
            np.array(heatmap_accuracy[:250]))))
