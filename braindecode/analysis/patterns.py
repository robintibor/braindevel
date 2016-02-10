import numpy as np
import lasagne
import theano.tensor
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from braindecode.veganlasagne.layers import get_input_shape,\
    create_suitable_theano_input_var, create_suitable_theano_output_var,\
    BiasLayer
import logging
log = logging.getLogger(__name__)
from theano.compile import UnusedInputError

def create_pattern_deconv_fn(layers, patterns, patterns_flipped,
                             return_all=False, 
                             enforce_positivity_after_relu=True,
                             enforce_positivity_everywhere=False):
    assert patterns_flipped == False, "Otherwise implement it"

    inputs = create_suitable_theano_input_var(layers[-1])
    outputs = create_suitable_theano_output_var(layers[-1])
    
    n_patterns = len([l for l in layers if hasattr(l, 'W')])
    deconved_var = pattern_deconv(outputs, inputs, layers, patterns[:n_patterns],
                                  patterns_flipped=patterns_flipped,
                                  return_all=return_all,
                                  enforce_positivity_after_relu=enforce_positivity_after_relu,
                                 enforce_positivity_everywhere=enforce_positivity_everywhere)
    try:
        pattern_deconv_fn = theano.function([outputs, inputs], deconved_var)
    except UnusedInputError:
        pattern_deconv_fn = theano.function([outputs], deconved_var)
    return pattern_deconv_fn

def create_pattern_reconstruct_fn(layers, patterns, patterns_flipped,
                             return_all=False, 
                             enforce_positivity_after_relu=True,
                             enforce_positivity_everywhere=False):
    """ Attempts to reconstruct given input from specific layer
    by first computing feature representation..."""
    assert patterns_flipped == False, "Otherwise implement it"

    inputs = create_suitable_theano_input_var(layers[-1])
    outputs = lasagne.layers.get_output(layers[-1], 
        deterministic=True, input_var=inputs, inputs=inputs)
    
    n_patterns = len([l for l in layers if hasattr(l, 'W')])
    deconved_var = pattern_deconv(outputs, inputs, layers, patterns[:n_patterns],
                                  patterns_flipped=patterns_flipped,
                                  return_all=return_all,
                                  enforce_positivity_after_relu=enforce_positivity_after_relu,
                                 enforce_positivity_everywhere=enforce_positivity_everywhere)
    pattern_deconv_fn = theano.function([inputs], deconved_var)
    return pattern_deconv_fn
        
        
def compute_patterns_for_model(model, inputs):
    layers = lasagne.layers.get_all_layers(model)
    layers_with_weights = [l for  l in layers if hasattr(l, 'W')]
    return compute_patterns_for_layers(layers_with_weights, inputs)

def compute_patterns_for_layers(needed_layers, input_data):
    in_layers = [l.input_layer for l in needed_layers]
    in_and_out = zip(in_layers, needed_layers)
    in_and_out = np.array(in_and_out).flatten()
    if len(get_input_shape(needed_layers[0])) == 2:
        inputs = T.fmatrix()
    else:
        inputs = T.ftensor4()
    output = lasagne.layers.get_output(in_and_out, deterministic=True,
        inputs=inputs)
    log.info("Compiling forward pass...")
    out_fn = theano.function([inputs], output)
    outs_by_layer = out_fn(input_data)
    in_outs_by_layer = [outs_by_layer[2*i:2*i + 2]
        for i in xrange(len(outs_by_layer) /2)]
    patterns_per_layer = []
    for i_layer in xrange(len(needed_layers)):
        layer = needed_layers[i_layer]
        inputs = in_outs_by_layer[i_layer][0]
        outputs = in_outs_by_layer[i_layer][1]
        if hasattr(layer, 'filter_size'):
            log.info("Transforming to patterns for layer {:d}: {:s}...".format(
                i_layer, layer.__class__.__name__))
            conv_weights = layer.W.get_value()
            # TODO: take different implementation
            # if inputs bytes times conv_weights.shape[2:]
            # are below 2GB...
            pattern = transform_to_patterns(conv_weights, inputs, outputs,
                flip_filters=True) # conv weighs were flipped...
            log.info("Done.")
        elif isinstance(layer, lasagne.layers.DenseLayer):
            log.info("Transforming to patterns for layer {:d}: {:s}...".format(
                i_layer, layer.__class__.__name__))
            if inputs.ndim > 2:
                inputs = inputs.reshape(inputs.shape[0], -1)
            in_cov = np.cov(inputs.T)
            pattern = np.dot(in_cov, layer.W.get_value())
        else:
            raise ValueError("Trying to compute patterns for unknown layer "
                "type {:s}", layer.__class__.__name__)
        patterns_per_layer.append(pattern)
    return patterns_per_layer

def pattern_deconv(final_out, input_var, layers, pattern_weights,
        patterns_flipped, return_all=False, enforce_positivity_after_relu=True, 
        enforce_positivity_everywhere=False):
    """pattern_weights should be numpy variables and be unflipped,
    i.e. correlation patterns not convolution patterns"""
    assert patterns_flipped == False, "Otherwise implement it"
    layers_with_weights = [l for l in layers if hasattr(l, 'W')]
    assert len(layers_with_weights) == len(pattern_weights)
    layer_to_pattern = dict(zip(layers_with_weights, pattern_weights))
    all_outs = lasagne.layers.get_output(layers, inputs=input_var,
        deterministic=True)
    cur_out = final_out
    # determine input constraints by checking if there is some rectify
    input_constraints = []
    relu_before = False
    for l in layers:
        if relu_before and enforce_positivity_after_relu:
            input_constraints.append('positive')
            relu_before = False
        else:
            input_constraints.append(None)
        if hasattr(l, 'nonlinearity') and l.nonlinearity.__name__ == 'rectify':
            relu_before=True
        
    all_cur_outs = []
    # Stop before 0, no need to propagate through input layer...
    for i_layer in xrange(len(layers)-1,0,-1):
        layer = layers[i_layer]
        inputs = all_outs[i_layer-1]
        if hasattr(layer, 'pool_size'):
            cur_out = upsample_pool(cur_out, inputs, 
                layer.pool_size, layer.stride)
        elif hasattr(layer, 'filter_size'):
            #cur_out += layer.b.dimshuffle('x',0,'x','x')
            pattern = T.constant(layer_to_pattern[layer], dtype=np.float32)
            # filter flip true since we a) assume that patterns
            # are unflipped, so we have to flip them back(!)
            # this is all very confusing but should be correct now :)
            cur_out = conv2d(cur_out, pattern.dimshuffle(1,0,2,3),
                              filter_flip=True, border_mode='full')
        elif isinstance(layer, lasagne.layers.DenseLayer):
            #cur_out -= layer.b.dimshuffle('x', 0)
            pattern = T.constant(layer_to_pattern[layer], dtype=np.float32)
            cur_out = T.dot(cur_out, pattern.T)
        elif isinstance(layer, lasagne.layers.DimshuffleLayer):
            shuffle_pattern = layer.pattern
            reverse_pattern = np.array([shuffle_pattern.index(i) 
                for i in xrange(len(shuffle_pattern))])
            reverse_pattern = (reverse_pattern[1:]-1).tolist()
            # starting at 1 since no trial axis there..
            cur_out = cur_out.dimshuffle(reverse_pattern)
        elif layer.__class__.__name__ == 'BiasLayer':
            ndim = cur_out.ndim
            cur_out = cur_out - layer.b.dimshuffle(('x', 0) 
                + ('x',) * (ndim - 2))
        elif (isinstance(layer, lasagne.layers.DropoutLayer) or
            isinstance(layer, lasagne.layers.FlattenLayer) or
            isinstance(layer, lasagne.layers.NonlinearityLayer)):
                pass
        else:
            raise ValueError("Trying to propagate through unknown layer "
                "{:s}".format(layer.__class__.__name__))
        if hasattr(layer, 'nonlinearity') and layer.nonlinearity.__name__ == 'elu':
            # first make sure within admissible range (-1, inf)
            cur_out = T.maximum(cur_out, -0.9999)
            cur_out = T.gt(cur_out, 0) * cur_out + T.le(cur_out, 0) * T.log(cur_out + 1)
        if cur_out.shape != inputs.shape:
            cur_out = cur_out.reshape(inputs.shape)
        if input_constraints[i_layer] == 'positive' or enforce_positivity_everywhere:
            cur_out = cur_out * T.gt(cur_out,0)
        all_cur_outs.append(cur_out)
    if return_all:
        return all_cur_outs
    else:
        return cur_out
    
def upsample_pool(outputs, inputs, pool_size, pool_stride):
    pool_ones_shape = [outputs.shape[1], outputs.shape[1],
                       pool_size[0], pool_size[1]]

    pool_ones = T.ones(pool_ones_shape, dtype=np.float32)
    # only within a channel spread values of that channel...
    # therefore set all values of indices like
    # filt_i, channel_j with j!=i to zero! 
    pool_ones = pool_ones * T.eye(outputs.shape[1],
                              outputs.shape[1]).dimshuffle(0,1,'x','x')
    upsampled_out = T.zeros((outputs.shape[0], outputs.shape[1], 
            outputs.shape[2] * pool_stride[0] - pool_stride[0] + 1, 
            outputs.shape[3] * pool_stride[1] - pool_stride[1] + 1, 
            ), dtype=np.float32)

    upsampled_out = T.set_subtensor(
        upsampled_out[:, :, ::pool_stride[0], ::pool_stride[1]], 
        outputs)

    upsampled_out = conv2d(upsampled_out, 
                           pool_ones, subsample=(1,1),
                           border_mode='full')

    # quick check just distribute equally instead of respecting maxima?
    #actual_in = upsampled_out / np.prod(pool_size).astype(np.float32)
    pooled = downsample.max_pool_2d(inputs,ds=pool_size, st=pool_stride,
        ignore_border=False)

    grad = T.grad(None, inputs, known_grads={pooled: pooled})

    actual_in = upsampled_out * T.neq(0,grad)
    return actual_in

def compute_topo_covariances(all_inputs, weight_shape):
    """ all inputs and weight shape should be bc01"""
    n_chans = weight_shape[1]
    n_x = weight_shape[2]
    n_y = weight_shape[3]
    all_covariances = np.ones((n_chans,n_x,n_y,n_chans,n_x,n_y)) * np.nan
    for c1 in xrange(n_chans):
        for x1 in xrange(n_x):
            for y1 in xrange(n_y):
                for c2 in xrange(n_chans):
                    for x2 in xrange(n_x):
                        for y2 in xrange(y1,n_y):
                            if not np.isnan(all_covariances[c1,x1,y1,c2,x2,y2]):
                                continue # already computed
                                
                            # hmhm lets make it equal for comparison
                            end_x1 = -weight_shape[2] + x1 +1
                            if end_x1 == 0:
                                end_x1 = None
                            end_y = -weight_shape[3] + y1 + 1
                            end_x2 = -weight_shape[2] + x2 +1
                            if end_x2 == 0:
                                end_x2 = None
                            end_y = -weight_shape[3] + y1 + 1
                            if end_y == 0:
                                end_y = None
                            end_y2 = -weight_shape[3] + y2 + 1
                            if end_y2 == 0:
                                end_y2 = None
                            
                            in_1 = all_inputs[:,c1,x1:end_x1,y1:end_y].flatten()
                            in_2 = all_inputs[:,c2,x2:end_x2,y2:end_y2].flatten()

                            in_1_demeaned = in_1 - np.mean(in_1)
                            in_2_demeaned = in_2 - np.mean(in_2)
                            cov = np.dot(in_1_demeaned, in_2_demeaned) / float(len(in_1_demeaned) - 1)
                            
                            assert cov.shape == ()
                            assert not np.isnan(cov)
                            all_covariances[c1,x1,y1,c2,x2,y2] = cov
                            all_covariances[c2,x2,y2,c1,x1,y1] = cov
                                
    assert not np.any(np.isnan(all_covariances))                        
    return all_covariances

def transform_to_patterns(conv_weights, all_ins, all_outs,
        flip_filters):
    """
    Set flip filters to true if you are giving conv weights,
    to false if you are giving corr weights.
    """
    all_covariances = compute_topo_covariances(all_ins, conv_weights.shape)
    all_covariances_flat = all_covariances.reshape(
        np.prod(all_covariances.shape[:3]), np.prod(all_covariances.shape[:3]))
    out_covs = np.cov(all_outs.swapaxes(0,1).reshape(all_outs.shape[1], -1))
    if flip_filters:
        corr_weights = conv_weights[:,:,::-1,::-1]
    else:
        corr_weights = conv_weights
    flat_W = corr_weights.reshape(conv_weights.shape[0],-1)

    patterns = np.dot(all_covariances_flat, flat_W.T)
    """
    # Make hacky fix to out covs to prevent numerical instabilities
    # due to dead units
    diag_of_cov = np.diag(out_covs)
    # 100 below the max is a guess basically...
    bad_units = diag_of_cov < (np.max(diag_of_cov) / 100)
    #bad_units = np.array([False] * out_covs.shape[0])
    good_units = np.logical_not(bad_units)
    good_covs = out_covs[good_units][:, good_units]
    inv_good_covs = np.linalg.pinv(good_covs)
    inv_all_covs = np.ones((out_covs.shape), dtype=np.float32) * np.nan
    for i_filt in xrange(out_covs.shape[0]):
        inv_all_covs[i_filt,bad_units] = 0
        inv_all_covs[bad_units,i_filt] = 0
    good_inds = np.flatnonzero(good_units)
    for i_in_good_invs, i_good in enumerate(good_inds):
        inv_all_covs[i_good,good_units] = inv_good_covs[i_in_good_invs]
        inv_all_covs[good_units, i_good] = inv_good_covs[i_in_good_invs]
    assert not np.any(np.isnan(inv_all_covs))
    """
    inv_all_covs = np.linalg.pinv(out_covs)
    patterns = np.dot(patterns, inv_all_covs).T
    topo_patterns = patterns.reshape(conv_weights.shape)
    return topo_patterns

def transform_conv_weights_to_patterns(conv_weights, topo, activations,
    flip_filters):
    """Assume weights are bc0, not having fourth dimension..in case of fourth dimension,
    please recheck code
    Assuming topo is bc01, assuming activations is bc01"""
    kernel_length = conv_weights.shape[2]
    # create topo with virtual channels
    topo_transformed = add_time_delayed_channels(topo, kernel_length)
    topo_transformed = topo_transformed.transpose(1,0,2,3)
    topo_transformed = topo_transformed.reshape(topo_transformed.shape[0],-1)
    ## flatten and put channel/unit dimension first
    acts_out_for_cov = activations.transpose(1,0,2,3)
    acts_out_for_cov = acts_out_for_cov.reshape(acts_out_for_cov.shape[0], -1)
    
    if flip_filters:
        corr_weights = conv_weights[:,:,::-1]
    else:
        corr_weights = conv_weights
    conv_vectors = corr_weights.reshape(corr_weights.shape[0],-1)
    flat_patterns = transform_weights_to_patterns(conv_vectors,
        topo_transformed, acts_out_for_cov)
    unflat_patterns = flat_patterns.reshape(*corr_weights.shape)
    return unflat_patterns

def transform_raw_net_to_spat_temp_patterns(final_layer, train_topo):
    """ Assumes that net consists of "separated" conv in first two layers"""
    
   
    # compute activations
    layers = lasagne.layers.get_all_layers(final_layer)
    is_conv_layer = [hasattr(l, 'filter_size') for l in layers]
    i_second_conv_layer = np.flatnonzero(np.array(is_conv_layer))[1]
    second_conv = layers[i_second_conv_layer]
    assert not second_conv.nonlinearity == theano.tensor.sqr, ("squaring "
       "should be in own nonlinearitylayer for this to work")
    inputs = T.ftensor4()
    activation_vars = lasagne.layers.get_output(layer_or_layers=[second_conv],
                                                inputs=inputs, deterministic=True)
    activation_func = theano.function([inputs], activation_vars)
    
    all_trial_acts = [activation_func(train_topo[i:i+1]) for i in xrange(len(train_topo))]
    activations = np.array(all_trial_acts)[:,0].swapaxes(0,1) # swap layer axes in front of trial axes
    
    ## flatten and put channel/unit dimension first
    acts_out_for_cov = activations[0].transpose(1,0,2,3)
    acts_out_for_cov = acts_out_for_cov.reshape(acts_out_for_cov.shape[0], -1)
    
    # create train topo with virtual channels
    #kernel_length = combined_weights.shape[-1]
    kernel_length = layers[1].filter_size[0]
    new_train_topo = add_time_delayed_channels(train_topo, kernel_length)
    ## flatten and put channel/unit dimension first
    new_train_topo_for_cov = new_train_topo.transpose(1,0,2,3)
    new_train_topo_for_cov = new_train_topo_for_cov.reshape(
        new_train_topo_for_cov.shape[0], -1)
    ## Get combined weights
    combined_weights = get_combined_weights_rawnet(final_layer)
    
    combined_vectors = combined_weights.reshape(combined_weights.shape[0],-1)
    flat_patterns = transform_weights_to_patterns(combined_vectors,
        new_train_topo_for_cov, acts_out_for_cov)
    unflat_patterns = flat_patterns.reshape(*combined_weights.shape)
    return unflat_patterns

def get_combined_weights_rawnet(final_layer):
    # Compute combined the weights. We have to reverse them with ::-1 to change convolution to cross-correlation

    params = lasagne.layers.get_all_params(final_layer, trainable=True)

    temporal_weights = params[0].get_value()[:,:,::-1,::-1]
    spat_filt_weights = params[2].get_value()[:,:,::-1,::-1]

    combined_weights = np.tensordot(spat_filt_weights, temporal_weights, axes=(1,0))

    combined_weights = combined_weights.squeeze()
    return combined_weights

def add_time_delayed_channels(topo, kernel_length):
    """Expects bc01 format. """
    new_topo = np.empty((topo.shape[0], topo.shape[1] * kernel_length,
                         topo.shape[2] - kernel_length + 1, 1))
    n_chans = topo.shape[1]

    for i_sample in xrange(kernel_length):
        # do only take samples from 0 to ... n-30, 1..n-29 etc.
        # (assuming 30 is kernel_length)
        end = -kernel_length+1+i_sample
        if end == 0:
            end = None
        for i_chan in xrange(n_chans):
            new_topo[:, i_chan*kernel_length+i_sample, :] = \
                topo[:,i_chan,i_sample:end]
    return new_topo

def transform_weights_to_patterns(weights, inputs, outputs):
    """ Inputs are expected in features x samples/trials structure.
    See http://www.sciencedirect.com/science/article/pii/S1053811913010914
    Theorem 1."""
    input_cov = np.cov(inputs)
    output_cov = np.cov(outputs)
    patterns = np.dot(input_cov, weights.T)
    patterns = np.dot(patterns, np.linalg.pinv(output_cov)).T
    return patterns

def compute_soft_weights_patterns(final_layer, train_topo):
    """Returns in bc0 format, i.e. class x input filter x time"""
    assert False, "Please recheck this code before use.. maybe remove parantheses around layer, this is highly confusing... "
    inputs = T.ftensor4()
    activation_vars = lasagne.layers.get_output(layer_or_layers=[final_layer.input_layer],
                                                inputs=inputs, deterministic=True)
    activation_func = theano.function([inputs], activation_vars)

    all_trial_acts = [activation_func(train_topo[i:i+1]) for i in xrange(len(train_topo))]
    # wrong comment?: swap layer axes in front of trial axes
    # probably swaps filter axis in front of trial axes
    activations = np.array(all_trial_acts)[:,0].swapaxes(0,1) 
    soft_weights = np.array(final_layer.W.get_value())
    assert False, "sure this is correct? "
    cov_act = np.cov(activations[0].reshape(len(activations[0]), -1).T)
    soft_pattern = np.dot(cov_act, soft_weights)
    # make them into nice shape
    return soft_weights.T.reshape(4,40,54), soft_pattern.T.reshape(4,40,54)