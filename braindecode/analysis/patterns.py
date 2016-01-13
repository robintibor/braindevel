import numpy as np
import lasagne
import theano.tensor
import theano.tensor as T

def transform_conv_weights_to_patterns(conv_weights, topo, activations):
    """Assume weights are bc0, not having fourth dimension..in case of fourth dimension,
    please recheck code
    Assuming topo is bc01, assuming activations is bc01"""
    kernel_length = conv_weights.shape[2]
    # create topo with virtual channels
    topo_transformed = transform_train_topo(topo, kernel_length)
    topo_transformed = topo_transformed.transpose(1,0,2,3)
    topo_transformed = topo_transformed.reshape(topo_transformed.shape[0],-1)
    ## flatten and put channel/unit dimension first
    acts_out_for_cov = activations.transpose(1,0,2,3)
    acts_out_for_cov = acts_out_for_cov.reshape(acts_out_for_cov.shape[0], -1)
    conv_vectors = conv_weights.reshape(conv_weights.shape[0],-1)
    flat_patterns = transform_weights_to_patterns(conv_vectors, topo_transformed, acts_out_for_cov)
    unflat_patterns = flat_patterns.reshape(*conv_weights.shape)
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
    new_train_topo = transform_train_topo(train_topo, kernel_length)
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

def transform_train_topo(train_topo, kernel_length):
    """Expects bc01 format. """
    new_train_topo = np.empty((train_topo.shape[0], train_topo.shape[1] * kernel_length,
                         train_topo.shape[2] - kernel_length + 1, 1))
    n_chans = train_topo.shape[1]

    for i_sample in xrange(kernel_length):
        # do only take samples from 0 to ... n-30, 1..n-29 etc.
        # (assuming 30 is kernel_length)
        end = -kernel_length+1+i_sample
        if end == 0:
            end = None
        for i_chan in xrange(n_chans):
            new_train_topo[:, i_chan*kernel_length+i_sample, :] = \
                train_topo[:,i_chan,i_sample:end]
    return new_train_topo

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