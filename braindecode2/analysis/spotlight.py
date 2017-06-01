from lasagne.objectives import categorical_crossentropy
from lasagne.updates import adam
import theano
import numpy as np
import theano.tensor as T
import lasagne

def create_spotlight_fn(final_layer, blur_axes, free_axes, weight_axes, trials_shape):
    ones_shape = [trials_shape[i_ax] if i_ax in blur_axes + free_axes else 1 
                  for i_ax in xrange(len(trials_shape))]

    means_stds_shape = [trials_shape[i_ax] if i_ax in free_axes else 1 
                  for i_ax in xrange(len(trials_shape))]
    means_stds_shape = [len(blur_axes)] + means_stds_shape
    #toadd: mixture of gaussians
    full_mask = T.ones(ones_shape, dtype=np.float32)
    broadcast_pattern = [True if ax not in (free_axes) else False 
                         for ax in xrange(len(trials_shape))]
    broadcast_pattern = [False] + broadcast_pattern

    means = theano.shared((np.ones(means_stds_shape)* 0.5).astype(np.float32),
                          broadcastable=broadcast_pattern)
    stds = theano.shared((np.ones(means_stds_shape)* 1).astype(np.float32),
                          broadcastable=broadcast_pattern)

    for i_blur_axis, axis in enumerate(blur_axes):
        ax_mask = T.constant(np.linspace(0,1, trials_shape[axis], dtype=np.float32))
        dimshuffle_pattern = [0 if ax == axis else 'x' for ax in xrange(len(trials_shape))]
        ax_mask = ax_mask.dimshuffle(*dimshuffle_pattern)
        # todo maybe have to fix this here?
        ax_gaussian = T.exp(-T.square((ax_mask - means[i_blur_axis]) / stds[i_blur_axis]) * 0.5)
        full_mask = full_mask * ax_gaussian
    
    weights_shape = [trials_shape[i_ax] if i_ax in weight_axes else 1 
                  for i_ax in xrange(1,len(trials_shape))]
    weights_shape = [trials_shape[0]] + weights_shape
    broadcast_pattern = [True if ax not in (weight_axes) else False 
                         for ax in xrange(1, len(trials_shape))]
    broadcast_pattern = [False] + broadcast_pattern
    weights = theano.shared((np.ones(weights_shape)).astype(np.float32),
                          broadcastable=broadcast_pattern)
    full_mask = full_mask * (T.maximum(weights,0) / 
        T.mean(T.maximum(weights,0), axis=0, keepdims=True))
    
    trials_var = T.ftensor4()
    scaled_trials = trials_var * full_mask
    targets = T.ivector()

    outputs = lasagne.layers.get_output(final_layer, inputs=scaled_trials, input_var=scaled_trials)

    loss = categorical_crossentropy(outputs, targets).sum()
    loss += T.mean(T.sqr(stds)) * 0.1
    loss -= T.mean(T.abs_(weights - T.mean(weights, axis=0, keepdims=True))) * 10
    adam_updates = adam(loss,[means, stds, weights], learning_rate=0.01)
    adam_grad_fn = theano.function([trials_var, targets], 
                                   [loss,outputs, scaled_trials, full_mask, weights], 
                                   updates=adam_updates)
    return adam_grad_fn