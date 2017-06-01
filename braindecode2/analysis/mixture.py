import theano.tensor as T
import numpy as np
import theano
from theano.tensor.nnet import neighbours
from lasagne.layers import Layer

# see https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py#L375
def logsumexp(arr, axis=0): 
    
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=axis, keepdims=True)
    out = T.log(T.sum(np.exp(arr - vmax), axis=axis))
    out += vmax
    return out

def compute_log_probs_gaussian_mixture(trials, means_sym, covars_sym, weights_sym):
    diffs = trials.dimshuffle(0,'x',1) - means_sym.dimshuffle('x',0,1)
    # now location x gaussian x features
    scaled_diffs = T.sqr(diffs / covars_sym.dimshuffle('x',0,1))
    # still location x gaussian x features
    exponent = -T.sum(scaled_diffs, axis=2) / 2.0
    # now location x gaussian
    # ignoring constant sqrt(2pi ** #num_features) for more stability(?)
    log_denominator =  T.log(T.sum(covars_sym, axis=1)) / 2.0
    log_prob_per_location_per_gaussian = (exponent - 
        log_denominator.dimshuffle('x', 0))
    log_weighted = log_prob_per_location_per_gaussian + (
        weights_sym.dimshuffle('x', 0))
    # still location x gaussian
    # sums over gaussian so have to logsumexp...
    log_prob_per_location = logsumexp(log_weighted, axis=1)
    return log_prob_per_location

def compute_probs_gaussian_mixture(trials, means_sym, covars_sym, weights_sym):
    """Warning: usually failed ... possibly numerical problems?"""
    diffs = trials.dimshuffle(0,'x',1) - means_sym.dimshuffle('x',0,1)
    # now location x gaussian x features
    scaled_diffs = T.sqr(diffs / covars_sym.dimshuffle('x',0,1))
    # still location x gaussian x features
    exponent = -T.sum(scaled_diffs, axis=2) / 2.0
    nominator = T.exp(exponent)
    # now location x gaussian
    # ignoring constant sqrt(2pi ** #num_features) for more stability(?)
    denominator =  T.prod(T.sqrt(covars_sym), axis=1)
    prob_per_location_per_gaussian = nominator / (
        denominator.dimshuffle('x', 0))
    prob_weighted = prob_per_location_per_gaussian * (
        weights_sym.dimshuffle('x', 0))
    # still location x gaussian
    prob_per_location = T.sum(prob_weighted, axis=1)
    return prob_per_location

def img_2_neibs_with_chans(inputs_sym, patch_size):
    flat_patches = neighbours.images2neibs(inputs_sym, patch_size, (1,1))
    topo_flat_patches = T.reshape(flat_patches,(inputs_sym.shape[0],
                                            inputs_sym.shape[1],
                                            inputs_sym.shape[2]-patch_size[0]+1,
                                            inputs_sym.shape[3]-patch_size[1]+1,
                                            patch_size[0],
                                            patch_size[1]))


    flat_patches = topo_flat_patches.dimshuffle(0,2,3,1,4,5)
    flat_patches = T.reshape(flat_patches, (T.prod(flat_patches.shape[:3]),
                                                 T.prod(flat_patches.shape[3:])))
    return flat_patches

def create_neibs_fn(patch_size):
    inputs_sym = T.ftensor4()
    flat_patches = img_2_neibs_with_chans(inputs_sym, patch_size)
    return theano.function([inputs_sym], flat_patches)

def get_patch_size(layer):
    if hasattr(layer, 'filter_size'):
        patch_size = layer.filter_size
    else:
        patch_size = layer.pool_size
    return patch_size

class GaussianMixtureSimilarityLayer(Layer):
    def __init__(self, incoming, means, covariances, weights, patch_size, 
                 pool_func=T.sum, **kwargs):
        self.means = T.constant(means, dtype=theano.config.floatX)
        self.covariances = T.constant(covariances, dtype=theano.config.floatX)
        self.weights = T.constant(weights, dtype=theano.config.floatX)
        self.patch_size = patch_size
        self.pool_func = pool_func
        super(GaussianMixtureSimilarityLayer,self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return [input_shape[0]]
    
    def get_output_for(self, input, **kwargs):
        flat_patches = img_2_neibs_with_chans(input, self.patch_size)
        log_prob_per_location = compute_log_probs_gaussian_mixture(flat_patches, 
            self.means, self.covariances, self.weights)
        log_prob_per_input = log_prob_per_location.reshape(input.shape[0],-1)
        return self.pool_func(log_prob_per_input, axis=1)     