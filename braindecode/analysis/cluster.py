import theano.tensor as T
import numpy as np
from numpy.random import RandomState
import theano
import lasagne
from braindecode.veganlasagne.layers import get_input_shape

def distance_to_cluster(actual_activation,
    cluster_activation,  n_cluster_samples):
    """Compute the difference of the actual activation to the
    closest n_cluster samples."""
    if cluster_activation.ndim == 4:
        actual_activation = actual_activation.dimshuffle('x',0,1,2)
        squared_distances = T.mean(T.square(cluster_activation -
            actual_activation), axis=(1,2,3))
    elif cluster_activation.ndim == 2:
        actual_activation = actual_activation.dimshuffle('x',0)
        squared_distances = T.mean(T.square(cluster_activation -
            actual_activation), axis=(1,))
        
    
    squared_distances_sorted = T.sort(squared_distances)
    
    distance = T.mean(squared_distances_sorted[:n_cluster_samples])
    return distance
        
def optimize_to_move_to_cluster(out_layer, n_cluster_samples, learning_rate=0.1,
                               seed=983748374, input_cost=None):
    """Create function to optimize input to be close to cluster.
    Returns shared random variable which will be optimized and update function.
    Supply cluster activations to the update function"""
    rng = RandomState(seed)
    in_shape = get_input_shape(out_layer)
    # push only one trial through
    in_shape = [1] + list(in_shape[1:])
    rand_input = rng.randn(*in_shape).astype(np.float32)
    rand_in_var = theano.shared(rand_input)
    
    
    # have to supply input_var extra in case of final reshape layer
    output = lasagne.layers.get_output(out_layer, deterministic=True, inputs=rand_in_var,
                                      input_var=rand_in_var)
    if output.ndim == 4:
        cluster_activations_var = T.ftensor4()
    elif output.ndim == 2:
        cluster_activations_var = T.fmatrix()

    distance = distance_to_cluster(output[0], #(should only have 1 output) 
        cluster_activations_var,  n_cluster_samples)
    
    if input_cost is None:
        cost = distance
    else:
        cost = distance + input_cost(rand_in_var)
        
    updates = lasagne.updates.adam(cost, [rand_in_var], learning_rate=learning_rate)
    update_fn = theano.function([cluster_activations_var], cost, updates=updates)
    return rand_in_var, update_fn