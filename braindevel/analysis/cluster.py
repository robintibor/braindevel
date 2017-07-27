import theano.tensor as T
import numpy as np
from numpy.random import RandomState
import theano
import lasagne
from braindevel.veganlasagne.layers import get_input_shape
from copy import deepcopy
import sklearn

def compute_dist_to_k_th_neighbour_sorted(features, k_neighbours):
    pairwise_distances = sklearn.metrics.pairwise.pairwise_distances(features)
    sorted_dist = np.sort(pairwise_distances, axis=1)
    dist_to_k_th_neighbour = sorted_dist[:,k_neighbours+1] #1 to ignore distance to self in indexing
    dist_to_k_th_neighbour_sorted = np.sort(dist_to_k_th_neighbour)
    return dist_to_k_th_neighbour_sorted

def compute_to_move_to_cluster(cluster_activations, out_layer, 
                               n_cluster_samples, input_cost, n_trials, n_epochs,
                               seed=398498,
                              learning_rate=0.3,
                              print_only_final=False):
    rand_in_var, update_fn = optimize_to_move_to_cluster(out_layer,n_cluster_samples=n_cluster_samples,
                                                    input_cost=input_cost, seed=seed,
                                                    n_trials=n_trials,
                                                        learning_rate=learning_rate)
    orig_rand_val = deepcopy(rand_in_var.get_value().squeeze())
    for i_epoch in range(n_epochs):
        cost = update_fn(cluster_activations)
        if not print_only_final:
            if i_epoch % (n_epochs // 10) == 0:
                print cost
    print cost
    return rand_in_var, cost, update_fn, orig_rand_val
    

def mean_min_distance_to_cluster(actual_activation,
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
                               seed=983748374, input_cost=None,
                               n_trials=1):
    """Create function to optimize input to be close to cluster.
    Returns shared random variable which will be optimized and update function.
    Supply cluster activations to the update function"""
    rng = RandomState(seed)
    in_shape = get_input_shape(out_layer)
    in_shape = [n_trials] + list(in_shape[1:])
    rand_input = rng.randn(*in_shape).astype(np.float32)
    rand_in_var = theano.shared(rand_input)
    
    
    # have to supply input_var extra in case of final reshape layer
    output = lasagne.layers.get_output(out_layer, deterministic=True, 
        inputs=rand_in_var, input_var=rand_in_var)
    if output.ndim == 4:
        cluster_activations_var = T.ftensor4()
    elif output.ndim == 2:
        cluster_activations_var = T.fmatrix()

    # Calculate distances for all given "trials"
    distance = T.constant(0, dtype=np.float32)
    for i_trial in xrange(n_trials):
        distance += mean_min_distance_to_cluster(output[i_trial],
            cluster_activations_var, n_cluster_samples)
    distance = distance / n_trials # to get mean..not sure if smart
    
    if input_cost is None:
        cost = distance
    else:
        cost = distance + input_cost(rand_in_var)
        
    updates = lasagne.updates.adam(cost, [rand_in_var], learning_rate=learning_rate)
    update_fn = theano.function([cluster_activations_var], cost, updates=updates)
    return rand_in_var, update_fn
        