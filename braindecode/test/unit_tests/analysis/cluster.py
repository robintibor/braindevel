import theano.tensor as T
import theano
import numpy as np
from braindecode.analysis.cluster import distance_to_cluster

def test_distance_to_cluster():
    wanted_activations_var = T.ftensor4()
    
    actual_activations_var = T.ftensor3()
    
    cost_var = distance_to_cluster(actual_activations_var, 
        wanted_activations_var,  n_cluster_samples=2)
    
    cost_fn = theano.function([actual_activations_var,
        wanted_activations_var], 
        cost_var)
    
    wanted_activations = np.array([[[[0,1,1]]],
                                  [[[0,1,0,]]],
                                  [[[0,1,0,]]]]).astype(np.float32)
    assert cost_fn([[[0,1,0]]], wanted_activations) == 0
    assert np.allclose(cost_fn([[[0,1,1]]], wanted_activations), 1/6.0)
    assert cost_fn([[[1,1,1]]], wanted_activations) == 0.5                              