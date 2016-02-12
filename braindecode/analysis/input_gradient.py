import numpy as np
import lasagne
from numpy.random import RandomState
import theano
import theano.tensor as T
from braindecode.veganlasagne.layers import get_input_shape

def create_descent_function(layer, wanted_activation,  learning_rate=0.1,
                            input_cost=None, n_trials=1, seed=983748374,
                            loss='sqr'):
    rng = RandomState(seed)
    wanted_activation = np.array(wanted_activation)
    in_shape = get_input_shape(layer)
    
    in_shape = [n_trials] + list(in_shape[1:])
    
    rand_input = rng.randn(*in_shape).astype(np.float32)
    rand_in_var = theano.shared(rand_input)
    # have to supply input_var extra in case of final reshape layer
    output = lasagne.layers.get_output(layer, deterministic=True, 
        inputs=rand_in_var, input_var=rand_in_var)
    
    if loss == 'sqr':
        output_cost = T.sqr(output - wanted_activation[np.newaxis])
    else:
        output_cost = loss(output, wanted_activation[np.newaxis])
    output_cost = T.mean(output_cost)
    
    if input_cost is None:
        cost = output_cost
    else:
        cost = output_cost + input_cost(rand_in_var)
    
    updates = lasagne.updates.adam(cost, [rand_in_var], learning_rate=learning_rate)
    update_fn = theano.function([], cost, updates=updates)
    return rand_in_var, update_fn