import numpy as np
import lasagne
from numpy.random import RandomState
import theano
import theano.tensor as T
from braindevel.veganlasagne.layers import get_input_shape

def create_descent_function(layer, wanted_activation,  learning_rate=0.1,
                            input_cost=None, n_trials=1, seed=983748374,
                            deterministic=True,
                            loss='sqr', init_factor=0.1):
    """
    Create descent function that updates random variable to match given wanted activation.
    
    Parameters
    ----------
    layer : 
        Layer to compute descent from.
    wanted_activation: list or nd array
        Activation to move towards.
    learning_rate : float
        Learning rate for adam updates
    input_cost : function or None
        Optional additional cost on the input.
    n_trials : int
        Number of inputs to randomly initialize and optimize.
    seed : int
        Random seed to initialize random variable.
    deterministic : boolean
        Whether to use deterministic mode when computing activations,
        i.e. no dropout etc.
    loss : function or 'sqr'
        Loss to use between wanted activation and actual activation.
    init_factor : float
        Factor for initialization of random variable.
        
    Returns
    -------
    rand_in_var: theano shared variable
        Random input variable to be optimized
    update_fn: theano compiled function
        Function to compute updates, returns current cost
        
    """
    rng = RandomState(seed)
    wanted_activation = np.array(wanted_activation)
    in_shape = get_input_shape(layer)
    
    in_shape = [n_trials] + list(in_shape[1:])
    
    rand_input = rng.randn(*in_shape).astype(np.float32) * init_factor
    rand_in_var = theano.shared(rand_input)
    # have to supply input_var extra in case of final reshape layer
    output = lasagne.layers.get_output(layer, deterministic=deterministic, 
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