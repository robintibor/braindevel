import theano.tensor as T
import theano
import numpy as np
from braindevel.veganlasagne.objectives import tied_losses
import lasagne
from numpy.random import RandomState

def test_tied_losses():
    # Regression Test, expected values never checked for correctness ....
    n_classes = 3
    n_sample_preds = 2
    n_pairs =1
    lasagne.random.set_rng(RandomState(83347))
    
    
    preds_sym = T.fmatrix()
    
    loss = tied_losses(preds_sym, n_sample_preds, n_classes, n_pairs)
    
    
    loss_fn = theano.function([preds_sym], loss)

    # First example, both predictions identical, can only lead to one result
    assert np.allclose([[ 0.89794558]],
       loss_fn(np.array([[0.6,0.3,0.1],[0.6,0.3,0.1]], dtype=np.float32)))
    assert np.allclose([[ 0.89794558]],
       loss_fn(np.array([[0.6,0.3,0.1],[0.6,0.3,0.1]], dtype=np.float32)))

        
    # Second example, two different predictions, can lead to different results
    assert np.allclose([[ 1.46424174]],
                       loss_fn(np.array([[0.1,0.6,0.3],[0.6,0.3,0.1]], 
                           dtype=np.float32)))
    assert np.allclose([[ 1.65519595]],
                       loss_fn(np.array([[0.1,0.6,0.3],[0.6,0.3,0.1]], 
                           dtype=np.float32)))
    
    
def test_tied_losses_multiple_pairs():
    # Regression Test, expected values never checked for correctness ....
    n_classes = 2
    n_sample_preds = 4
    n_pairs =2
    lasagne.random.set_rng(RandomState(329489393))
    preds = np.array([[0.1,0.9],[0.2,0.8],[0.3,0.7],[0.2,0.8],
                     [0.4,0.6],[0.4,0.6],[0.4,0.6],[0.4,0.6],
                     [0.6,0.4],[0.4,0.6],[0.2,0.8],[0.5,0.5]], dtype=np.float32)
    
    preds_sym = T.fmatrix()
    loss = tied_losses(preds_sym, n_sample_preds, n_classes, n_pairs)
    loss_fn = theano.function([preds_sym], loss)
    assert np.allclose([[ 0.63903177,  0.36177287],
       [ 0.67301154,  0.67301154],
       [ 0.59191853,  0.69314712]],
                   loss_fn(preds))
    assert np.allclose([[ 0.54480541,  0.52613449],
       [ 0.67301154,  0.67301154],
       [ 0.71355808,  0.7776612 ]],
                   loss_fn(preds))
    assert np.allclose([[ 0.63903177,  0.54480541],
       [ 0.67301154,  0.67301154],
       [ 0.59191853,  0.71355808]],
    loss_fn(preds))

