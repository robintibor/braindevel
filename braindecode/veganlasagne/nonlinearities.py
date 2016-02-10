import theano.tensor as T
from lasagne.nonlinearities import softmax

def safe_log(x, eps=1e-6):
    """ Prevents log(0) by using max of eps and given x."""
    return  T.log(T.maximum(x, eps))

def safe_softmax(x, eps=1e-6):
    """ Prevents that any of the outputs become exactly 1 or 0 """
    x = softmax(x)
    x = T.maximum(x, eps)
    x = T.minimum(x, 1 - eps)
    return x

def square(x):
    return T.sqr(x)
