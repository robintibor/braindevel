import theano.tensor as T
from lasagne.nonlinearities import softmax, elu

def safe_log(x, eps=1e-6):
    """ Prevents log(0) by using max of eps and given x."""
    return  T.log(T.maximum(x, eps))

def safe_sqrt(x, eps=1e-4):
    """ Prevents that input of sqrt is too close to 0... just for check."""
    return  T.sqrt(T.maximum(x, eps))

def safe_softmax(x, eps=1e-6):
    """ Prevents that any of the outputs become exactly 1 or 0 """
    x = softmax(x)
    x = T.maximum(x, eps)
    x = T.minimum(x, 1 - eps)
    return x

def square(x):
    return T.sqr(x)

def elu_square(x):
    return T.sqr(elu(x))

def log_softmax(x):
    """Prevents instabilities when used with categorical_crossentropy_domain.
    From https://github.com/Lasagne/Lasagne/issues/332#issuecomment-122328992"""
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))
