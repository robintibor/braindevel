import theano.tensor as T
def safe_log(x, eps=1e-6):
    """ Prevents log(0) by using max of eps and given x."""
    return  T.log(T.maximum(x, eps))