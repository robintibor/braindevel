import theano
import numpy as np
def to_4d_time_array(array):
    array = np.array(array)
    if array.ndim == 1:
        return array[np.newaxis,np.newaxis,:,np.newaxis].astype(theano.config.floatX)
    else:
        assert array.ndim == 2
        return array[:,np.newaxis,:,np.newaxis].astype(theano.config.floatX)

def equal_without_nans(a,b):
    return np.all(np.logical_or(a == b, np.logical_and(np.isnan(a), np.isnan(b))))
def allclose_without_nans(a,b):
    return np.all(np.logical_or(np.isclose(a,b), np.logical_and(np.isnan(a), np.isnan(b))))
