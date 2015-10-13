from lasagne.layers.pool import Pool2DLayer, Layer
import theano.tensor as T
from theano.tensor.signal import downsample
import numpy as np
from copy import copy
import collections

class SumPool2dLayer(Pool2DLayer):
    def get_output_for(self, input, **kwargs):
        pooled = downsample.max_pool_2d(input,
                                        ds=self.pool_size,
                                        st=self.stride,
                                        ignore_border=self.ignore_border,
                                        padding=self.pad,
                                        mode=self.mode,
                                        )
        # cast size to float32 to prevent upcast to float64 of entire data
        return pooled * np.float32(np.prod(self.pool_size))
    
    
    
class GlobalPoolLayerAxisWise(Layer):
    """
    lasagne.layers.GlobalPoolLayer(incoming,
    pool_function=theano.tensor.mean, **kwargs)
    Global pooling layer where you can also set the axes to pool over.
    This layer pools globally across all trailing dimensions beyond the 2nd.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.
    pool_function : callable
        the pooling function to use. This defaults to `theano.tensor.mean`
        (i.e. mean-pooling) and can be replaced by any other aggregation
        function.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, pool_function=T.mean, 
        axis=2, **kwargs):
        super(GlobalPoolLayerAxisWise, self).__init__(incoming, **kwargs)
        self.pool_function = pool_function
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        new_shape = copy(list(input_shape))
        
        if isinstance(self.axis, collections.Iterable):
            for ax in self.axis:
                new_shape[ax] = 1
        else: # should be just an int
            new_shape[self.axis] = 1
        return new_shape

    def get_output_for(self, input, **kwargs):
        return self.pool_function(input, axis=self.axis)