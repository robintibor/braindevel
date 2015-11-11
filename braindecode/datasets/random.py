from numpy.random import RandomState
import numpy as np
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper


class RandomSet(DenseDesignMatrixWrapper):
    """ Random set for debugs"""
    def __init__(self, topo_shape, y_shape):
        self.topo_shape = topo_shape
        self.y_shape = y_shape
        
    def load(self): 
        rng = RandomState(328764)
        topo = rng.rand(*self.topo_shape).astype(np.float32)
        y = rng.rand(*self.y_shape)        
        y = (y ==  np.max(y, axis=1, keepdims=True)).astype(np.int32)
        super(RandomSet, self).__init__(topo_view=topo, y=y, axes=('b','c',0,1))