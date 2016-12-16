from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np
import logging
log = logging.getLogger(__name__)
class DenseDesignMatrixWrapper(DenseDesignMatrix):
    reloadable=False
    def ensure_is_loaded(self):
        pass

class SimpleSet(object):
    """Only for compatibility with old pylearn interface"""
    reloadable = False
    def __init__(self, topo_view, y, axes=None):
        self.topo_view = topo_view
        self.y = y
        if axes is not None:
            log.warn("axes {:s} are being ignored ".format(str(axes)))
        self.view_converter = lambda : None
        self.view_converter.axes = ['b', 'c', 0, 1]
    
    def ensure_is_loaded(self):
        pass
    
    def set_topological_view(self, topo_view, axes=None):
        self.topo_view = topo_view
        if (axes is not None) and (not np.array_equal(axes,
                ['b', 'c', 0, 1])):
            log.warn("axes {:s} are being ignored ".format(str(axes)))
        
    def get_topological_view(self):
        return self.topo_view