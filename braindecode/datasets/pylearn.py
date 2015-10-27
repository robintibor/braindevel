from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class DenseDesignMatrixWrapper(DenseDesignMatrix):
    reloadable=False
    def ensure_is_loaded(self):
        pass
