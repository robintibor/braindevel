from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
from braindecode.datasets.batch_iteration import SampleWindowsIterator,\
    BalancedBatchIterator
from braindecode.veganlasagne.monitors import SampleWindowMisclassMonitor
import numpy as np

def test_sample_window_misclass_monitor():
    """Test by using a prediction function which returns negative and positive
    sums over windows as predictions for two classes."""
    def pred_func(X):
        return np.array([-np.sum(X, axis=(1,2,3)), np.sum(X, axis=(1,2,3))]).T
    topo = np.array([[[[-1,0,-1,0]]],  [[[1,0,1,0]]], [[[4,0,-1,-1]]]])
    #sums should be -3, 3,1 for sample window frac 2
    y = np.array([0,1,0]) # last should be wrongly predicted as class 1
    
    dataset= DenseDesignMatrixWrapper(topo_view=topo,y=y, axes=('b', 'c', 0, 1))
    iterator = SampleWindowsIterator(trial_window_fraction=0.5, 
        batch_iterator=BalancedBatchIterator(batch_size=1),
        sample_axes_name=1)
    
    monitor = SampleWindowMisclassMonitor()
    monitor_chans = {'train_misclass':[]}
    monitor.monitor_epoch(monitor_chans, pred_func=pred_func, loss_func=None,
                         datasets={'train':dataset}, iterator=iterator)
    assert np.allclose(1/3.0, monitor_chans['train_misclass'])