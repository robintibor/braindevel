from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
from braindecode.datasets.batch_iteration import SampleWindowsIterator,\
    BalancedBatchIterator, FlatSampleWindowsIterator
from braindecode.veganlasagne.monitors import SampleWindowMisclassMonitor,\
    FlatSampleWindowMisclassMonitor
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
    
def test_flat_sample_window_misclass_monitor():
    pred_func = lambda x: np.array((-(np.mean(x, axis=(1,2,3)) - 3), 
                                np.mean(x, axis=(1,2,3)) - 3, 
                                [0.0] * len(x))).T
    # should lead to predictions 0,1,1 which should lead to misclass 1/3.0

    topo_data = [range(i_trial,i_trial+6) for i_trial in range(3)]
    topo_data = np.array(topo_data)[:,np.newaxis,:,np.newaxis]
    
    y = np.int32(range(topo_data.shape[0]))
    dataset = DenseDesignMatrixWrapper(topo_view=topo_data, y=y, 
        axes=('b','c',0,1))
    
    iterator = FlatSampleWindowsIterator(batch_size=7, 
        trial_window_fraction=1/3.0, sample_axes_name=0, stride=1)
    
    monitor = FlatSampleWindowMisclassMonitor()
    monitor_chans = {'train_misclass': []}
    monitor.monitor_epoch(monitor_chans, pred_func, None, {'train':dataset}, iterator)
    assert np.allclose([1/3.0], monitor_chans['train_misclass'])