from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
from braindecode.datahandling.batch_iteration import WindowsIterator
from braindecode.veganlasagne.monitors import WindowMisclassMonitor
import numpy as np

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
    
    iterator = WindowsIterator(batch_size=7, 
        trial_window_fraction=1/3.0, sample_axes_name=0, stride=1)
    
    monitor = WindowMisclassMonitor()
    monitor_chans = {'train_misclass': []}
    monitor.monitor_epoch(monitor_chans, pred_func, None, {'train': dataset}, 
        iterator)
    assert np.allclose([1/3.0], monitor_chans['train_misclass'])