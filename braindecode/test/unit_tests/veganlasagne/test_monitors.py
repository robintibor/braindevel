from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
from braindecode.datahandling.batch_iteration import WindowsIterator
from braindecode.veganlasagne.monitors import WindowMisclassMonitor,\
    MonitorManager, CntTrialMisclassMonitor
import numpy as np
import theano.tensor as T

def test_window_misclass_monitor():
    inputs = T.ftensor4()
    targets = T.ivector()
    
    preds = T.stack((-(T.mean(inputs, axis=(1,2,3)) - 3),
        T.mean(inputs, axis=(1,2,3)) - 3,
        0.0 * T.mean(inputs, axis=(1,2,3)))).T
    loss = T.mean(targets) # some dummy stuff
    # should lead to predictions 0,1,1 which should lead to misclass 1/3.0

    topo_data = [range(i_trial,i_trial+6) for i_trial in range(3)]
    topo_data = np.array(topo_data,dtype=np.float32)[:,np.newaxis,:,np.newaxis]
    
    y = np.int32(range(topo_data.shape[0]))
    dataset = DenseDesignMatrixWrapper(topo_view=topo_data, y=y, 
        axes=('b','c',0,1))
    
    iterator = WindowsIterator(batch_size=7, n_samples_per_window=2,
        sample_axes_name=0, n_sample_stride=1)
    
    monitor = WindowMisclassMonitor()
    monitor_manager = MonitorManager([monitor])
    monitor_manager.create_theano_functions(inputs, targets, preds, loss)
    monitor_chans = {'train_misclass': []}
    monitor_manager.monitor_epoch(monitor_chans, {'train': dataset}, iterator)
    assert np.allclose([1/3.0], monitor_chans['train_misclass'])


def test_cnt_trial_misclass_monitor():
    monitor_chans = dict(test_misclass=[])
    fake_set = lambda: None
    # actually exact targets dont matter..
    # just creating 3 trials here in the y signal...
    fake_set.y = np.array([[0,0,0,0],[0,0,0,1],[0,0,0,1],[0,0,0,0], 
                            [0,0,1,0],[0,0,1,0],[0,0,0,0],
                          [0,0,0,0],[1,0,0,0],[1,0,0,0],[0,0,0,0]])
    
    # first batch has two rows
    # second has one
    all_preds = np.array([
        np.array([[0,0.1,0.1,0.8], [0,0.1,0.1,0.8], [0,0.8,0.1,0.1],[0,0.8,0.1,0.1]]),
                 np.array([[0.8,0.1,0.1,0.1],[0.8,0.1,0.1,0.1]])])
    
    all_targets = np.array([[[0,0,0,1], [0,0,0,1], [0,0,1,0],[0,0,1,0]],
                 [[1,0,0,0],[1,0,0,0]]])
    
    all_losses=None # ignoring
    batch_sizes=[2,1]
    
    monitor = CntTrialMisclassMonitor(input_time_length=1)
    monitor.monitor_set(monitor_chans, 'test', all_preds, all_losses,
            batch_sizes, all_targets, fake_set)
    
    assert np.allclose(1/3.0, monitor_chans['test_misclass'][-1])
    
    # longer input time length and corresponding padding at start
    monitor_chans = dict(test_misclass=[])
    fake_set = lambda: None
    # actually exact targets dont matter..
    # just creating 3 trials here in the y signal...
    fake_set.y = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                           [0,0,0,1],[0,0,0,1],[0,0,0,0], 
                            [0,0,1,0],[0,0,1,0],[0,0,0,0],
                          [0,0,0,0],[1,0,0,0],[1,0,0,0],[0,0,0,0]])
    
    # first batch has two rows
    # second has one
    all_preds = np.array([
        np.array([[0,0.1,0.1,0.8], [0,0.1,0.1,0.8], [0,0.8,0.1,0.1],[0,0.8,0.1,0.1]]),
                 np.array([[0.8,0.1,0.1,0.1],[0.8,0.1,0.1,0.1]])])
    
    all_targets = np.array([[[0,0,0,1], [0,0,0,1], [0,0,1,0],[0,0,1,0]],
                 [[1,0,0,0],[1,0,0,0]]])
    
    all_losses=None # ignoring
    batch_sizes=[2,1]
    
    monitor = CntTrialMisclassMonitor(input_time_length=3)
    monitor.monitor_set(monitor_chans, 'test', all_preds, all_losses,
            batch_sizes, all_targets, fake_set)
    
    assert np.allclose(1/3.0, monitor_chans['test_misclass'][-1])
    
    # Ignore the predictions on empty targets
    # expect it creates 3 outputs per length-2trial
    
    fake_set = lambda: None
    
    # actually exact targets dont matter..
    # just creating 3 trials here in the y signal...
    fake_set.y = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                           [0,0,0,1],[0,0,0,1],[0,0,0,0], 
                            [0,0,1,0],[0,0,1,0],[0,0,0,0],
                          [0,0,0,0],[1,0,0,0],[1,0,0,0],[0,0,0,0]])
    
    all_preds = np.array([
        np.array([[-1,-1,-1,-1], [0,0.1,0.1,0.8], [0,0.1,0.1,0.8], 
                  [-1,-1,-1,-1], [0,0.8,0.1,0.1],[0,0.8,0.1,0.1]]),
                 np.array([[-1,-1,-1,-1],[0.8,0.1,0.1,0.1],[0.8,0.1,0.1,0.1]])])
    
    all_targets = np.array([[[0,0,0,0], [0,0,0,1], [0,0,0,1], 
                             [0,0,0,0], [0,0,1,0],[0,0,1,0]],
                            [[0,0,0,0], [1,0,0,0],[1,0,0,0]]])
    
    all_losses=None # ignoring
    batch_sizes=[2,1]
    
    monitor = CntTrialMisclassMonitor(input_time_length=3)
    monitor.monitor_set(monitor_chans, 'test', all_preds, all_losses,
            batch_sizes, all_targets, fake_set)
    
    assert np.allclose(1/3.0, monitor_chans['test_misclass'][-1])