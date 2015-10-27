import numpy as np
from braindecode.datahandling.batch_iteration import WindowsIterator
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper

def test_flat_sample_windows_iterator():
    topo_data = [range(i_trial,i_trial+6) for i_trial in range(3)]
    topo_data = np.array(topo_data)[:,np.newaxis,:,np.newaxis]
    
    y = np.int32(range(topo_data.shape[0]))
    dataset = DenseDesignMatrixWrapper(topo_view=topo_data, y=y, axes=('b','c',0,1))
    
    iterator = WindowsIterator(batch_size=7, trial_window_fraction=1/3.0,
                                        sample_axes_name=0, stride=1)
    
    batches = list(iterator.get_batches(dataset, deterministic=True))
    assert(len(batches) == 2)
    assert np.array_equal((8,1,2,1), np.array(batches[0][0]).shape)
    assert np.array_equal([0,1], batches[0][0][0].squeeze())
    assert np.array_equal([1,2], batches[0][0][1].squeeze())
    assert np.array_equal([2,3], batches[0][0][2].squeeze())
    assert np.array_equal([3,4], batches[0][0][3].squeeze())
    assert np.array_equal([4,5], batches[0][0][4].squeeze())
    assert np.array_equal([1,2], batches[0][0][5].squeeze())
    assert np.array_equal([2,3], batches[0][0][6].squeeze())
    assert np.array_equal([3,4], batches[0][0][7].squeeze())
    
    assert np.array_equal([1,2], batches[0][0][1].squeeze())
    
    assert np.array_equal((7,1,2,1), np.array(batches[1][0]).shape)
    
    assert np.array_equal([4,5], batches[1][0][0].squeeze())
    assert np.array_equal([5,6], batches[1][0][1].squeeze())
    assert np.array_equal([2,3], batches[1][0][2].squeeze())
    assert np.array_equal([3,4], batches[1][0][3].squeeze())
    assert np.array_equal([4,5], batches[1][0][4].squeeze())
    assert np.array_equal([5,6], batches[1][0][5].squeeze())
    assert np.array_equal([6,7], batches[1][0][6].squeeze())
    

def test_flat_sample_windows_iterator_get_trial():
    topo_data = [range(i_trial,i_trial+6) for i_trial in range(3)]
    topo_data = np.array(topo_data)[:,np.newaxis,:,np.newaxis]
    
    y = np.int32(range(topo_data.shape[0]))
    dataset = DenseDesignMatrixWrapper(topo_view=topo_data, y=y, 
        axes=('b','c',0,1))
    
    iterator = WindowsIterator(batch_size=7, 
        trial_window_fraction=1/3.0, sample_axes_name=0, stride=1)
    trial_0_batches = list(iterator.get_batches_for_trial(dataset, 0))
    trial_1_batches = list(iterator.get_batches_for_trial(dataset, 1))
    trial_2_batches = list(iterator.get_batches_for_trial(dataset, 2))
    
    assert len(trial_0_batches) == 1
    assert len(trial_1_batches) == 1
    assert len(trial_2_batches) == 1
    assert np.array_equal([
             [[[ 0.], 
               [ 1.]]], 
             [[[ 1.],
               [ 2.]]],
             [[[ 2.],
               [ 3.]]],
             [[[ 3.],
               [ 4.]]],
             [[[ 4.],
               [ 5.]]]], trial_0_batches[0][0])
    
    assert np.array_equal([ 
             [[[ 1.],
               [ 2.]]],
             [[[ 2.],
               [ 3.]]],
             [[[ 3.],
               [ 4.]]],
             [[[ 4.],
               [ 5.]]],
             [[[ 5.],
               [ 6.]]]], trial_1_batches[0][0])
    
    assert np.array_equal([ 
             [[[ 2.],
               [ 3.]]],
             [[[ 3.],
               [ 4.]]],
             [[[ 4.],
               [ 5.]]],
             [[[ 5.],
               [ 6.]]],
             [[[ 6.],
               [ 7.]]]], trial_2_batches[0][0])
    assert np.array_equal(trial_0_batches[0][1], [0,0,0,0,0])
    assert np.array_equal(trial_1_batches[0][1], [1,1,1,1,1])
    assert np.array_equal(trial_2_batches[0][1], [2,2,2,2,2])