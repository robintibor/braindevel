import numpy as np
from braindecode.datasets.batch_iteration import (
    BalancedBatchIterator, SampleWindowsIterator, FlatSampleWindowsIterator)
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper

def test_sample_windows_iterator():
    iterator = SampleWindowsIterator(trial_window_fraction=0.5, 
                                     batch_iterator=BalancedBatchIterator(batch_size=2))
    
    # 10 trials, all with 0,1,2...,19 as data in the third dim (0), other  (third/fourth, 'c') dims empty
    fake_train_data = np.array([range(4) for _ in range(6)])[:,np.newaxis,:,np.newaxis]
    fake_y = np.arange(6)
    
    fake_train_set = DenseDesignMatrixWrapper(topo_view=fake_train_data, y=fake_y, axes=('b','c',0,1))
    
    
    batches = list(iterator.get_batches(fake_train_set, deterministic=False))
    
    
    
    for i_batch in range(3):
        batch = batches[i_batch]
        features = batch[0]
        targets = batch[1]
        assert features.shape == (6,1,2,1)
        assert np.array_equal(features[0].squeeze(), range(2))
        assert np.array_equal(features[1].squeeze(), range(1,3))
        assert np.array_equal(features[2].squeeze(), range(2,4))
        assert np.array_equal(features[3].squeeze(), range(2))
        assert np.array_equal(features[4].squeeze(), range(1,3))
        assert np.array_equal(features[5].squeeze(), range(2,4))
        
        # check targets of sample original trial are the same
        assert np.all(targets[0:3] == targets[0])
        assert np.all(targets[3:6] == targets[3])
        assert targets[0] != targets[3]

def test_flat_sample_windows_iterator():
    topo_data = [range(i_trial,i_trial+6) for i_trial in range(3)]
    topo_data = np.array(topo_data)[:,np.newaxis,:,np.newaxis]
    
    y = np.int32(range(topo_data.shape[0]))
    dataset = DenseDesignMatrixWrapper(topo_view=topo_data, y=y, axes=('b','c',0,1))
    
    iterator = FlatSampleWindowsIterator(batch_size=7, trial_window_fraction=1/3.0,
                                        sample_axes_name=0, stride=1)
    
    batches = list(iterator.get_batches(dataset, deterministic=True))
    #list(get_flat_batches(dataset, batch_size=7, trial_window_fraction=1/3.0,
                               #sample_axes_name=0, stride=1, rng=batch_rng, deterministic=True))
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