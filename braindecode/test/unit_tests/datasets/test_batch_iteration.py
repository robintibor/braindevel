import numpy as np
from braindecode.datasets.batch_iteration import (
    BalancedBatchIterator, SampleWindowsIterator)
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
