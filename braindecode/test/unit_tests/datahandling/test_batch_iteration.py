import numpy as np
from braindecode.datahandling.batch_iteration import WindowsIterator
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
from braindecode.datahandling.batch_iteration import CntWindowsFromCntIterator
from braindecode.test.util import to_4d_time_array


def test_windows_iterator():
    topo_data = [range(i_trial,i_trial+6) for i_trial in range(3)]
    topo_data = np.array(topo_data)[:,np.newaxis,:,np.newaxis]
    
    y = np.int32(range(topo_data.shape[0]))
    dataset = DenseDesignMatrixWrapper(topo_view=topo_data, y=y, axes=('b','c',0,1))
    
    iterator = WindowsIterator(batch_size=7, n_samples_per_window=2,
                                        sample_axes_name=0, n_sample_stride=1)
    
    batches = list(iterator.get_batches(dataset, shuffle=False))
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
    
def test_cnt_windows_iterator():
    iterator = CntWindowsFromCntIterator(batch_size=2, input_time_length=6,
            n_sample_preds=4)
    
    in_topo = to_4d_time_array(range(20)).swapaxes(2,0)
    y = np.outer(range(20), np.ones(4))
    in_set = DenseDesignMatrixWrapper(topo_view=in_topo, y=y)
    
    batches = list(iterator.get_batches(in_set, shuffle=False))
    
    assert 2 == len(batches)
    # we have two lost samples so expect wraparound from back
    assert np.array_equal([[18,19] + range(4), range(2,8), range(6,12)], batches[0][0].squeeze())
    assert np.array_equal(np.outer([0,1,2,3,4,5,6,7,8,9,10,11], np.ones(4)), batches[0][1])
    assert np.array_equal([range(10,16), range(14,20)], batches[1][0].squeeze())
    assert np.array_equal(np.outer([12,13,14,15,16,17,18,19], np.ones(4)), batches[1][1])

def test_cnt_windows_iterator_shuffle():
    #Random Regression test, values should not change unless randomization changes...
    iterator = CntWindowsFromCntIterator(batch_size=2, input_time_length=6, n_sample_preds=4)
    iterator.reset_rng()
    
    in_topo = to_4d_time_array(range(20)).swapaxes(2,0)
    y = np.outer(range(20), np.ones(4))
    in_set = DenseDesignMatrixWrapper(topo_view=in_topo, y=y)
    
    batches = list(iterator.get_batches(in_set, shuffle=True))
    assert 2 == len(batches)
    assert np.array_equal([range(6,12), [18,19] + range(4), range(10,16)], batches[0][0].squeeze())
    assert np.array_equal(np.outer([8,9,10,11,0,1,2,3,12,13,14,15], np.ones(4)), batches[0][1])
    assert np.array_equal([range(14,20), range(2,8)], batches[1][0].squeeze())
    assert np.array_equal(np.outer([16,17,18,19,4,5,6,7], np.ones(4)), batches[1][1])

def test_cnt_windows_iterator_oversample():
    iterator = CntWindowsFromCntIterator(batch_size=3, input_time_length=6, n_sample_preds=4, oversample_targets=True)
    iterator.reset_rng()
    y = np.append(np.zeros((16,4)), np.ones((4,4)), axis=0)
    in_set = DenseDesignMatrixWrapper(topo_view=to_4d_time_array(range(20)).swapaxes(2,0), y=y)
    
    batches = list(iterator.get_batches(in_set, shuffle=True))
    assert 3 == len(batches)
    # Note that 14,20 is oversampled
    assert np.array_equal(batches[0][0].squeeze(), [range(6,12), range(10,16), range(14,20)])
    assert np.array_equal(batches[1][0].squeeze(), [range(14,20), range(14,20), range(14,20)])
    assert np.array_equal(batches[2][0].squeeze(), [[18,19] + range(4), range(14,20), range(2,8)])   
