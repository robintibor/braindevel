from sklearn.cross_validation import KFold
from numpy.random import RandomState
import numpy as np

class BalancedBatchIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.rng = RandomState(328774)
    
    def get_batches(self, dataset, shuffle):
        n_trials = dataset.get_topological_view().shape[0]
        batches = get_balanced_batches(n_trials,
            batch_size=self.batch_size, rng=self.rng, shuffle=shuffle)
        topo = dataset.get_topological_view()
        y = dataset.y
        for batch_inds in batches:
            yield (topo[batch_inds], y[batch_inds])

    def reset_rng(self):
        self.rng = RandomState(328774)
        
def get_balanced_batches(n_trials, batch_size, rng, shuffle):
    n_batches = n_trials // batch_size
    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial =  n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches
    all_inds = np.array(range(n_trials))
    if shuffle:
        rng.shuffle(all_inds)
    i_trial = 0
    end_trial = 0
    batches = []
    for i_batch in xrange(n_batches):
        end_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            end_trial += 1
        batch_inds = all_inds[range(i_trial, end_trial)]
        batches.append(batch_inds)
        i_trial = end_trial
    assert i_trial == n_trials
    return batches

class WindowsIterator(object):
    def __init__(self, n_samples_per_window, batch_size, sample_axes_name=0,
            n_sample_stride=1):
        """Note sample sample_axes_name should be 'c', 0, or 1 from bc01 convention!"""
        self.n_samples_per_window = n_samples_per_window
        self.rng = RandomState(348846723)
        self.batch_size = batch_size
        self.sample_axes_name = sample_axes_name
        self.n_sample_stride = n_sample_stride

    def get_batches(self, dataset, shuffle):
        sample_axes_dim = dataset.view_converter.axes.index(self.sample_axes_name)
        topo = dataset.get_topological_view()
        y = dataset.y
        return create_sample_window_batches(topo, y, self.batch_size,
             sample_axes_dim, self.n_samples_per_window, self.n_sample_stride, 
             shuffle=shuffle, rng=self.rng)
    
    def reset_rng(self):
        self.rng = RandomState(348846723)

def create_sample_window_batches(topo, y, batch_size, 
       sample_axes_dim, n_samples_per_window, n_sample_stride, shuffle, rng):
    """Creates batches of windows from given trials (topo should have trials 
    as first dimension). """
    n_trials = topo.shape[0]
    n_samples_per_trial = topo.shape[sample_axes_dim]
    # + 1 necessary since range exclusive...
    start_sample_inds = range(0, n_samples_per_trial - n_samples_per_window + 1,
        n_sample_stride)
    n_sample_windows = len(start_sample_inds)
    n_flat_trials = n_sample_windows * n_trials

    n_batches = n_flat_trials // batch_size
    if (n_batches > 1):
        folds = KFold(n_flat_trials,n_folds=n_batches,
                      random_state=rng, shuffle=shuffle)
        all_batch_inds = [f[1] for f in folds]
    else:
        all_batch_inds = [range(n_flat_trials)]
        if shuffle:
            rng.shuffle(all_batch_inds)

    for batch_inds in all_batch_inds:
        batch_topo_shape = list(topo.shape)
        batch_topo_shape[0] = len(batch_inds)
        batch_topo_shape[sample_axes_dim] = n_samples_per_window
        batch_topo = np.ones(batch_topo_shape, dtype=np.float32) * np.nan
        batch_y = np.ones(len(batch_inds), dtype=np.int32) * -1
        for i_batch_trial, i_flat_trial in enumerate(batch_inds):
            i_trial = i_flat_trial // n_sample_windows
            i_sample_window = i_flat_trial % n_sample_windows
            i_start_sample = start_sample_inds[i_sample_window]
            #http://stackoverflow.com/a/28685499/1469195
            batch_topo[i_batch_trial] = np.rollaxis(
                np.rollaxis(topo[i_trial], 
                    sample_axes_dim-1, 0)[i_start_sample:i_start_sample+n_samples_per_window],
                    0, 
                sample_axes_dim)
            batch_y[i_batch_trial] = y[i_trial]

        assert not np.any(np.isnan(batch_topo))
        assert not np.any(batch_y == -1)
        assert np.array_equal(batch_topo.shape, batch_topo_shape)
        # maybe remove this and remove multiplication with nan 
        # and assertion check in case
        # this is too slow?
        batch_topo = batch_topo.astype(np.float32)
        yield batch_topo, batch_y

class WindowsFromCntIterator(object):
    def __init__(self, batch_size, input_time_length):
        self.batch_size = batch_size
        self.input_time_length = input_time_length
        self.rng = RandomState(328774)
    
    def get_batches(self, dataset, shuffle):
        predictable_samples = len(dataset.y) - self.input_time_length + 1
        batches = get_balanced_batches(predictable_samples, 
            self.batch_size, rng=self.rng, shuffle=shuffle)

        topo = dataset.get_topological_view()
        for batch_inds in batches:
            topo_trials = []
            topo_y = []
            for i_start_sample in batch_inds:
                end_sample = i_start_sample+self.input_time_length-1
                topo_trials.append(topo[i_start_sample:end_sample+1])
                topo_y.append(dataset.y[end_sample])

            topo_batch = np.array(topo_trials).swapaxes(1,2)[:,:,:,:,0]
            topo_y = np.array(topo_y)
            yield (topo_batch, topo_y)

    def reset_rng(self):
        self.rng = RandomState(328774)
        
class CntWindowsFromCntIterator(object):
    def __init__(self, batch_size, input_time_length, n_sample_preds):
        self.batch_size = batch_size
        self.input_time_length = input_time_length
        self.n_sample_preds = n_sample_preds
        self.rng = RandomState(328774)
    
    def get_batches(self, dataset, shuffle):
        n_samples = dataset.get_topological_view().shape[0]
        n_lost_samples = self.input_time_length - self.n_sample_preds
        start_end_blocks = []
        for i_start_sample in range(0, n_samples - self.input_time_length + self.n_sample_preds, self.n_sample_preds):
            i_adjusted_start = min(i_start_sample, n_samples - self.input_time_length)
            start_end_blocks.append((i_adjusted_start, i_adjusted_start + self.input_time_length))
        
        block_inds = range(0, len(start_end_blocks))
        if shuffle:
            self.rng.shuffle(block_inds)
            
        topo = dataset.get_topological_view()
        block_step = 1
        if shuffle:
            # Only use the batch size when shuffling
            # makes it easier to later correctly get the predictions
            # when you are just using one trial per batch
            block_step = self.batch_size
        #block_step = self.batch_size
        for i_block in xrange(0,len(block_inds),block_step):
            n_blocks = min(block_step, len(block_inds) - i_block)
            # have to wrap into float32, cause np.nan upcasts to float64!
            batch_topo = np.float32(np.ones((n_blocks, topo.shape[1],
                 self.input_time_length, topo.shape[3])) * np.nan)
            batch_y = np.ones((self.n_sample_preds * n_blocks, dataset.y.shape[1])) * np.nan
            for i_stride in xrange(n_blocks):
                i_actual_block = block_inds[i_block + i_stride]
                start,end = start_end_blocks[i_actual_block]
                # switch samples into last axis, (dim 2 shd be empty before)
                assert topo.shape[2] == 1
                batch_topo[i_stride] = topo[start:end].swapaxes(0,2)
                start_y = self.n_sample_preds * i_stride
                end_y = start_y + self.n_sample_preds
                batch_y[start_y:end_y] = dataset.y[start+n_lost_samples:end]
                
            assert not np.any(np.isnan(batch_y))
            batch_y = batch_y.astype(np.int32)
            # reshape this from
            # batch 1 sample 1(all classes), batch 1 sample 2 (all classes), ..., batch 1 sample n, batch 2 sample2, ...
            # to
            # batch 1 sample 1, batch 2 sample 1, ....
            # (because that is the order of the output of the model using stridereshape etc)
            n_classes = dataset.y.shape[1]
            batch_y = batch_y.reshape(n_blocks,-1, n_classes).swapaxes(0,1).reshape(-1, n_classes)
            yield batch_topo, batch_y

    def reset_rng(self):
        self.rng = RandomState(328774)