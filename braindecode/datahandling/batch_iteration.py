from sklearn.cross_validation import KFold
from numpy.random import RandomState
import numpy as np

class BalancedBatchIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.rng = RandomState(328774)
    
    def get_batches(self, dataset, shuffle):
        num_trials = dataset.get_topological_view().shape[0]
        
        batches = get_balanced_batches(num_trials,
            batch_size=self.batch_size, rng=self.rng, shuffle=shuffle)
        for batch_inds in batches:
            yield (dataset.get_topological_view()[batch_inds],
                dataset.y[batch_inds])

    def reset_rng(self):
        self.rng = RandomState(328774)

def get_balanced_batches(num_trials, batch_size, rng, shuffle=True):
    # We will use the test folds as our mini-batches,
    # training fold indices are completely ignored here
    # test folds should all be distinct and together be the complete set
    n_batches = num_trials // batch_size
    if n_batches > 1:
        folds = KFold(num_trials, n_folds=n_batches, 
                      shuffle=shuffle, random_state=rng)
        test_folds = [f[1] for f in folds]
    else:
        # In case batch size is bigger than number of trials just return 
        # all trial indices
        batch = range(num_trials)
        if (shuffle):
            rng.shuffle(batch)
        test_folds = [batch]
    return test_folds


class WindowsIterator(object):
    def __init__(self,trial_window_fraction, batch_size, sample_axes_name=0,
            stride=1):
        """Note sample sample_axes_name should be 'c', 0, or 1 from bc01 convention!"""
        self.trial_window_fraction = trial_window_fraction
        self.rng = RandomState(348846723)
        self.batch_size = batch_size
        self.sample_axes_name = sample_axes_name
        self.stride = stride

    def get_batches(self, dataset, shuffle):
        sample_axes_dim = dataset.view_converter.axes.index(self.sample_axes_name)
        topo = dataset.get_topological_view()
        y = dataset.y
        return create_sample_window_batches(topo, y, self.batch_size,
             sample_axes_dim, self.trial_window_fraction, self.stride, 
             shuffle=shuffle, rng=self.rng)
    
    def reset_rng(self):
        self.rng = RandomState(348846723)

def create_sample_window_batches(topo, y, batch_size, 
       sample_axes_dim, trial_window_fraction, stride, shuffle, rng):
    """Creates batches of windows from given trials (topo should have trials 
    as first dimension). """
    n_trials = topo.shape[0]
    n_samples_per_trial = topo.shape[sample_axes_dim]
    n_samples_per_window = int(np.round(n_samples_per_trial * 
        trial_window_fraction))
    # + 1 necessary since range exclusive...
    start_sample_inds = range(0, n_samples_per_trial - n_samples_per_window + 1, stride)
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