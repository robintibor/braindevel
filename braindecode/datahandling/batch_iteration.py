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
        all_batch_inds = get_balanced_batches(n_flat_trials,batch_size=batch_size,
                      rng=rng, shuffle=shuffle)
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
    def __init__(self, batch_size, input_time_length, n_sample_preds,
            oversample_targets=False, remove_baseline_mean=False):
        self.batch_size = batch_size
        self.input_time_length = input_time_length
        self.n_sample_preds = n_sample_preds
        self.oversample_targets=oversample_targets
        self.remove_baseline_mean = remove_baseline_mean
        self.rng = RandomState(328774)
    
    def get_batches(self, dataset, shuffle):
        n_samples = dataset.get_topological_view().shape[0]
        n_lost_samples = self.input_time_length - self.n_sample_preds
        # Create blocks with start and end sample for entire dataset
        start_end_blocks = []
        last_input_sample = n_samples - self.input_time_length
        
        # To get predictions for all samples, also block at the start that
        # account for the lost samples at the start
        n_sample_start = -n_lost_samples
        n_sample_stop = last_input_sample + self.n_sample_preds
        for i_start_sample in range(n_sample_start, n_sample_stop, 
                self.n_sample_preds):
            i_adjusted_start = min(i_start_sample, n_samples - self.input_time_length)
            start_end_blocks.append((i_adjusted_start, i_adjusted_start + self.input_time_length))
        
        
        if shuffle and self.oversample_targets:
            # Hacky heuristic for oversampling...
            # duplicate those blocks that contain
            # more targets than the mean per block
            # if they contian 2 times as much as mean,
            # put them in 2 times, 3 times as much,
            # put in 3 times, etc.
            n_targets_in_block = []
            for start, end in start_end_blocks:
                n_targets_in_block.append(np.sum(dataset.y[start+n_lost_samples:end]))
            mean_targets_in_block = np.mean(n_targets_in_block)
            for i_block in xrange(len(start_end_blocks)):
                target_ratio = int(np.round(n_targets_in_block[i_block] / 
                    float(mean_targets_in_block)))
                if target_ratio > 1:
                    for _ in xrange(target_ratio - 1):
                        start_end_blocks.append(start_end_blocks[i_block])
            
        block_ind_batches = get_balanced_batches(len(start_end_blocks), 
            batch_size=self.batch_size, rng=self.rng, shuffle=shuffle)
       
        topo = dataset.get_topological_view()
        for block_inds in block_ind_batches:
            batch_size = len(block_inds)
            # have to wrap into float32, cause np.nan upcasts to float64!
            batch_topo = np.float32(np.ones((batch_size, topo.shape[1],
                 self.input_time_length, topo.shape[3])) * np.nan)
            batch_y = np.ones((self.n_sample_preds * batch_size, dataset.y.shape[1])) * np.nan
            for i_batch_block, i_block in enumerate(block_inds):
                start,end = start_end_blocks[i_block]
                # switch samples into last axis, (dim 2 shd be empty before)
                assert topo.shape[2] == 1
                # check if start is negative and end positive
                # could happen from padding blocks at start
                if start >= 0 or (start < 0 and end < 0):
                    batch_topo[i_batch_block] = topo[start:end].swapaxes(0,2)
                else:
                    assert start < 0 and end >= 0
                    # do wrap around padding
                    batch_topo[i_batch_block] = np.concatenate((topo[start:],
                        topo[:end])).swapaxes(0,2)
                assert start + n_lost_samples >= 0, ("Wrapping should only "
                    "account for lost samples at start and never lead "
                    "to negative y inds")
                batch_start_y = self.n_sample_preds * i_batch_block
                batch_end_y = batch_start_y + self.n_sample_preds
                batch_y[batch_start_y:batch_end_y] = (
                    dataset.y[start+n_lost_samples:end])
    
            if self.remove_baseline_mean:
                batch_topo -= np.mean(
                    # should produce mean per batchxchan
                    batch_topo[:,:,:n_lost_samples,:], axis=(2,3),
                    keepdims=True)
                
            assert not np.any(np.isnan(batch_topo))
            assert not np.any(np.isnan(batch_y))
            batch_y = batch_y.astype(np.int32)
            yield batch_topo, batch_y 

    def reset_rng(self):
        self.rng = RandomState(328774)
        
class CntWindowTrialIterator(object):
    """Cut out windows for several predictions from a continous dataset
     with a trial marker y signal."""
    def __init__(self, batch_size, input_time_length, n_sample_preds):
        self.batch_size = batch_size
        self.input_time_length = input_time_length
        self.n_sample_preds = n_sample_preds
        self.rng = RandomState(328774)
        
    def reset_rng(self):
        self.rng = RandomState(328774)
    
    def get_batches(self, dataset, shuffle):
        i_trial_starts, i_trial_ends = compute_trial_start_end_samples(
            dataset.y, check_trial_lengths_equal=False,
            input_time_length=self.input_time_length)
        for start, end in zip(i_trial_starts, i_trial_ends):
            assert end - start + 1 >= self.n_sample_preds, (
                "Trial should be longer or equal than number of sample preds, "
                "Trial length: {:d}, sample preds {:d}...".format(
                    end - start + 1,
                    self.n_sample_preds))
        start_end_blocks_per_trial = self.compute_start_end_block_inds(
            i_trial_starts, i_trial_ends)

        topo = dataset.get_topological_view()
        y = dataset.y

        return self.yield_block_batches(topo, y, start_end_blocks_per_trial, shuffle=shuffle)

    

    def compute_start_end_block_inds(self, i_trial_starts, i_trial_ends):
        # create start stop indices for all batches still 2d trial -> start stop
        start_end_blocks_per_trial = []
        for i_trial in xrange(len(i_trial_starts)):
            trial_start = i_trial_starts[i_trial] - 1
            trial_end = i_trial_ends[i_trial]
            start_end_blocks = get_start_end_blocks_for_trial(trial_start,
                trial_end, self.input_time_length, self.n_sample_preds)
        
            # check that block is correct, all predicted samples should be the trial samples
            all_predicted_samples = [range(start_end[1] - self.n_sample_preds + 1, 
                start_end[1]+1) for start_end in start_end_blocks]
            # this check takes about 50 ms in performance test
            # whereas loop itself takes only 5 ms.. deactivate it if not necessary
            assert np.array_equal(range(i_trial_starts[i_trial], i_trial_ends[i_trial] + 1), 
                           np.unique(np.concatenate(all_predicted_samples)))

            start_end_blocks_per_trial.append(start_end_blocks)
        return start_end_blocks_per_trial
    
    def yield_block_batches(self, topo, y, start_end_blocks_per_trial, shuffle):
        start_end_blocks_flat = np.concatenate(start_end_blocks_per_trial)
        if shuffle:
            self.rng.shuffle(start_end_blocks_flat)

        for i_block in xrange(0, len(start_end_blocks_flat), self.batch_size):
            i_block_stop = min(i_block + self.batch_size, len(start_end_blocks_flat))
            start_end_blocks = start_end_blocks_flat[i_block:i_block_stop]
            batch = create_batch(topo,y, start_end_blocks, self.n_sample_preds)
            yield batch
    
def get_start_end_blocks_for_trial(trial_start, trial_end, input_time_length,
        n_sample_preds):
    start_end_blocks = []
    i_window_end = trial_start
    while i_window_end < trial_end:
        i_window_end += n_sample_preds
        i_adjusted_end = min(i_window_end, trial_end)
        i_window_start = i_adjusted_end - input_time_length + 1
        start_end_blocks.append((i_window_start, i_adjusted_end))
    return start_end_blocks
        
def compute_trial_start_end_samples(y, check_trial_lengths_equal=True,
        input_time_length=None):
    """ Specify input time length to kick out trials that are too short after
    signal start."""
    trial_part = np.sum(y, 1) == 1
    boundaries = np.diff(trial_part.astype(np.int32))
    i_trial_starts = np.flatnonzero(boundaries == 1) + 1
    i_trial_ends = np.flatnonzero(boundaries == -1)
    # it can happen that a trial is only partially there since the
    # cnt signal was split in the middle of a trial
    # for now just remove these
    # use that start marker should always be before or equal to end marker
    if i_trial_starts[0] > i_trial_ends[0]:
        # cut out first trial which only has end marker
        i_trial_ends = i_trial_ends[1:]
    if i_trial_starts[-1] > i_trial_ends[-1]:
        # cut out last trial which only has start marker
        i_trial_starts = i_trial_starts[:-1]
    
    assert(len(i_trial_starts) == len(i_trial_ends))
    assert(np.all(i_trial_starts < i_trial_ends))
    # possibly remove first trials if they are too early
    if input_time_length is not None:
        while i_trial_starts[0] < input_time_length:
            i_trial_starts = i_trial_starts[1:]
            i_trial_ends = i_trial_ends[1:]
    if check_trial_lengths_equal:
        # just checking that all trial lengths are equal
        all_trial_lens = np.array(i_trial_ends) - np.array(i_trial_starts)
        assert all(all_trial_lens == all_trial_lens[0]), (
            "All trial lengths should be equal...")
    return i_trial_starts, i_trial_ends

def create_batch(topo, y, start_end_blocks, n_sample_preds):
    batch_y = [y[end-n_sample_preds+1:end+1] 
        for _, end in start_end_blocks]
    batch_topo = [topo[start:end+1].swapaxes(0,2)
        for start, end in start_end_blocks]
    batch_y = np.concatenate(batch_y).astype(np.int32)
    batch_topo = np.concatenate(batch_topo).astype(np.float32)
    return batch_topo, batch_y

def transform_batches_of_trials(batches, n_sample_preds,
    n_samples_per_trial):
    """Utility function to merge batches back into trials.
    Assumes batches are in trials x batches x channels x 0 x 1 format"""
    # restrict to relevant part
    batches = np.array(batches)[:,:,:,-n_sample_preds:]
    n_batches_per_trial = batches.shape[1]
    # first concatenate all batches except last one of each trial,
    # since last one contains overlap with the one before
    trial_batches = np.concatenate(batches[:,:-1].transpose(1,0,2,3), axis=2)
    legitimate_last_preds = n_samples_per_trial - n_sample_preds * (n_batches_per_trial - 1)
    trial_batches = np.append(trial_batches, batches[:,-1,:,-legitimate_last_preds:],axis=2)
    return trial_batches
