import numpy as np
from numpy.random import RandomState


def get_balanced_batches(n_trials, rng, shuffle, n_batches=None,
                         batch_size=None):
    """Create indices for batches balanced in size (batches will have maximum size difference of 1).
    Supply either batch size or number of batches. Resulting batches
    will not have the given batch size but rather the next largest batch size
    that allows to split the set into balanced batches (maximum size difference 1).

    Parameters
    ----------
    n_trials : int
        Size of set.
    rng :

    shuffle :
        Whether to shuffle indices before splitting set.
    n_batches :
         (Default value = None)
    batch_size :
         (Default value = None)

    Returns
    -------

    """
    assert batch_size is not None or n_batches is not None
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))

    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
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
    for i_batch in range(n_batches):
        end_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            end_trial += 1
        batch_inds = all_inds[range(i_trial, end_trial)]
        batches.append(batch_inds)
        i_trial = end_trial
    assert i_trial == n_trials
    return batches

class BalancedBatchSizeIterator(object):
    """
    Create batches of balanced size.
    Parameters
    ----------
    batch_size: int
        Resulting batches will not necessarily have the given batch size
        but rather the next largest batch size that allows to split the set into
        balanced batches (maximum size difference 1).
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.rng = RandomState(328774)

    def get_batches(self, dataset, shuffle):
        n_trials = dataset.X.shape[0]
        batches = get_balanced_batches(n_trials,
                                       batch_size=self.batch_size,
                                       rng=self.rng,
                                       shuffle=shuffle)
        for batch_inds in batches:
            yield (dataset.X[batch_inds], dataset.y[batch_inds])

    def reset_rng(self):
        self.rng = RandomState(328774)