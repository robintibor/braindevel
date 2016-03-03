import numpy as np
import random

def perm_mean_diffs_sampled(a,b, n_diffs=None):
    """Compute differences between all permutations of  labels.
    Version that samples.
    Parameters
    --------------
    a: list or numpy array
    b: list or numpy array
    n_diffs: int
        How many diffs/samples to compute.
    Returns
    -------
    all_diffs: 1d-array of float
        Sampled mean differences.
    """
    
    n_exps = len(a)
    all_bit_masks = [2 ** n for n in xrange(n_exps-1,-1,-1)]
    if n_diffs is None:
        n_diffs = 2**n_exps
        i_all_masks = xrange(n_diffs)
    else:
        random.seed(39483948)
        i_all_masks = random.sample(xrange(2**n_exps), n_diffs)
    all_diffs = np.float32(np.ones(n_diffs) * np.nan)
    for i_diff, i_mask in enumerate(i_all_masks):
        # masks has -1s and 1s,
        # 1 interpretable as
        # correct value selected
        # -1 as randomly flipped value/"incorrect" value selected
        # *2 makes values between 2 and 0, then -1 to make 
        # values between 1 and -1
        mask = (np.bitwise_and(i_mask, all_bit_masks) > 0) * 2 - 1
        diff = np.mean((mask * a)  -mask * b)
        all_diffs[i_diff] = diff
    return all_diffs

def perm_mean_diffs(a,b):
    """Compute differences between all permutations of  labels.
    Assumes a and b are paired values,
    a are values with label 0 and b with label 1.
    Computes mean differences for all possible   
    switches of 0 and 1 (but keeping pairs together, i.e.
    2 ^ len(a) switches).
    
    Parameters
    --------------
    a: list or numpy array
    b: list or numpy array
    
    Returns
    -------
    diffs: 1d-numpy array
        Differences between means of labelled values
        for all label-switched values.
    """
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    n_exps = len(a)
    all_masks = _create_masks(n_exps)
    diffs = _compute_diffs(a, b, all_masks)
    return diffs


def _create_masks(n_exps):
    """ Create all (2^n_exps) binary selection masks for this number of experiments.
    E.g. for 3 experiments:
    False, False, False
    False, False, True
    False, True, False
    False, True, True
    True, False, False
    True, False, True
    True, True, False
    True, True, True""" 
    all_masks = np.array([[False] * n_exps] * (2 ** n_exps))
    i_block_size = all_masks.shape[0] / 2 
    for i_col in xrange(0,all_masks.shape[1]):
        for i_row in xrange(0,all_masks.shape[0], i_block_size * 2):
            all_masks[i_row:i_row+i_block_size,i_col] = [[True]] * i_block_size
        i_block_size /= 2
    return all_masks
    
def _compute_diffs(a, b, all_masks):
    # first add "first set" part
    # positive labels from a
    # and negative labels from b
    diffs = all_masks * a
    diffs += (1 - all_masks) * b
    # subtract "second set" part
    # negative labels from a
    # positive labels from b
    diffs -= (1 - all_masks) * a
    diffs -= all_masks * b
    return np.mean(diffs, axis=1)
