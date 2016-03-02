import numpy as np

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
