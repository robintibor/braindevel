import numpy as np
from braindevel.analysis.stats import _create_masks, perm_mean_diffs

def test_create_masks():
    mask =  _create_masks(3)
    assert np.array_equal(np.array(
          [[ True,  True,  True],
           [ True,  True, False],
           [ True, False,  True],
           [ True, False, False],
           [False,  True,  True],
           [False,  True, False],
           [False, False,  True],
           [False, False, False]], dtype=bool), mask)
    
def test_perm_diffs():
    a = np.array([0, 3, 5])
    b = np.array([1, 2, 7])
    exp_diffs = np.array([-2,2,-4,0,0,4,-2,2]) / 3.0
    diffs = perm_mean_diffs(a,b)
    np.allclose(exp_diffs, diffs)
    
    
    
