from braindecode.datasets.filterbank import generate_filterbank
import numpy as np
import pytest

def test_generate_filterbank():
    filterbands = generate_filterbank(min_freq=0, max_freq=16, 
        last_low_freq=6, low_width=2, high_width=6)
    assert np.array_equal( 
        [[0.5, 1], [1,3],[3,5],[5,7], [7,13],[13,19]],
        filterbands)
    
def test_generate_filterbank_only_low_width_freqs():
    filterbands = generate_filterbank(min_freq=0, max_freq=12, 
        last_low_freq=12, low_width=4, high_width=6)
    assert np.array_equal([[0.5, 2], [2,6],[6,10],[10,14]], filterbands)

def test_generate_filterbank_failure():
    with pytest.raises(AssertionError) as excinfo:
        filterbands = generate_filterbank(min_freq=0, max_freq=18, 
            last_low_freq=6, low_width=2, high_width=6)
    assert excinfo.value.message == ("max freq needs to be exactly the center "
            "of a filter band")
    
    with pytest.raises(AssertionError) as excinfo:
        generate_filterbank(min_freq=0, max_freq=50, 
            last_low_freq=50, low_width=4, high_width=6)
    assert excinfo.value.message == ("last low freq "
        "needs to be exactly the center of a low_width filter band")
