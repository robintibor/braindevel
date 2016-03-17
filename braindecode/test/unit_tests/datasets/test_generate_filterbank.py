from braindecode.datasets.filterbank import generate_filterbank
import numpy as np
import pytest

def test_generate_filterbank():
    filterbands = generate_filterbank(min_freq=2, max_freq=16,
        last_low_freq=8, low_width=6, low_overlap=4,
        high_width=10, high_overlap=6)
    assert np.array_equal([[0.2,5],[1,7],[3,9],[5,11],[7,17],[11,21]],
        filterbands)
    
def test_generate_filterbank_only_low_width_freqs():
    filterbands = generate_filterbank(min_freq=2, max_freq=8,
        last_low_freq=8, low_width=6, low_overlap=4,
        high_width=10, high_overlap=6)
    assert np.array_equal([[0.2,5],[1,7],[3,9],[5,11]],
        filterbands)

def test_generate_filterbank_failure():
    with pytest.raises(AssertionError) as excinfo:
        generate_filterbank(min_freq=2, max_freq=22,
            last_low_freq=8, low_width=6, low_overlap=4,
            high_width=10, high_overlap=6)
    assert ("max freq needs to be exactly the center "
            "of a filter band  "
            "Nearest center: 20") == excinfo.value.message  
    
    with pytest.raises(AssertionError) as excinfo:
        generate_filterbank(min_freq=2, max_freq=20,
            last_low_freq=9, low_width=6, low_overlap=4,
            high_width=10, high_overlap=6)
    assert ("last low freq "
        "needs to be exactly the center of a low_width filter band.  "
        "Nearest center: 8") == excinfo.value.message 
