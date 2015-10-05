import numpy as np

def generate_filterbank(min_freq, max_freq, last_low_freq,
        low_width, high_width):
    assert isinstance(min_freq, int) or min_freq.is_integer()
    assert isinstance(max_freq, int) or max_freq.is_integer()
    assert isinstance(last_low_freq, int) or last_low_freq.is_integer()
    assert isinstance(low_width, int) or low_width.is_integer()
    assert isinstance(high_width, int) or high_width.is_integer()
    
    assert high_width % 2  == 0
    assert low_width % 2  == 0
    assert (last_low_freq - min_freq) % low_width  == 0, ("last low freq "
        "needs to be exactly the center of a low_width filter band")
    assert max_freq >= last_low_freq
    assert (max_freq == last_low_freq or  
            (max_freq - (last_low_freq + low_width/2 + high_width/2)) % 
        high_width == 0), ("max freq needs to be exactly the center "
            "of a filter band")
    low_centers = range(min_freq,last_low_freq+1, low_width)
    high_start = last_low_freq + low_width/2 + high_width/2
    high_centers = range(high_start, max_freq+1, high_width)
    
    low_band = np.array([np.array(low_centers) - low_width/2, 
                         np.array(low_centers) + low_width/2]).T
    low_band = np.maximum(0.5, low_band)
    high_band = np.array([np.array(high_centers) - high_width/2, 
                         np.array(high_centers) + high_width/2]).T
    filterbank = np.concatenate((low_band, high_band))
    return filterbank
