import numpy as np
import scipy.signal

def generate_filterbank(min_freq, max_freq, last_low_freq,
        low_width, low_overlap, high_width, high_overlap):
    # int checks probably not necessary?
    # since we are using np.arange now below, not range
    #assert isinstance(min_freq, int) or min_freq.is_integer()
    #assert isinstance(max_freq, int) or max_freq.is_integer()
    #assert isinstance(last_low_freq, int) or last_low_freq.is_integer()
    #assert isinstance(low_width, int) or low_width.is_integer()
    #assert isinstance(high_width, int) or high_width.is_integer()
    assert low_overlap < low_width, "overlap needs to be smaller than width"
    assert high_overlap < high_width, "overlap needs to be smaller than width"
    low_step = low_width - low_overlap
    assert (last_low_freq - min_freq) % low_step  == 0, ("last low freq "
        "needs to be exactly the center of a low_width filter band. "
        " Close center: {:.0f}".format(
            last_low_freq - ((last_low_freq - min_freq) % low_step)))
    assert max_freq >= last_low_freq
    high_step = high_width - high_overlap
    high_start = last_low_freq + high_step
    assert (max_freq == last_low_freq or  
        (max_freq - high_start) % high_step == 0), ("max freq needs to be "
            "exactly the center of a filter band "
        " Close center: {:d}".format(
            max_freq - ((max_freq - high_start) % high_step)))
    low_centers = np.arange(min_freq,last_low_freq+1e-5, low_step)
    high_centers = np.arange(high_start, max_freq+1e-5, high_step)
    
    low_band = np.array([np.array(low_centers) - low_width/2.0, 
                         np.array(low_centers) + low_width/2.0]).T
    low_band = np.maximum(0.2, low_band)
    high_band = np.array([np.array(high_centers) - high_width/2.0, 
                         np.array(high_centers) + high_width/2.0]).T
    filterbank = np.concatenate((low_band, high_band))
    return filterbank

def filter_is_stable(a):
    # from http://stackoverflow.com/a/8812737/1469195
    return np.all(np.abs(np.roots(a))<1)

def filterbank_is_stable(filterbank, filt_order, sampling_rate):
    nyq_freq = 0.5 * sampling_rate
    for low_cut_hz, high_cut_hz in filterbank:
        low = low_cut_hz / nyq_freq
        high = high_cut_hz / nyq_freq
        b, a = scipy.signal.butter(filt_order, [low, high], btype='bandpass')
        if not filter_is_stable(a):
            return False
    return True