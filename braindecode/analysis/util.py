import numpy as np
from scipy.signal import blackmanharris

def amplitudes_and_freqs(inputs, sampling_rate, axis=1, n=None):
    amplitudes = np.abs(np.fft.rfft(inputs, axis=axis, n=n))
    n_samples = n
    if n_samples is None:
        n_samples = inputs.shape[axis]
    freq_bins = np.fft.rfftfreq(n_samples, d=1.0/sampling_rate)
    return amplitudes, freq_bins

def bps_and_freqs(inputs, sampling_rate, axis=1, n=None):
    if n is None:
        n = inputs.shape[axis]
    amplitudes, freqs  = amplitudes_and_freqs(inputs, sampling_rate, axis, n)
    return (amplitudes * amplitudes) / n, freqs
    
def multiply_blackmann_harris_window(inputs, axis=1):
    n = inputs.shape[axis]
    w = blackmanharris(n)
    #w=w/np.mean(w)
    # make w have same dimensionality as inputs,
    # to ensure correct dimensions are multiplied
    for i_axis in xrange(inputs.ndim):
        if i_axis != axis:
            w = np.expand_dims(w, i_axis)
    return w * inputs