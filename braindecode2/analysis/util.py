import numpy as np
from scipy.signal import blackmanharris
import scipy.signal

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


def lowpass_topo(topo, high_cut_hz, sampling_rate, axis=0, filt_order=4):
    nyq_freq = 0.5 * sampling_rate
    b, a = scipy.signal.butter(filt_order, high_cut_hz / nyq_freq, btype='lowpass')
    filtered = scipy.signal.filtfilt(b,a, topo, axis=0)
    return filtered

def highpass_topo(topo, low_cut_hz, sampling_rate, axis=0, filt_order=4):
    nyq_freq = 0.5 * sampling_rate
    b, a = scipy.signal.butter(filt_order, low_cut_hz / nyq_freq, btype='highpass')
    filtered = scipy.signal.filtfilt(b,a, topo, axis=0)
    return filtered

def bandpass_topo(topo, low_cut_hz, high_cut_hz, sampling_rate, axis=0, filt_order=4):
    nyq_freq = 0.5 * sampling_rate
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype='bandpass')
    filtered = scipy.signal.filtfilt(b,a, topo, axis=0)
    return filtered