import numpy as np

def bps_and_freqs(weights, axis=1, sampling_rate=150.0):
    bps = np.abs(np.fft.rfft(weights, axis=axis))
    freq_bins = np.fft.rfftfreq(weights.shape[axis], d=1.0/sampling_rate)
    return bps, freq_bins