import numpy as np

def bps_and_freqs(weights, sampling_rate, axis=1, n=None):
    bps = np.abs(np.fft.rfft(weights, axis=axis, n=n))
    n_samples = n
    if n_samples is None:
        n_samples = weights.shape[axis]
    freq_bins = np.fft.rfftfreq(n_samples, d=1.0/sampling_rate)
    return bps, freq_bins