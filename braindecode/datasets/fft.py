import numpy as np
from braindecode.datasets.raw import CleanSignalMatrix
from scipy.signal.windows import blackmanharris

class FFTPreprocessor(object):
    def __init__(self, window_length, window_stride,
                include_phase=False,
                 square_amplitude=False):
        self.include_phase = include_phase
        self.square_amplitude = square_amplitude
        self.window_length = window_length
        self.window_stride = window_stride
        
    
    def apply(self, dataset, can_fit=False):
        old_topo = dataset.get_topological_view().squeeze()
        if self.include_phase:
            new_topo = compute_power_and_phase(old_topo,
               window_length=self.window_length,
               window_stride=self.window_stride,
               divide_win_length=False, 
               square_amplitude=self.square_amplitude,
                phases_diff=False)
        else:
            new_topo = compute_power_spectra(old_topo,
               window_length=self.window_length,
               window_stride=self.window_stride,
               divide_win_length=False, 
               square_amplitude=self.square_amplitude)
        dataset.set_topological_view(new_topo.astype(np.float32), 
            dataset.view_converter.axes)
        return dataset
    
class FFTCleanSignalMatrix(CleanSignalMatrix):
    def __init__(self, transform_function_and_args,
        frequency_start=None, frequency_stop=None, **kwargs):
        self.frequency_start = frequency_start
        self.frequency_stop = frequency_stop
        self.transform_function_and_args = transform_function_and_args
        super(FFTCleanSignalMatrix, self).__init__(**kwargs)
        
    def load_signal(self):
        self.clean_and_load_set()
        self.transform_with_fft()
            
    def transform_with_fft(self):
        # transpose as transform functions expect 
        # #trials x#channels x#samples order
        # (in wyrm framework it is #trials x#samples x#channels normally)
        trials = self.signal_processor.epo.data.transpose(0,2,1)
        fs = self.signal_processor.epo.fs
        win_length_ms, win_stride_ms = 500,250
        assert (win_length_ms == 500 and win_stride_ms == 250), ("if not, "
            "check if fft_pipeline function would still work")
        
        
        win_length = win_length_ms * fs / 1000
        win_stride = win_stride_ms * fs / 1000
        # Make sure that window length in ms is exactly fitting 
        # a certain number of samples
        assert (win_length_ms * fs) % 1000 == 0
        assert (win_stride_ms * fs) % 1000 == 0
        transform_func = self.transform_function_and_args[0]
        transform_kwargs = self.transform_function_and_args[1]
        # transform function will return 
        # #trials x #channels x#timebins x #freqboms
        transformed_trials = transform_func(trials, 
                window_length=win_length, 
                window_stride=win_stride, **transform_kwargs)
        self.signal_processor.epo.data = transformed_trials.transpose(0,2,1,3) # should be
        #transposed back in pylearnbbcidataset class

        # possibly select frequencies
        freq_bins = np.fft.rfftfreq(win_length, 1.0/self.signal_processor.epo.fs)
        if self.frequency_start is not None:
            freq_bin_start = freq_bins.tolist().index(self.frequency_start)
        else:
            freq_bin_start = 0
        if self.frequency_stop is not None:
            # + 1 as later indexing will exclude stop
            freq_bin_stop = freq_bins.tolist().index(self.frequency_stop) + 1
        else:
            freq_bin_stop = self.signal_processor.epo.data.shape[-1]
        self.signal_processor.epo.data = self.signal_processor.epo.data[:,:,:,
            freq_bin_start:freq_bin_stop]

def compute_power_and_phase(trials, window_length, window_stride,
        divide_win_length, square_amplitude, phases_diff):
    """Expects trials #trialsx#channelsx#samples order"""
    fft_trials = compute_short_time_fourier_transform(trials, 
        window_length=window_length, window_stride=window_stride)
    # now #trialsx#channelsx#samplesx#freqs
    # Todelay: Not sure if division by window length is necessary/correct?
    power_spectra = np.abs(fft_trials)
    if (square_amplitude):
        power_spectra = power_spectra ** 2
    if (divide_win_length):
        power_spectra = power_spectra / window_length
    phases = np.angle(fft_trials)
    if phases_diff:
        # Transform phases to "speed": diff of phase
        # minus diff of phase in timebin before for each
        # frequency bin separately
        phases =  phases[:, :, 1:, :] - phases[:, :, :-1, :]
        # pad a zero at the beginning to preserve dimensionality
        phases = np.pad(phases, ((0,0),(0,0),(1,0),(0,0)), 
            mode='constant', constant_values=0)
    power_and_phases = np.concatenate((power_spectra, phases), axis=1)
    return power_and_phases

def compute_power_spectra(trials, window_length, window_stride, 
        divide_win_length, square_amplitude):
    """Expects trials #trialsx#channelsx#samples order"""
    fft_trials = compute_short_time_fourier_transform(trials, 
        window_length=window_length, window_stride=window_stride)
    # Todelay: Not sure if division by window length is necessary/correct?
    # Hopefully does not matter since standardization will divide by variance
    # anyways
    power_spectra = np.abs(fft_trials)
    if (square_amplitude):
        power_spectra = power_spectra ** 2
    if (divide_win_length):
        power_spectra = power_spectra / window_length
    return power_spectra
    

def compute_short_time_fourier_transform(trials, window_length, window_stride):
    """Expects trials #trialsx#channelsx#samples order"""
    start_times = np.arange(0, trials.shape[2] - window_length+1, window_stride)
    freq_bins = int(np.floor(window_length / 2) + 1)
    fft_trials = np.empty((trials.shape[0], trials.shape[1], len(start_times), freq_bins), dtype=complex)
    for time_bin, start_time in enumerate(start_times):        
        w = blackmanharris(window_length)
        w=w/np.linalg.norm(w)
        trials_for_fft = trials[:,:,start_time:start_time+window_length] * w
        fft_trial = np.fft.rfft(trials_for_fft, axis=2)
        fft_trials[:,:,time_bin, :] = fft_trial
    return fft_trials
