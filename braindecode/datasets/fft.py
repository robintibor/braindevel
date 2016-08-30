import numpy as np
from braindecode.datasets.raw import CleanSignalMatrix
from scipy.signal.windows import blackmanharris
from braindecode.mywyrm.processing import segment_dat_fast

class FFTPreprocessor(object):
    def __init__(self, window_length, window_stride,
                include_phase=False,
                 square_amplitude=False,
                 frequency_start=None,
                 frequency_end=None,
                 fs=None):
        self.include_phase = include_phase
        self.square_amplitude = square_amplitude
        self.window_length = window_length
        self.window_stride = window_stride
        assert not ((fs is None) and ((frequency_start is not None) or
            (frequency_end is not None))), ("Need to know sampling rate "
            "to select subsets of frequencies")
        self.frequency_start = frequency_start
        self.frequency_end = frequency_end
        self.fs = fs
        
    
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
        
        if self.frequency_end is not None or self.frequency_start is not None:
            new_topo = self.select_subset_of_freqs(new_topo)
            
        dataset.set_topological_view(new_topo.astype(np.float32), 
            dataset.view_converter.axes)
        return dataset
    
    def select_subset_of_freqs(self, new_topo):
        assert self.fs is not None
        freq_bins = np.fft.rfftfreq(self.window_length, 1.0 / self.fs).tolist()
        # None means to ignore => take all...
        i_freq_start = None
        i_freq_stop = None
        if self.frequency_start is not None:
            assert self.frequency_start in freq_bins, ("Please choose center "
                "of a frequency bin. Choose from: {:s}".format(str(freq_bins)))
            i_freq_start = freq_bins.index(self.frequency_start)
        if self.frequency_end is not None:
            assert self.frequency_end in freq_bins, ("Please choose center "
                "of a frequency bin. Choose from: {:s}".format(str(freq_bins)))
            # +1 since index will be exclusive
            i_freq_stop = freq_bins.index(self.frequency_end) + 1
        new_topo = new_topo[:,:,:,i_freq_start:i_freq_stop]
        return new_topo
        
        
    
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
    """Expects #trialsx#channelsx#samples order"""
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

def amplitude_phase_to_complex(amplitude, phase):
    return amplitude * np.cos(phase) + amplitude * np.sin(phase) * 1j


def compute_amps_baseline_before(cnt, square, divide_win_length,
        marker_def=None):
    if marker_def is None:
        marker_def = dict([(str(i), [i]) for i in xrange(1,5)])
    trial_start = 0
    trial_stop = 4000
    win_length_ms = 500
    win_length = win_length_ms * cnt.fs / 1000.0
    win_stride = win_length_ms * cnt.fs / 1000.0
    
    epo = segment_dat_fast(cnt,marker_def=marker_def, ival=[trial_start,trial_stop])
    amplitudes = compute_power_spectra(epo.data.transpose(0,2,1),
        window_length=win_length, window_stride=win_stride,
        divide_win_length=divide_win_length,square_amplitude=square)
    baseline_epo = segment_dat_fast(cnt,marker_def=marker_def, 
        ival=[-500,0])
    
    baseline_amps = compute_power_spectra(
        baseline_epo.data.transpose(0,2,1),
        window_length=win_length, window_stride=win_stride,
        divide_win_length=divide_win_length,square_amplitude=square)
    
    # median across trials
    median_baseline_amp = np.median(baseline_amps, axis=(0))
    assert median_baseline_amp.shape[1] == 1, "should only have one timebin"
    corrected_amps = amplitudes / median_baseline_amp[np.newaxis]
    all_class_amps = []
    for i_class in xrange(len(marker_def)):
        this_class_amps = corrected_amps[epo.axes[0] == i_class]
        class_amp = np.log(np.median(this_class_amps, axis=(0,2)))
        all_class_amps.append(class_amp)
    
    all_class_amps = np.array(all_class_amps)
    return all_class_amps
    
def compute_amps_relative(cnt, square, divide_win_length,
        marker_def=None):
    if marker_def is None:
        marker_def = dict([(str(i), [i]) for i in xrange(1,5)])
    amplitudes, classes = compute_trial_amplitudes(cnt, square,
        divide_win_length, marker_def)

    median_baseline_amp = np.median(amplitudes, axis=(0,))
    corrected_amps = amplitudes / median_baseline_amp[None,:]
    all_class_amps = []
    for i_class in xrange(len(marker_def)):
        this_class_amps = corrected_amps[classes == i_class]
        class_amp = np.log(np.median(this_class_amps, axis=(0,2)))
        all_class_amps.append(class_amp)
    
    all_class_amps = np.array(all_class_amps)
    return all_class_amps


def compute_amps_to_rest(cnt, square, divide_win_length,
        marker_def=None, rest_class=2):
    """Assumes class 2 (0-based) is rest class"""
    if marker_def is None:
        marker_def = dict([(str(i), [i]) for i in xrange(1,5)])
    amplitudes, classes = compute_trial_amplitudes(cnt, square, 
        divide_win_length,
        marker_def)

    # to rest class
    median_baseline_amp = np.median(amplitudes[classes == rest_class],
        axis=(0,))
    corrected_amps = amplitudes / median_baseline_amp[None]
    all_class_amps = []
    for i_class in xrange(len(marker_def)):
        this_class_amps = corrected_amps[classes == i_class]
        class_amp = np.log(np.median(this_class_amps, axis=(0,2)))
        all_class_amps.append(class_amp)
    
    all_class_amps = np.array(all_class_amps)
    return all_class_amps

def compute_trial_amplitudes(cnt, square, divide_win_length, marker_def):
    trial_start = 0
    trial_stop = 4000
    trial_len = trial_stop - trial_start
    n_samples_per_trial = trial_len * cnt.fs / 1000.0
    epo = segment_dat_fast(cnt,marker_def=marker_def,
        ival=[trial_start, trial_stop])
    
    assert epo.data.shape[-2] == n_samples_per_trial 
    amplitudes = compute_power_spectra(epo.data.transpose(0,2,1),
        window_length=n_samples_per_trial, window_stride=n_samples_per_trial,
        divide_win_length=divide_win_length, square_amplitude=square)
    assert amplitudes.shape[2] == 1, "should only have one timebin"
    trial_classes = epo.axes[0]
    assert len(trial_classes) == len(amplitudes)
    return amplitudes, trial_classes

def compute_amps_to_others(cnt, square, divide_win_length):
    amplitudes, classes = compute_trial_amplitudes(cnt, square,
        divide_win_length)
    assert amplitudes.shape[2] == 1, "should only have one timebin"
    all_class_amps = []
    for i_class in xrange(4):
        this_class_amps = amplitudes[classes == i_class]
        other_class_amps = amplitudes[classes != i_class]
        median_other_amp = np.median(other_class_amps, axis=(0,))
        median_other_amp = median_other_amp[None]
        class_amp = np.log(np.median(this_class_amps / median_other_amp, 
            axis=(0,2)))
        all_class_amps.append(class_amp)
    
    all_class_amps = np.array(all_class_amps)
    return all_class_amps