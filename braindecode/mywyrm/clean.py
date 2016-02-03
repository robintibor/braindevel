import numpy as np
from braindecode.datasets.loaders import BBCIDataset
from braindecode.mywyrm.processing import (bandpass_cnt, segment_dat_fast,
    highpass_cnt, lowpass_cnt)
from wyrm.processing import select_channels, append_cnt, append_epo
from braindecode.datasets.signal_processor import SignalProcessor

class NoCleaner():
    def __init__(self, segment_ival=None, marker_def=None):
        self.marker_def = marker_def
        if self.marker_def is None:
            self.marker_def = {'1 - Right Hand': [1], '2 - Left Hand': [2], 
                    '3 - Rest': [3], '4 - Feet': [4]}
        self.segment_ival = segment_ival
        if self.segment_ival is None:
            self.segment_ival = [0, 4000]

    def clean(self, bbci_set_cnt):
        # Segment into trials and take all! :)
        # Segment just to select markers and kick out out of bounds
        # trials
        epo = segment_dat_fast(bbci_set_cnt, marker_def=self.marker_def, 
           ival=self.segment_ival)
        clean_trials = range(epo.data.shape[0])
        (rejected_chans, rejected_trials, clean_trials) = ([],[], clean_trials)
        return (rejected_chans, rejected_trials, clean_trials) 
        
class SingleSetCleaner():
    """ Determines rejected trials and channels """
    def __init__(self, eog_set, rejection_var_ival=[0,4000], 
            rejection_blink_ival=[-500,4000],
            max_min=600, whisker_percent=10, whisker_length=3,
            marker_def=None, low_cut_hz=0.1, high_cut_hz=None):
        local_vars = locals()
        del local_vars['self']
        self.__dict__.update(local_vars)
        
        if self.marker_def is None:
            self.marker_def = {'1 - Right Hand': [1], '2 - Left Hand': [2], 
                    '3 - Rest': [3], '4 - Feet': [4]}
        self.eog_set = SignalProcessor(set_loader=self.eog_set, 
            segment_ival=rejection_blink_ival, marker_def=self.marker_def)
        
    def clean(self, bbci_set_cnt):
        if self.low_cut_hz is not None:
            assert self.low_cut_hz > 0
        if self.high_cut_hz is not None:
            assert self.high_cut_hz < int(bbci_set_cnt.fs / 2), ("Frequency "
                "should be below Nyquist frequency.")
        if self.low_cut_hz is not None and self.high_cut_hz is not None:
            assert self.low_cut_hz < self.high_cut_hz
        
        cleaner = Cleaner(
                    bbci_set_cnt, 
                    self.eog_set, 
                    rejection_blink_ival=self.rejection_blink_ival, 
                    max_min=self.max_min, 
                    rejection_var_ival=self.rejection_var_ival, 
                    whisker_percent=self.whisker_percent, 
                    whisker_length=self.whisker_length,
                    low_cut_hz=self.low_cut_hz,
                    high_cut_hz=self.high_cut_hz,
                    filt_order=4,
                    marker_def=self.marker_def)
        cleaner.clean()
        
        self.rejected_chans = cleaner.rejected_chans # remember in case other cleaner needs it
        return (cleaner.rejected_chan_names, cleaner.rejected_trials,
            cleaner.clean_trials) 



class Cleaner(object):
    """ Real cleaning class, should get all necessary information for the cleaning"""
    def __init__(self, cnt, eog_set, rejection_blink_ival,
        max_min, rejection_var_ival, whisker_percent, whisker_length,
        low_cut_hz, high_cut_hz,filt_order, marker_def, preremoved_chans=None):
        local_vars = locals()
        del local_vars['self']
        self.__dict__.update(local_vars)
    
    def clean(self):
        self.load_and_preprocess_data()
        if self.preremoved_chans is not None:
            self.ignore_channels = True
        else:
            self.ignore_channels = False
        self.compute_rejected_chans_trials()
        
        
    def load_and_preprocess_data(self):
        # First create eog set for blink rejection
        self.eog_set.load_signal_and_markers()
        self.eog_set.segment_into_trials()
        self.eog_set.remove_continuous_signal()
        
        if self.preremoved_chans is not None:
            self.cnt = select_channels(self.cnt, self.preremoved_chans, invert=True)
        else:
            self.ignore_channels = False
        
        # Then create bandpassed set for variance rejection
        # in case low or high cut hz is given
        if self.low_cut_hz is not None and self.high_cut_hz is not None:
            self.cnt = bandpass_cnt(self.cnt, self.low_cut_hz, self.high_cut_hz,
                self.filt_order)
        elif self.low_cut_hz is not None:
            self.cnt = highpass_cnt(self.cnt, self.low_cut_hz, self.filt_order)
        elif self.high_cut_hz is not None:
            self.cnt = lowpass_cnt(self.cnt, self.high_cut_hz, self.filt_order)
        else:
            assert self.low_cut_hz is None and self.high_cut_hz is None
            
        
        
        self.epo = segment_dat_fast(self.cnt, marker_def=self.marker_def,
            ival=self.rejection_var_ival)
            
    def compute_rejected_chans_trials(self):
        orig_trials = range(self.epo.data.shape[0])
        good_trials = range(self.epo.data.shape[0])
        rejected_trials_max_min = compute_rejected_trials_max_min(
            self.eog_epo, max_min=self.max_min)
        good_trials = np.delete(good_trials, rejected_trials_max_min) # delete deletes indices
        variances = np.var(self.epo.data[good_trials], axis=1)
        rejected_chan_inds, rejected_trials_var = compute_rejected_channels_trials_by_variance(
            variances, self.whisker_percent, self.whisker_length, 
            self.ignore_channels)
        rejected_var_original = [good_trials[i] for i in rejected_trials_var]
        good_trials = np.delete(good_trials, rejected_trials_var) # delete deletes indices
        rejected_trials = np.setdiff1d(orig_trials, good_trials)
        rejected_chan_names = self.epo.axes[2][rejected_chan_inds]
        if self.preremoved_chans is not None:
            assert len(rejected_chan_names) == 0
            rejected_chan_names = self.preremoved_chans
        self.rejected_chan_names = rejected_chan_names
        self.rejected_trials = rejected_trials
        self.rejected_trials_max_min = rejected_trials_max_min
        self.rejected_var_original = rejected_var_original
        self.clean_trials = good_trials

def compute_rejected_channels_trials_cnt(cnt, eog_set, rejection_blink_ival,
        max_min, rejection_var_ival, whisker_percent, whisker_length,
        low_cut_hz, high_cut_hz,filt_order, marker_def, preremoved_chans=None):
    """ preremoved_chans : Only reject trials, before remove the `preremoved chans`.
    Useful for example if you want to remove same chans as in another set."""
    
    # First create eog set for blink rejection
    eog_set.load_signal_and_markers()
    eog_set.segment_into_trials()
    eog_set.remove_continuous_signal()
    
    if preremoved_chans is not None:
        cnt = select_channels(cnt, preremoved_chans, invert=True)
        ignore_channels = True
    else:
        ignore_channels = False
    
    # Then create bandpassed set for variance rejection
    # in case low or high cut hz is given
    if low_cut_hz is not None and high_cut_hz is not None:
        bandpassed_cnt = bandpass_cnt(cnt, low_cut_hz, high_cut_hz, filt_order)
    elif low_cut_hz is not None:
        bandpassed_cnt = highpass_cnt(cnt, low_cut_hz, filt_order)
    elif high_cut_hz is not None:
        bandpassed_cnt = lowpass_cnt(cnt, high_cut_hz, filt_order)
    else:
        assert low_cut_hz is None and high_cut_hz is None
        bandpassed_cnt = cnt
        
    
    
    epo = segment_dat_fast(bandpassed_cnt, marker_def=marker_def,
        ival=rejection_var_ival)
    (rejected_chan_names, rejected_trials, rejected_trials_max_min,
        rejected_var_original, 
        good_trials) = compute_rejected_channels_trials(epo, 
            eog_set.epo, max_min=max_min, 
            whisker_percent=whisker_percent, 
            whisker_length=whisker_length,
            ignore_channels=ignore_channels)
    
    if preremoved_chans is not None:
        assert len(rejected_chan_names) == 0
        rejected_chan_names = preremoved_chans
        
    return (rejected_chan_names, rejected_trials, rejected_trials_max_min,
        rejected_var_original, good_trials)
        

def compute_rejected_channels_trials(epo, eog_epo, max_min,
        whisker_percent, whisker_length, ignore_channels):
    orig_trials = range(epo.data.shape[0])
    good_trials = range(epo.data.shape[0])
    rejected_trials_max_min = compute_rejected_trials_max_min(
        eog_epo, max_min=max_min)
    good_trials = np.delete(good_trials, rejected_trials_max_min) # delete deletes indices
    variances = np.var(epo.data[good_trials], axis=1)
    rejected_chan_inds, rejected_trials_var = compute_rejected_channels_trials_by_variance(
        variances, whisker_percent, whisker_length, ignore_channels)
    rejected_var_original = [good_trials[i] for i in rejected_trials_var]
    good_trials = np.delete(good_trials, rejected_trials_var) # delete deletes indices
    rejected_trials = np.setdiff1d(orig_trials, good_trials)
    rejected_chan_names = epo.axes[2][rejected_chan_inds]
    return (rejected_chan_names, rejected_trials, rejected_trials_max_min,
        rejected_var_original, good_trials)

def compute_rejected_trials_max_min(epo, max_min):
    max_vals = np.max(epo.data, axis=1)
    min_vals = np.min(epo.data, axis=1)
    maxmin_diffs = max_vals -min_vals
    assert maxmin_diffs.ndim == 2 # trials x channels
    # from theses diffs, take maximum over chans, since we throw out trials if any chan
    # is exceeding the limit
    maxmin_diffs = np.max(maxmin_diffs, axis=1)
    assert maxmin_diffs.ndim == 1 # just trials
    rejected_trials_max_min = np.flatnonzero(maxmin_diffs > max_min)
    return rejected_trials_max_min

def get_variance_threshold(variances, whisker_percent, whisker_length):
    """Get the threshold variance, above which variance is defined as an outlier/to be rejected."""
    low_percentiles, high_percentiles = np.percentile(variances, (whisker_percent, 100-whisker_percent))
    threshold = high_percentiles + (high_percentiles - low_percentiles) * whisker_length

    return threshold

# test create set three trials, one trial has excessive variance, should be removed 
# create set with three channels, one excessive variance, should be removed
# create set 
def compute_rejected_channels_trials_by_variance(variances, whisker_percent,
    whisker_length, ignore_channels):
    orig_chan_inds = range(variances.shape[1])
    orig_trials = range(variances.shape[0])
    good_chan_inds = np.copy(orig_chan_inds)
    good_trials = np.copy(orig_trials)

    # remove trials with excessive variances
    bad_trials = compute_excessive_outlier_trials(variances, whisker_percent, whisker_length)
    good_trials = np.delete(good_trials, bad_trials, axis=0)
    variances = np.delete(variances, bad_trials, axis=0)

    # now remove channels (first)
    if not ignore_channels:
        no_further_rejections = False
        while not no_further_rejections:
            bad_chans = compute_outlier_chans(variances, whisker_percent,whisker_length)
            variances = np.delete(variances, bad_chans, axis=1)
            good_chan_inds = np.delete(good_chan_inds, bad_chans, axis=0)
            no_further_rejections = len(bad_chans) == 0

    # now remove trials (second)
    no_further_rejections = False
    while not no_further_rejections:
        bad_trials = compute_outlier_trials(variances, whisker_percent, whisker_length)
        good_trials = np.delete(good_trials, bad_trials, axis=0)
        variances = np.delete(variances, bad_trials, axis=0)
        no_further_rejections = len(bad_trials) == 0

    # remove unstable chans
    if not ignore_channels:
        bad_chans = compute_unstable_chans(variances, whisker_percent, whisker_length)
        variances = np.delete(variances, bad_chans, axis=1)
        good_chan_inds = np.delete(good_chan_inds, bad_chans, axis=0)
    
    rejected_chan_inds = np.setdiff1d(orig_chan_inds, good_chan_inds)
    rejected_trials = np.setdiff1d(orig_trials, good_trials)
    return rejected_chan_inds, rejected_trials

def compute_outlier_chans(variances, whisker_percent,whisker_length):
    num_trials = variances.shape[0]
    threshold = get_variance_threshold(variances, whisker_percent, whisker_length)
    above_threshold = variances > threshold
    # only remove any channels if more than 5 percent of trials across channels are exceeding the variance 
    if (np.sum(above_threshold) > 0.05*num_trials):
        fraction_of_all_outliers_per_chan = np.sum(above_threshold, axis=0) / float(np.sum(above_threshold))
        chan_has_many_bad_trials = np.mean(above_threshold, axis=0) > 0.05
        chan_has_large_fraction_of_outliers = fraction_of_all_outliers_per_chan > 0.1
        bad_chans = np.logical_and(chan_has_large_fraction_of_outliers, chan_has_many_bad_trials)
        assert bad_chans.ndim == 1
        bad_chans = np.flatnonzero(bad_chans)
    else:
        bad_chans = []
    return bad_chans

def compute_unstable_chans(variances, whisker_percent, whisker_length):
    variance_of_variance = np.var(variances,axis=0)
    threshold = get_variance_threshold(variance_of_variance, whisker_percent, whisker_length)
    bad_chans = variance_of_variance > threshold
    bad_chans = np.flatnonzero(bad_chans)
    return bad_chans

def compute_outlier_trials(variances, whisker_percent, whisker_length):
    threshold = get_variance_threshold(variances, whisker_percent, whisker_length)
    above_threshold = variances > threshold
    trials_one_chan_above_threshold = np.any(above_threshold, axis=1)
    outlier_trials = np.flatnonzero(trials_one_chan_above_threshold)
    return outlier_trials

def compute_excessive_outlier_trials(variances, whisker_percent, whisker_length):
    # clean trials with "excessive variance": 
    # trials, where 20 percent of chans are above 
    # whisker determined threshold
    threshold = get_variance_threshold(variances, whisker_percent, whisker_length)
    above_threshold = variances > threshold
    fraction_chans_above_threshold = np.mean(above_threshold, axis=1)
    assert fraction_chans_above_threshold.ndim == 1
    outlier_trials = np.flatnonzero(fraction_chans_above_threshold > 0.2)
    return outlier_trials