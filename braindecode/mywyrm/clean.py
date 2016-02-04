import numpy as np
from braindecode.mywyrm.processing import (bandpass_cnt, segment_dat_fast,
    highpass_cnt, lowpass_cnt)
from wyrm.processing import select_channels
from braindecode.datasets.signal_processor import SignalProcessor
from collections import namedtuple
import logging 
log = logging.getLogger(__name__)

CleanResult = namedtuple('CleanResult', ['rejected_chan_names',
    'rejected_trials', 'clean_trials', 'rejected_max_min',
    'rejected_var'])

def log_clean_result(clean_result):
    log.info("Rejected channels: {:s}".format(clean_result.rejected_chan_names))
    log.info("#Clean trials:     {:d}".format(len(clean_result.clean_trials)))
    log.info("#Rejected trials:  {:d}".format(len(clean_result.rejected_trials)))
    log.info("Fraction Clean:    {:.1f}%".format(
        100 * len(clean_result.clean_trials) / 
        (len(clean_result.clean_trials) + len(clean_result.rejected_trials))))
    log.info("(from maxmin):     {:d}".format(
        len(clean_result.rejected_max_min)))
    log.info("(from var):        {:d}".format(
        len(clean_result.rejected_var)))

class NoCleaner():
    def __init__(self, segment_ival=None, marker_def=None):
        self.marker_def = marker_def
        if self.marker_def is None:
            self.marker_def = {'1 - Right Hand': [1], '2 - Left Hand': [2], 
                    '3 - Rest': [3], '4 - Feet': [4]}
        self.segment_ival = segment_ival
        if self.segment_ival is None:
            self.segment_ival = [0, 4000]

    def clean(self, bbci_set_cnt, ignore_chans=False):
        # Segment into trials and take all! :)
        # Segment just to select markers and kick out out of bounds
        # trials
        # chans ignored always anyways... so parameter does not
        # matter
        epo = segment_dat_fast(bbci_set_cnt, marker_def=self.marker_def, 
           ival=self.segment_ival)
        clean_trials = range(epo.data.shape[0])
        
        clean_result = CleanResult(rejected_chan_names=[],
            rejected_trials=[],
            clean_trials=clean_trials,
            rejected_max_min=[],
            rejected_var=[])
        return clean_result
        

class SetCleaner():
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
        
    def clean(self, bbci_set_cnt, ignore_chans=False):
        """preremoved_chans will remove the given channels (as strings) and 
        not reject any further channels. Useful if you cleaned a train set
        and intend to clean a test set trials while rejecting the same 
        channels."""
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
                    marker_def=self.marker_def,
                    ignore_chans=ignore_chans)
        cleaner.clean()
        
        clean_result = CleanResult(rejected_chan_names=cleaner.rejected_chan_names,
            rejected_trials=cleaner.rejected_trials,
            clean_trials=cleaner.clean_trials,
            rejected_max_min=cleaner.rejected_trials_max_min,
            rejected_var=cleaner.rejected_var_original)
        
        
        return clean_result



class Cleaner(object):
    """ Real cleaning class, should get all necessary information for the cleaning"""
    def __init__(self, cnt, eog_set, rejection_blink_ival,
        max_min, rejection_var_ival, whisker_percent, whisker_length,
        low_cut_hz, high_cut_hz,filt_order, marker_def, ignore_chans=False):
        local_vars = locals()
        del local_vars['self']
        self.__dict__.update(local_vars)
    
    def clean(self):
        self.load_and_preprocess_data()
        self.compute_rejected_chans_trials()
        del self.epo # No longer needed
        del self.eog_set # no longer needed
        
        
    def load_and_preprocess_data(self):
        # First create eog set for blink rejection
        self.eog_set.load_signal_and_markers()
        self.eog_set.segment_into_trials()
        self.eog_set.remove_continuous_signal()
        
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

        # Finally create trials        
        self.epo = segment_dat_fast(self.cnt, marker_def=self.marker_def,
            ival=self.rejection_var_ival)
        del self.cnt # No longer needed
            
    def compute_rejected_chans_trials(self):
        # Remember number of original trials,
        # successively remove trials,
        # first by max min then by variance criterion
        orig_trials = range(self.epo.data.shape[0])
        good_trials = range(self.epo.data.shape[0])
        rejected_trials_max_min = compute_rejected_trials_max_min(
            self.eog_set.epo, threshold=self.max_min)
        good_trials = np.delete(good_trials, rejected_trials_max_min) # delete deletes indices
        variances = np.var(self.epo.data[good_trials], axis=1)
        rejected_chan_inds, rejected_trials_var = compute_rejected_channels_trials_by_variance(
            variances, self.whisker_percent, self.whisker_length, 
            self.ignore_chans)
        rejected_var_original = [good_trials[i] for i in rejected_trials_var]
        # delete deletes indices, so e.g if after max min cleaning
        # the remaining good trials were 3,5,6,9 and now rejected trials
        # were 0,2 then 3 and 6 will be removed
        # and 3,9 are the actually remaining good/clean trials
        good_trials = np.delete(good_trials, rejected_trials_var) # delete deletes indices
        rejected_trials = np.setdiff1d(orig_trials, good_trials)
        rejected_chan_names = self.epo.axes[2][rejected_chan_inds]
        if self.ignore_chans:
            assert len(rejected_chan_names) == 0
        self.rejected_chan_names = rejected_chan_names
        self.rejected_trials = rejected_trials
        self.rejected_trials_max_min = rejected_trials_max_min
        self.rejected_var_original = rejected_var_original
        self.clean_trials = good_trials

def compute_rejected_trials_max_min(epo, threshold):
    max_vals = np.max(epo.data, axis=1)
    min_vals = np.min(epo.data, axis=1)
    maxmin_diffs = max_vals -min_vals
    assert maxmin_diffs.ndim == 2 # trials x channels
    # from theses diffs, take maximum over chans, since we throw out trials if any chan
    # is exceeding the limit
    maxmin_diffs = np.max(maxmin_diffs, axis=1)
    assert maxmin_diffs.ndim == 1 # just trials
    rejected_trials_max_min = np.flatnonzero(maxmin_diffs > threshold)
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
    whisker_length, ignore_chans):
    orig_chan_inds = range(variances.shape[1])
    orig_trials = range(variances.shape[0])
    good_chan_inds = np.copy(orig_chan_inds)
    good_trials = np.copy(orig_trials)

    # remove trials with excessive variances
    bad_trials = compute_excessive_outlier_trials(variances, whisker_percent, whisker_length)
    good_trials = np.delete(good_trials, bad_trials, axis=0)
    variances = np.delete(variances, bad_trials, axis=0)

    # now remove channels (first)
    if not ignore_chans:
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
    if not ignore_chans:
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