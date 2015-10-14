import numpy as np
from braindecode.datasets.bbci_dataset import BBCIDataset
from braindecode.mywyrm.processing import (bandpass_cnt, segment_dat_fast)
from wyrm.processing import select_channels, append_cnt, append_epo

class BBCISetNoCleaner():
    def clean(self, bbci_set_cnt):
        clean_trials = range(len(bbci_set_cnt.markers))
        (rejected_chans, rejected_trials, clean_trials) = ([],[], clean_trials)
        return (rejected_chans, rejected_trials, clean_trials) 
        
class BBCISetCleaner():
    """ Determines rejected trials and channels """
    def __init__(self, eog_set, rejection_var_ival=[0,4000], 
            rejection_blink_ival=[-500,4000],
            max_min=600, whisker_percent=10, whisker_length=3):
        local_vars = locals()
        del local_vars['self']
        self.__dict__.update(local_vars)
        
    def clean(self, bbci_set_cnt):
        (rejected_chans, rejected_trials, reject_max_min, reject_var, 
                clean_trials) =  compute_rejected_channels_trials_cnt(
                    bbci_set_cnt, 
                    self.eog_set, 
                    rejection_blink_ival=self.rejection_blink_ival, 
                    max_min=self.max_min, 
                    rejection_var_ival=self.rejection_var_ival, 
                    whisker_percent=self.whisker_percent, 
                    whisker_length=self.whisker_length,
                    low_cut_hz=0.5,
                    high_cut_hz=150,
                    filt_order=4)
        self.rejected_chans = rejected_chans # remember in case other cleaner needs it
        return (rejected_chans, rejected_trials, clean_trials) 
        
class BBCITwoSetsCleaner():
    """ Determines rejected trials and channels """
    def __init__(self, filename, second_filename,
        load_sensor_names,
        rejection_var_ival=[0,4000], 
            rejection_blink_ival=[-500,4000],
            max_min=600, whisker_percent=10, whisker_length=3):
        local_vars = locals()
        del local_vars['self']
        self.__dict__.update(local_vars)

    def clean(self, bbci_set_cnt, filename):
        second_cnt = self.load_second_cnt()
        
        n_first_trials = len(bbci_set_cnt.markers)
        # determine individual trial 
        (rejected_chans, rejected_trials, reject_max_min, reject_var, 
                clean_trials) =  compute_rejected_channels_trials_two_sets_cnt(
                    first_cnt=bbci_set_cnt,
                    second_cnt=second_cnt,
                    first_filename=self.filename, 
                    second_filename=self.second_filename,
                    rejection_blink_ival=self.rejection_blink_ival, 
                    max_min=self.max_min, 
                    rejection_var_ival=self.rejection_var_ival, 
                    whisker_percent=self.whisker_percent, 
                    whisker_length=self.whisker_length,
                    low_cut_hz=0.5,
                    high_cut_hz=150,
                    filt_order=4)
        assert np.intersect1d(rejected_trials, clean_trials).size == 0
        rejected_trials = np.array(rejected_trials)
        rejected_trials_first = rejected_trials[rejected_trials < n_first_trials]
        self.rejected_trials_second = (rejected_trials[rejected_trials >= n_first_trials] -
            n_first_trials)

        clean_trials = np.array(clean_trials)
        clean_trials_first = clean_trials[clean_trials < n_first_trials]
        self.clean_trials_second = (clean_trials[clean_trials >= n_first_trials] -
             n_first_trials)
        self.rejected_chans = rejected_chans # remember in case other cleaner needs it
        assert np.array_equal(np.union1d(rejected_trials_first, clean_trials_first), 
            range(len(bbci_set_cnt.markers)))
        assert np.array_equal(np.union1d(self.rejected_trials_second, 
            self.clean_trials_second), 
            range(len(second_cnt.markers)))
        return (rejected_chans, rejected_trials_first, clean_trials_first) 
    
    def load_second_cnt(self):
        set_2 = BBCIDataset(self.second_filename, 
            sensor_names=self.load_sensor_names) # for debug, allow load sensor names
        set_2.load_continuous_signal()
        set_2.add_markers()
        set_2.segment_into_trials()
        return set_2.cnt

class BBCISecondSetCleaner():
    """ Class just returns already computed values from a twoset cleaner """
    def __init__(self, filename, first_set_cleaner, rejection_var_ival=[0,4000], 
            rejection_blink_ival=[-500,4000],
            max_min=600, whisker_percent=10, whisker_length=3):
        local_vars = locals()
        del local_vars['self']
        self.__dict__.update(local_vars)
        # TODELAY: check that first set cleaner had same cleaning params
        # or just remove args
        
    def clean(self, bbci_set_cnt, filename):
        """ Don't do any cleaning, just return clean values already computed
        from a two-set cleaner """
        assert hasattr(self.first_set_cleaner, 'rejected_chans')
        assert hasattr(self.first_set_cleaner, 'rejected_trials_second')
        assert hasattr(self.first_set_cleaner, 'clean_trials_second')
        assert self.first_set_cleaner.second_filename == self.filename
                
        return (self.first_set_cleaner.rejected_chans,
                self.first_set_cleaner.rejected_trials_second, 
                self.first_set_cleaner.clean_trials_second) 
    
class BBCISecondSetOnlyChanCleaner():
    """ Class just returns already computed values from a twoset cleaner,
    this one will keep all trials but remove cleaned chans. """
    def __init__(self, first_set_cleaner, rejection_var_ival=[0,4000], 
            rejection_blink_ival=[-500,4000],
            max_min=600, whisker_percent=10, whisker_length=3):
        local_vars = locals()
        del local_vars['self']
        self.__dict__.update(local_vars)
        # TODELAY: check that first set cleaner had same cleaning params
        # or just remove args
        
    def clean(self, bbci_set_cnt):
        """ Don't do any cleaning, just return clean values already computed
        from a two-set cleaner """
        assert hasattr(self.first_set_cleaner, 'rejected_chans')
        rejected_trials = []
        clean_trials = range(len(bbci_set_cnt.markers))
        return (self.first_set_cleaner.rejected_chans,
                rejected_trials, 
                clean_trials) 

def compute_rejected_channels_trials_two_sets_cnt(first_cnt, second_cnt,
    first_filename, second_filename, rejection_blink_ival,
        max_min, rejection_var_ival, whisker_percent, whisker_length,
        low_cut_hz, high_cut_hz,filt_order):
    # First create eog set for blink rejection
    eog_sensor_names = BBCIDataset.get_all_sensors(first_filename, pattern="EO")
    eog_set_1 = BBCIDataset(filename=first_filename, 
                    sensor_names=eog_sensor_names, cnt_preprocessors=[],
                    epo_preprocessors=[],
                    segment_ival=rejection_blink_ival)
    eog_set_1.load_continuous_signal()
    eog_set_1.add_markers()
    eog_set_1.segment_into_trials()
    eog_set_1.remove_continuous_signal()
    
    eog_set_2 = BBCIDataset(filename=second_filename, 
                    sensor_names=eog_sensor_names, cnt_preprocessors=[],
                    epo_preprocessors=[],
                    segment_ival=rejection_blink_ival)
    eog_set_2.load_continuous_signal()
    eog_set_2.add_markers()
    eog_set_2.segment_into_trials()
    eog_set_2.remove_continuous_signal()
    
    merged_eog_epo = append_epo(eog_set_1.epo, eog_set_2.epo)
    
    merged_cnt = append_cnt(first_cnt, second_cnt)
    # Then create bandpassed set for variance rejection
    bandpassed_cnt = bandpass_cnt(merged_cnt, low_cut_hz, high_cut_hz, filt_order)
    epo = segment_dat_fast(bandpassed_cnt, 
           marker_def={'1 - Right Hand': [1], '2 - Left Hand': [2], 
               '3 - Rest': [3], '4 - Feet': [4]}, 
           ival=rejection_var_ival)
    return compute_rejected_channels_trials(epo, 
            merged_eog_epo, max_min=max_min, 
            whisker_percent=whisker_percent, 
            whisker_length=whisker_length)
    
def compute_rejected_channels_trials_cnt(cnt, eog_set, rejection_blink_ival,
        max_min, rejection_var_ival, whisker_percent, whisker_length,
        low_cut_hz, high_cut_hz,filt_order):
    # First create eog set for blink rejection
    eog_set.load_signal_and_markers()
    eog_set.segment_into_trials()
    eog_set.remove_continuous_signal()
    
    # Then create bandpassed set for variance rejection
    bandpassed_cnt = bandpass_cnt(cnt, low_cut_hz, high_cut_hz, filt_order)
    epo = segment_dat_fast(bandpassed_cnt, 
           marker_def={'1 - Right Hand': [1], '2 - Left Hand': [2], 
               '3 - Rest': [3], '4 - Feet': [4]}, 
           ival=rejection_var_ival)
    return compute_rejected_channels_trials(epo, 
            eog_set.epo, max_min=max_min, 
            whisker_percent=whisker_percent, 
            whisker_length=whisker_length)

def compute_rejected_channels_trials(epo, eog_epo, max_min,
        whisker_percent, whisker_length):
    orig_trials = range(epo.data.shape[0])
    good_trials = range(epo.data.shape[0])
    rejected_trials_max_min = compute_rejected_trials_max_min(
        eog_epo, max_min=max_min)
    good_trials = np.delete(good_trials, rejected_trials_max_min) # delete deletes indices
    variances = np.var(epo.data[good_trials], axis=1)
    rejected_chan_inds, rejected_trials_var = compute_rejected_channels_trials_by_variance(
        variances, whisker_percent, whisker_length)
    rejected_var_original = [good_trials[i] for i in rejected_trials_var]
    good_trials = np.delete(good_trials, rejected_trials_var) # delete deletes indices
    rejected_trials = np.setdiff1d(orig_trials, good_trials)
    rejected_chan_names = epo.axes[2][rejected_chan_inds]
    return (rejected_chan_names, rejected_trials, rejected_trials_max_min,
        rejected_var_original, good_trials)

def compute_rejected_trials_max_min_eog(filename, max_min, ival=[0,4000]):
    eog_set = BBCIDataset(filename=filename, 
                        sensor_names=['EOGh', 'EOGv'], cnt_preprocessors=[],
                        epo_preprocessors=[],
                        segment_ival=ival)
    eog_set.load()
    return compute_rejected_trials_max_min(eog_set.epo, max_min)

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

def compute_rejected_channels_trials_variance_crit(bbci_set, whisker_percent, whisker_length):
    variances = np.var(bbci_set.epo.data, axis=1)
    rejected_chan_inds, rejected_trials = compute_rejected_channels_trials_by_variance(
        variances, whisker_percent, whisker_length)
    rejected_chan_names = np.array(bbci_set.sensor_names)[rejected_chan_inds]
    return rejected_chan_names, rejected_trials

# test create set three trials, one trial has excessive variance, should be removed 
# create set with three channels, one excessive variance, should be removed
# create set 
def compute_rejected_channels_trials_by_variance(variances, whisker_percent,
    whisker_length):
    orig_chan_inds = range(variances.shape[1])
    orig_trials = range(variances.shape[0])
    good_chan_inds = np.copy(orig_chan_inds)
    good_trials = np.copy(orig_trials)

    # remove trials with excessive variances
    bad_trials = compute_excessive_outlier_trials(variances, whisker_percent, whisker_length)
    good_trials = np.delete(good_trials, bad_trials, axis=0)
    variances = np.delete(variances, bad_trials, axis=0)

    # now remove channels (first)
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
    bad_chans = compute_unstable_chans(variances, whisker_percent, whisker_length)
    variances = np.delete(variances, bad_chans, axis=1)
    good_chan_inds = np.delete(good_chan_inds, bad_chans, axis=0)
    
    rejected_chan_inds = np.setdiff1d(orig_chan_inds, good_chan_inds)
    rejected_trials = np.setdiff1d(orig_trials, good_trials)
    return rejected_chan_inds, rejected_trials

def compute_rejected_trials_cnt(cnt, filename, rejection_blink_ival,
        max_min, rejection_var_ival, whisker_percent, whisker_length,
        low_cut_hz, high_cut_hz,filt_order, preremoved_chans):
    """ Only reject trials, before remove the `preremoved chans`.
    Useful for example if you want to remove same chans as in another set."""
    # First create eog set for blink rejection
    eog_sensor_names = BBCIDataset.get_all_sensors(filename, pattern="EO")
    eog_set = BBCIDataset(filename=filename, 
                    sensor_names=eog_sensor_names, cnt_preprocessors=[],
                    epo_preprocessors=[],
                    segment_ival=rejection_blink_ival)
    eog_set.load_continuous_signal()
    eog_set.add_markers()
    eog_set.segment_into_trials()
    eog_set.remove_continuous_signal()
    
    cnt = select_channels(cnt, preremoved_chans, invert=True)
    
    # Then create bandpassed set for variance rejection
    bandpassed_cnt = bandpass_cnt(cnt, low_cut_hz, high_cut_hz, filt_order)
    epo = segment_dat_fast(bandpassed_cnt, 
           marker_def={'1 - Right Hand': [1], '2 - Left Hand': [2], 
               '3 - Rest': [3], '4 - Feet': [4]}, 
           ival=rejection_var_ival)
    return compute_only_rejected_trials(epo, 
            eog_set.epo, max_min=max_min, 
            whisker_percent=whisker_percent, 
            whisker_length=whisker_length)
    
def compute_only_rejected_trials(epo, eog_epo, max_min,
        whisker_percent, whisker_length):
    print ("epo shape", epo.data.shape)
    orig_trials = range(epo.data.shape[0])
    good_trials = range(epo.data.shape[0])
    rejected_trials_max_min = compute_rejected_trials_max_min(
        eog_epo, max_min=max_min)
    good_trials = np.delete(good_trials, rejected_trials_max_min) # delete deletes indices
    
    print ("good_trials", good_trials)
    variances = np.var(epo.data[good_trials], axis=1)
    rejected_trials_var = compute_only_rejected_trials_by_variance(
        variances, whisker_percent, whisker_length)
    rejected_var_original = [good_trials[i] for i in rejected_trials_var]
    # delete deletes indices, thats why we are using rejected_trials_var
    # and not rejected_var_original
    good_trials = np.delete(good_trials, rejected_trials_var) 
    rejected_trials = np.setdiff1d(orig_trials, good_trials)
    return (rejected_trials, rejected_trials_max_min,
        rejected_var_original, good_trials)

def compute_only_rejected_trials_by_variance(variances, whisker_percent,
    whisker_length):
    """ Only reject trials, keep all chans.
    Useful in case chans already removed, e.g., by an earlier cleaning or
    to try to remove same chans as in another set."""
    orig_trials = range(variances.shape[0])
    good_trials = np.copy(orig_trials)

    # remove trials with excessive variances
    bad_trials = compute_excessive_outlier_trials(variances, whisker_percent, whisker_length)
    print ("bad outlier trials", bad_trials)
    good_trials = np.delete(good_trials, bad_trials, axis=0)
    variances = np.delete(variances, bad_trials, axis=0)

    # now remove trials 
    no_further_rejections = False
    while not no_further_rejections:
        bad_trials = compute_outlier_trials(variances, whisker_percent, whisker_length)
        good_trials = np.delete(good_trials, bad_trials, axis=0)
        variances = np.delete(variances, bad_trials, axis=0)
        no_further_rejections = len(bad_trials) == 0
        print ("good trials", good_trials)

    
    rejected_trials = np.setdiff1d(orig_trials, good_trials)
    return rejected_trials

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