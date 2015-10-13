import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.signal import hilbert
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import pearsonr

class CorrCoeffIntervalOptimizer(object):
    def __init__(self, max_score_fraction=0.8,
            use_abs_for_threshold=True):
        self.use_abs_for_threshold = use_abs_for_threshold
        self.max_score_fraction = max_score_fraction

    def optimize(self, epo):
        return optimize_segment_ival(epo, 
            max_score_fraction=self.max_score_fraction,
            use_abs_for_threshold=self.use_abs_for_threshold,
            mode="corrcoeff")

class AucIntervalOptimizer(object):
    def __init__(self, max_score_fraction=0.8,
            use_abs_for_threshold=True):
        self.use_abs_for_threshold = use_abs_for_threshold
        self.max_score_fraction = max_score_fraction

    def optimize(self, epo):
        return optimize_segment_ival(epo, 
            max_score_fraction=self.max_score_fraction,
            use_abs_for_threshold=self.use_abs_for_threshold,
            mode="auc")

def optimize_segment_ival(epo, max_score_fraction=0.8,
        use_abs_for_threshold=True, mode="auc"):
    """ Optimizing segment ival following http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=4408441#app3
    (but using auc instead of corrcoef)"""
    epo_envelope = np.abs(hilbert(epo.data, axis=1))
    epo_smoothed = gaussian_filter(epo_envelope, (0,15,0), order=0, mode='reflect')
    labels = epo.axes[0]
    # labels should be 0,1 for auc but they may be 1,3 or anything else..
    # so convert them to 0/1
    assert len(np.unique(labels)) == 2
    binary_labels = np.int32(labels == np.max(labels))
    # Create blocks of 100 ms length (divided by 10 is same as *100(ms)/1000(ms))
    assert epo.fs % 10 == 0
    n_samples_per_block = epo.fs / 10
    n_samples = len(epo.axes[1])
    assert n_samples % n_samples_per_block == 0
    n_time_blocks = n_samples // n_samples_per_block
    n_chans = len(epo.axes[2])
    
    
    auc_scores = np.ones((n_time_blocks, n_chans)) * np.nan
    
    for i_time_block in range(n_time_blocks):
        for i_chan in range(n_chans):
            start_sample = i_time_block * n_samples_per_block
            epo_part = epo_smoothed[:,start_sample: start_sample + n_samples_per_block,i_chan]
            
            # auc values indicate good separability if they are close to 0 or close to 1
            # subtracting 0.5 transforms them to mean better separability more far away from 0
            # this makes later computations easier
            if mode =='auc':
                score = roc_auc_score(binary_labels, np.sum(epo_part, axis=(1))) - 0.5
            else:
                assert mode == 'corrcoeff'
                score = pearsonr(binary_labels, np.sum(epo_part, axis=(1)))[0]
            auc_scores[i_time_block, i_chan] = score
            
    
    auc_score_chan = np.sum(np.abs(auc_scores), axis=1)
    
    # sort time ivals so that best ival across chans is first
    time_blocks_sorted = np.argsort(auc_score_chan)[::-1]
    i_best_block = time_blocks_sorted[0]
    
    
    chan_above_zero = auc_scores[i_best_block, :] > 0
    chan_sign = 1 * chan_above_zero + -1 * np.logical_not(chan_above_zero)
    
    sign_adapted_scores = auc_scores * chan_sign
    
    chan_meaned_scores = np.sum(sign_adapted_scores, axis=1)
    best_meaned_block = np.argsort(chan_meaned_scores)[::-1][0]
    if use_abs_for_threshold:
        threshold = (np.sum(chan_meaned_scores[chan_meaned_scores > 0]) * 
            max_score_fraction)
    else:
        threshold = np.sum(chan_meaned_scores) * max_score_fraction
    
    t0 = best_meaned_block
    t1 = best_meaned_block
    # stop if either above threshold or
    # there are no timeblocks with positive values left to add
    # (this also implies stopping if both indices are at the borders)
    while (np.sum(chan_meaned_scores[t0:t1+1]) < threshold and 
            (np.sum(chan_meaned_scores[:t0] * (chan_meaned_scores[:t0] > 0) 
                +
            np.sum(chan_meaned_scores[t1+1:] * (chan_meaned_scores[t1+1:]> 0))) 
            > 0)):
        if ((np.sum(chan_meaned_scores[:t0]) > np.sum(chan_meaned_scores[t1+1:])
            and t0 > 0)
            or t1 == n_time_blocks - 1):
            t0 = t0 - 1
        else:
            t1 = t1 + 1
    
    start_sample = t0 * n_samples_per_block
    end_sample = (t1 + 1) * n_samples_per_block
    start_ms = start_sample * 1000.0 / epo.fs 
    end_ms = end_sample * 1000.0 / epo.fs
    return start_ms, end_ms