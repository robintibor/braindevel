import numpy as np
import random
import scipy.stats
from braindecode.results.results import (extract_combined_results,
    get_final_misclasses,
    get_training_times, extract_single_group_result_sorted)
import datetime

def perm_mean_diffs_sampled(a, b, n_diffs=None):
    """Compute differences between all permutations of  labels.
    Version that samples.
    Parameters
    --------------
    a: list or numpy array
    b: list or numpy array
    n_diffs: int
        How many diffs/samples to compute.
    Returns
    -------
    all_diffs: 1d-array of float
        Sampled mean differences.
    """
    
    n_exps = len(a)
    all_bit_masks = [2 ** n for n in xrange(n_exps-1,-1,-1)]
    if n_diffs is None:
        n_diffs = 2**n_exps
        i_all_masks = xrange(n_diffs)
    else:
        random.seed(39483948)
        # take samples of all masks, always add identity mask
        i_all_masks = random.sample(xrange(0,2**n_exps-1), n_diffs - 1)
        i_all_masks = i_all_masks + [(2**n_exps)-1]
        # verification this is actually identity mask for code below:
        test_i_mask = i_all_masks[-1]
        test_mask = (np.bitwise_and(test_i_mask, all_bit_masks) > 0) * 2 - 1
        assert np.array_equal(a - b, (test_mask * a)  -test_mask * b)

        
    all_diffs = np.float32(np.ones(n_diffs) * np.nan)
    for i_diff, i_mask in enumerate(i_all_masks):
        # masks has -1s and 1s,
        # 1 interpretable as
        # correct value selected
        # -1 as randomly flipped value/"incorrect" value selected
        # *2 makes values between 2 and 0, then -1 to make 
        # values between 1 and -1
        mask = (np.bitwise_and(i_mask, all_bit_masks) > 0) * 2 - 1
        # mean later by dividing by n_exp
        # seems to be a little bit faster that way
        diff = np.sum((mask * a)  -mask * b)
        all_diffs[i_diff] = diff
    all_diffs = all_diffs / float(n_exps)
    return all_diffs

def perm_mean_diffs(a,b):
    """Compute differences between all permutations of  labels.
    Assumes a and b are paired values,
    a are values with label 0 and b with label 1.
    Computes mean differences for all possible   
    switches of 0 and 1 (but keeping pairs together, i.e.
    2 ^ len(a) switches).
    
    Parameters
    --------------
    a: list or numpy array
    b: list or numpy array
    
    Returns
    -------
    diffs: 1d-numpy array
        Differences between means of labelled values
        for all label-switched values.
        
    Notes
    -----
    http://www.stat.ncsu.edu/people/lu/courses/ST505/Ch4.pdf#page=10
    http://stats.stackexchange.com/a/64215/56289
    http://www.jarrodmillman.com/publications/millman2015thesis.pdf ->
    https://github.com/statlab/permute python package 
    (probably, didnt read: http://finzi.psych.upenn.edu/R/library/EnvStats/html/twoSamplePermutationTestLocation.html)
    """
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    n_exps = len(a)
    all_masks = _create_masks(n_exps)
    diffs = _compute_diffs(a, b, all_masks)
    return diffs

def perm_mean_diff_test(a,b, n_diffs=None):
    """Return two sided p-value of perm mean diff."""
    if n_diffs is None:
        diffs = perm_mean_diffs(a, b)
    else:
        diffs = perm_mean_diffs_sampled(a, b, n_diffs)
    
    actual_diff = np.mean(a - b)
    n_samples_as_large_diff = np.sum(np.abs(diffs) >= np.abs(actual_diff))
    #if n_diffs is not None:
    #    p_val = n_samples_as_large_diff + 1 /
    return n_samples_as_large_diff / float(len(diffs))


def _create_masks(n_exps):
    """ Create all (2^n_exps) binary selection masks for this number of experiments.
    E.g. for 3 experiments:
    False, False, False
    False, False, True
    False, True, False
    False, True, True
    True, False, False
    True, False, True
    True, True, False
    True, True, True""" 
    all_masks = np.array([[False] * n_exps] * (2 ** n_exps))
    i_block_size = all_masks.shape[0] / 2 
    for i_col in xrange(0,all_masks.shape[1]):
        for i_row in xrange(0,all_masks.shape[0], i_block_size * 2):
            all_masks[i_row:i_row+i_block_size,i_col] = [[True]] * i_block_size
        i_block_size /= 2
    return all_masks
    
def _compute_diffs(a, b, all_masks):
    # first add "first set" part
    # positive labels from a
    # and negative labels from b
    diffs = all_masks * a
    diffs += (1 - all_masks) * b
    # subtract "second set" part
    # negative labels from a
    # positive labels from b
    diffs -= (1 - all_masks) * a
    diffs -= all_masks * b
    return np.mean(diffs, axis=1)

def print_stats(results, csp_results, n_diffs=None):
    res_misclasses = get_final_misclasses(results)
    csp_misclasses = get_final_misclasses(csp_results)
    res_times = get_training_times(results)
    csp_times = get_training_times(csp_results)
    # make a the smaller misclass, b the larger misclass
    if np.mean(res_misclasses) < np.mean(csp_misclasses):
        a = res_misclasses
        b = csp_misclasses
    else:
        a = csp_misclasses
        b = res_misclasses


    actual_diff = np.mean(a - b)

    if n_diffs is None:
        diffs = perm_mean_diffs(a, b)
    else:
        diffs = perm_mean_diffs_sampled(a,b,n_diffs=n_diffs)
    res_to_csp_diff = np.mean(res_misclasses - csp_misclasses)

    print ("deep accuracy:    {:.1f}".format( 100 * (1 - np.mean(res_misclasses))))
    print ("csp  accuracy:    {:.1f}".format( 100 * (1 - np.mean(csp_misclasses))))
    print ("diff accuracy:    {:.1f}".format( 100 * -res_to_csp_diff))
    print ("std          :    {:.1f}".format( 100 * np.std(res_misclasses - 
        csp_misclasses)))
    
    print("one sided perm     {:.5f}".format(np.sum(diffs <= actual_diff) 
        / float(len(diffs))))
    print("one sided wilcoxon {:.5f}".format(scipy.stats.wilcoxon(
        res_misclasses, csp_misclasses)[1] / 2))
    #print("two sided perm     {:.5f}".format(np.sum(
    #    abs(diffs) >= abs(actual_diff)) / float(len(diffs))))
    #print("two sided wilcoxon {:.5f}".format(scipy.stats.wilcoxon(
    #    res_misclasses, csp_misclasses)[1]))
    print ("deep time:        {:s}".format(str(datetime.timedelta(
                seconds=round(np.mean(res_times))))))
    print ("csp time:         {:s}".format(str(datetime.timedelta(
                seconds=round(np.mean(csp_times))))))
    #print ("deep time std:    {:s}".format(str(datetime.timedelta(
    #            seconds=round(np.std(res_times))))))
    #print ("csp time std:     {:s}".format(str(datetime.timedelta(
    #            seconds=round(np.std(csp_times))))))

def show_stats_for_combined_results(folder, params, folder_2,  params_2, 
        combined_csp_results):
    combined_results = extract_combined_results(folder, params, 
        folder_2, params_2)
    print_stats(combined_results, combined_csp_results, n_diffs=2**18)

def show_stats_for_result(folder, params, combined_csp_results, n_diffs=2**18):
    res = extract_single_group_result_sorted(folder, params=params)
    print_stats(res, combined_csp_results,  n_diffs=n_diffs)

def count_signrank(k,n):
    """k is the test statistic, n is the number of samples."""
    # ported from here:
    # https://github.com/wch/r-source/blob/e5b21d0397c607883ff25cca379687b86933d730/src/nmath/signrank.c#L84
    u = n * (n + 1) / 2
    c = (u / 2)
    w = np.zeros(c+1)
    if (k < 0 and k > u):
        return 0
    if (k > c):
        k = u - k
    if (n == 1):
        return 1.
    if (w[0] == 1.):
        return w[k]
    w[0] = w[1] = 1.
    for j in range(2,n+1):
        end = min(j*(j+1)//2, c)
        for i in range(end, j-1,-1):
            w[i] += w[i-j]
    return w[k]

def wilcoxon_signed_rank(a,b):
    """ See http://www.jstor.org/stable/pdf/3001968.pdf?_=1462738643716
    https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    Has been validated against R wilcox.test exact variant (with no ties 
    atleast), e.g.:
      wilcox.test(c(0,0,0,0,0,0,0,0,0,0,0), 
            c(1,2,3,-4,5,6,7,8,-9,10,11), 
            paired=TRUE,
           exact=TRUE)
    Ties are handled by using average rank
    Zeros are handled by assigning half of ranksum to 
    positive and half to negative
    ->
    p-value = 0.08301"""
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    n_samples = len(a)

    diff = a - b
    ranks = scipy.stats.rankdata(np.abs(diff), method='average')
    # unnecessary could have simply used diff in formulas below
    # also...
    signs = np.sign(diff)

    negative_rank_sum = np.sum(ranks * (signs < 0))
    positive_rank_sum = np.sum(ranks * (signs > 0))
    equal_rank_sum = np.sum(ranks * (signs == 0))

    test_statistic = min(negative_rank_sum, positive_rank_sum)
    # add equals half to both sides... so just add half now
    # after taking minimum, reuslts in the same
    test_statistic += equal_rank_sum / 2.0
    # make it more conservative by taking the ceil
    test_statistic = int(np.ceil(test_statistic))
    
    # apparently I start sum with 1
    # as count_signrank(0,n) is always 1
    # independent of n
    # so below is equivalent to
    # n_as_extreme_sums = 0
    # and using range(0, test-statistic+1)
    n_as_extreme_sums = 1
    for other_sum in range(1,test_statistic+1):
        n_as_extreme_sums += count_signrank(other_sum, n_samples)
    # I guess 2* for twosided test?
    # same as scipy does
    p_val = (2 * n_as_extreme_sums) / (2**float(n_samples))
    return p_val

def sign_test(a,b): 
    # Should be same as https://onlinecourses.science.psu.edu/stat464/node/49
    
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    n_samples = len(a)
    diffs = a - b
    n_positive = np.sum(diffs > 0)
    n_equal = np.sum(diffs == 0)
    # adding half of equal to positive (so implicitly
    # other half is added to negative)otal
    n_total = n_positive + (n_equal / 2)
    # rounding conservatively
    if n_total < (n_samples / 2):
        n_total = int(np.ceil(n_total))
    else:
        n_total = int(np.floor(n_total))
    
    return scipy.stats.binom_test(n_total, n_samples, p=0.5)

def median(a, axis=None, keepdims=False):
    """
    Just since I use old numpy version on one cluster which doesn't
    have keepdims
    """
    out = np.median(a, axis)
    if keepdims:
        for ax in axis:
            out = np.expand_dims(out, ax)
    return out
    
def median_absolute_deviation(arr, axis=None, keepdims=False):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variability of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
        http://stackoverflow.com/a/23535934/1469195
    """
    arr = np.array(arr)
    if axis is None:
        axis = range(arr.ndim)
    med = median(arr, axis=axis, keepdims=True)
    return median(np.abs(arr - med), axis=axis, keepdims=keepdims)

def corr(x,y):
    """
    Assumes x and y are features x samples
    """
    # Difference to numpy:
    # Correlation only between terms of x and y
    # not between x and x or y and y
    this_cov = cov(x,y)
    return cov_to_corr(this_cov,x,y)

def cov_to_corr(this_cov,x,y):
    # computing "unbiased" corr
    # ddof=1 for unbiased..
    var_x = np.var(x, axis=1, ddof=1)
    var_y = np.var(y, axis=1, ddof=1)
    return cov_and_var_to_corr(this_cov, var_x, var_y)
    
def cov_and_var_to_corr(this_cov, var_x, var_y):
    divisor = np.outer(np.sqrt(var_x), np.sqrt(var_y))
    return this_cov / divisor

def cov(x,y):
    # Difference to numpy:
    # Covariance only between terms of x and y
    # not between x and x or y and y
    demeaned_x = x - np.mean(x, axis=1, keepdims=True)
    demeaned_y = y - np.mean(y, axis=1, keepdims=True)
    this_cov = np.dot(demeaned_x,demeaned_y.T) / (y.shape[1] -1)
    return this_cov
    
def wrap_reshape_topo(stat_fn, topo_a, topo_b, axis_a, axis_b):
    other_axis_a = [i for i in xrange(topo_a.ndim) if i not in axis_a]
    other_axis_b = [i for i in xrange(topo_b.ndim) if i not in axis_b]
    transposed_topo_a = topo_a.transpose(tuple(other_axis_a) + tuple(axis_a))
    n_stat_axis_a = [topo_a.shape[i] for i in axis_a]
    n_other_axis_a = [topo_a.shape[i] for i in other_axis_a]
    flat_topo_a = transposed_topo_a.reshape(np.prod(n_other_axis_a), np.prod(n_stat_axis_a))
    transposed_topo_b = topo_b.transpose(tuple(other_axis_b) + tuple(axis_b))
    n_stat_axis_b = [topo_b.shape[i] for i in axis_b]
    n_other_axis_b = [topo_b.shape[i] for i in other_axis_b]
    flat_topo_b = transposed_topo_b.reshape(np.prod(n_other_axis_b), np.prod(n_stat_axis_b))
    assert np.array_equal(n_stat_axis_a, n_stat_axis_b)
    stat_result = stat_fn(flat_topo_a, flat_topo_b)
    topo_result = stat_result.reshape(tuple(n_other_axis_a) + tuple(n_other_axis_b))
    return topo_result
    
def running_mean(arr, window_len, axis=0):
    # adapted from http://stackoverflow.com/a/27681394/1469195
    # need to pad to get correct first value also
    arr_padded = np.insert(arr,0,values=0,axis=axis)
    cumsum = np.cumsum(arr_padded,axis=axis)
    later_sums = np.take(cumsum, xrange(window_len, arr_padded.shape[axis]), 
        axis=axis)
    earlier_sums = np.take(cumsum, xrange(0, arr_padded.shape[axis] - window_len), 
        axis=axis)
    

    moving_average = (later_sums - earlier_sums) / float(window_len)
    return moving_average

