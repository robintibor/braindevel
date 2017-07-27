import numpy as np
from copy import deepcopy

def exponential_running_standardize(data, factor_new, init_block_size=None,
                                    start_mean=None, start_var=None, axis=None,
                                    eps=1e-4):
    assert ((init_block_size is None and (start_mean is not None) and
             (start_var is not None)) or
            ((init_block_size is not None) and
             start_mean is None and start_var is None)), ("Supply either "
                                                          "init block size or start values...")
    demeaned = exponential_running_demean(data, factor_new=factor_new,
                                          init_block_size=init_block_size,
                                          start_mean=start_mean, axis=axis)
    stds = np.sqrt(exponential_running_var_from_demeaned(
        demeaned, factor_new=factor_new, init_block_size=init_block_size,
        start_var=start_var, axis=axis))
    standardized_data = demeaned / np.maximum(stds, eps)
    return standardized_data


def exponential_running_demean(data, factor_new, init_block_size=None,
                               start_mean=None, axis=None):
    assert ((init_block_size is None and (start_mean is not None)) or
            ((init_block_size is not None) and start_mean is None)), (
    "Supply either "
    "init block size or start values...")
    means = exponential_running_mean(data, factor_new=factor_new,
                                     init_block_size=init_block_size,
                                     start_mean=start_mean, axis=axis)
    demeaned = data - means
    return demeaned


def exponential_running_mean(data, factor_new, init_block_size=None,
                             start_mean=None, axis=None):
    """ Compute the running mean across axis 0.
    For each datapoint in axis 0 its "running exponential mean" is computed as:
    Its mean * factor_new + so far computed mean * (1-factor_new).
    You can either specify a start mean or an init_block_size to 
    compute the start mean of. 
    In any case one mean per datapoint in axis 0 is returned.
    If axis is None, no mean is computed per datapoint but datapoint
    is simply used as is."""
    assert not (start_mean is None and init_block_size is None), (
        "Need either an init block or a start mean")
    assert start_mean is None or init_block_size is None, (
    "Can only use start mean "
    "or init block size")
    assert factor_new <= 1.0
    assert factor_new >= 0.0
    if isinstance(axis, int):
        axis = (axis,)
    factor_old = 1 - factor_new

    # first preallocate the shape for the running means
    # shape depends on which axes will be removed
    running_mean_shape = list(data.shape)
    if axis is not None:
        for ax in axis:
            # keep dim as empty dim
            running_mean_shape[ax] = 1

    running_means = (np.ones(running_mean_shape) * np.nan).astype(np.float32)

    if start_mean is None:
        start_data = data[0:init_block_size]
        if axis is not None:
            axes_for_start_mean = (0,) + axis  # also average across init trials
        else:
            axes_for_start_mean = 0
        # possibly temporarily upcast to float32 to avoid overflows in sum
        # that is computed to compute mean
        current_mean = np.mean(start_data.astype(np.float32),
                               keepdims=True,
                               axis=axes_for_start_mean).astype(
            start_data.dtype)
        # repeat mean for running means
        running_means[:init_block_size] = current_mean
        i_start = init_block_size
    else:
        current_mean = start_mean
        i_start = 0

    for i in range(i_start, len(data)):
        if axis is not None:
            datapoint_mean = np.mean(data[i:i + 1], axis=axis, keepdims=True)
        else:
            datapoint_mean = data[i:i + 1]
        next_mean = factor_new * datapoint_mean + factor_old * current_mean
        running_means[i] = next_mean
        current_mean = next_mean

    assert not np.any(np.isnan(running_means)), (
        "RUnning mean has NaNs :\n{:s}".format(str(running_means)))
    assert not np.any(np.isinf(running_means)), (
        "RUnning mean has Infs :\n{:s}".format(str(running_means)))
    return running_means


def exponential_running_var_from_demeaned(demeaned_data, factor_new,
                                          start_var=None,
                                          init_block_size=None, axis=None):
    """ Compute the running var across axis 0 + given axis from demeaned data.
    For each datapoint in axis 0 its "running exponential var" is computed as:
    Its (datapoint)**2 * factor_new + so far computed var * (1-factor_new).
    You can either specify a start var or an initial block size to 
    compute the start var of. 
    In any case one var per datapoint in axis 0 is returned.
    If axis is None, no mean is computed but trial is simply used as is."""
    # TODELAY: split out if and else case into different functions
    # i.e. split apart a common function having a start value (basically the loop)
    # and then split if and else into different functions
    factor_old = 1 - factor_new
    # first preallocate the shape for the running vars for performance (otherwise much slower)
    # shape depends on which axes will be removed
    running_vars_shape = list(demeaned_data.shape)
    if axis is not None:
        for ax in axis:
            running_vars_shape.pop(ax)
    running_vars = (np.ones(running_vars_shape) * np.nan).astype(np.float32)

    if start_var is None:
        if axis is not None:
            axes_for_start_var = (0,) + axis  # also average across init trials
        else:
            axes_for_start_var = 0

        # possibly temporarily upcast to float32 to avoid overflows in sum
        # that is computed to compute mean
        start_running_var = np.mean(
            np.square(demeaned_data[0:init_block_size].astype(np.float32)),
            axis=axes_for_start_var, keepdims=True).astype(demeaned_data.dtype)
        running_vars[0:init_block_size] = start_running_var
        current_var = start_running_var
        start_i = init_block_size
    else:
        current_var = start_var
        start_i = 0

    for i in range(start_i, len(demeaned_data)):
        squared = np.square(demeaned_data[i:i + 1])
        if axis is not None:
            this_var = np.mean(squared, axis=axis, keepdims=True)
        else:
            this_var = squared
        next_var = factor_new * this_var + factor_old * current_var
        running_vars[i] = next_var
        current_var = next_var
    assert not np.any(np.isnan(running_vars))
    return running_vars


def compute_combined_mean(num_old, num_new, old_mean, new_mean):
    # formula according to http://stats.stackexchange.com/a/43183
    return (num_old * old_mean + num_new * new_mean) / \
           (num_old + num_new)


def compute_combined_std(num_old, num_new, old_mean, new_mean,
                         combined_mean, old_std, new_std):
    # formulas according to http://stats.stackexchange.com/a/43183
    combined_var = ((num_old * (old_std ** 2 + old_mean ** 2) + \
                     num_new * (new_std ** 2 + new_mean ** 2)) / \
                    (num_old + num_new)) - \
                   combined_mean ** 2
    combined_std = np.sqrt(combined_var)
    return combined_std


# adapted from https://subluminal.wordpress.com/2008/07/31/running-standard-deviations/
# using degree of freedom 0 which is default for numpy std
# and also simplifies calculations even more
def online_standardize(topo, n_old_trials, old_mean, old_std, dims_to_squash=None,
    std_eps=1e-5):
    n = n_old_trials
    mean = old_mean
    power_avg = (old_std * old_std + old_mean * old_mean)
    if dims_to_squash is not None:
        # have to subtract one since batch dimension is gone inside loop
        dims_to_squash = tuple(np.array(dims_to_squash) - 1)
    new_topo = deepcopy(topo)
    for i_trial in xrange(len(topo)):
        n += 1
        this_topo = topo[i_trial]
        if dims_to_squash is not None:
            this_topo = np.mean(this_topo, axis=dims_to_squash, keepdims=True)
        mean = mean + ((this_topo - mean) / n)
        power_avg = power_avg + ((this_topo * this_topo - power_avg) / n)
        std = np.sqrt((power_avg - mean * mean))
        new_topo[i_trial] = ((topo[i_trial] - mean) / (std_eps + std))
    return new_topo