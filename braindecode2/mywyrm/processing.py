import re
from copy import deepcopy
import logging

import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal
import scipy as sp
from sklearn.covariance import LedoitWolf as LW
import resampy

from braindecode2.csp.generate_filterbank import filter_is_stable
from braindecode2.util import deepcopy_xarr

log = logging.getLogger(__name__)


def concatenate_cnt(cnt1, cnt2):
    """
    Concatenate two continous datasets, shift times of second set.
    Times of second set are shifted to start one time-sampling step
    after the end of the first set.
    Events are shifted accordingly as well.
    Sampling rates must be identical for both sets.
    Parameters
    ----------
    cnt1
    cnt2

    Returns
    -------
    concatenated: DateArray
        DateArray with second set appended to first set.

    """
    assert cnt1.attrs['fs'] == cnt2.attrs['fs']
    assert np.array_equal(cnt1.channels, cnt2.channels)
    cnt1 = deepcopy_xarr(cnt1)
    cnt2 = deepcopy_xarr(cnt2)

    # make next series appear exactly one time step after old series
    time_offset = -cnt2.time.data[0] + cnt1.time.data[-1] + 1000.0 / cnt2.fs

    cnt2.time.data += time_offset
    concatenated = xr.concat((cnt1, cnt2), dim='time')
    cnt2.attrs['events'][:, 0] += len(cnt1.data)
    concat_events = np.concatenate((cnt1.attrs['events'],
                                    cnt2.attrs['events']))
    concatenated.attrs['events'] = concat_events
    return concatenated


def select_classes(epo, class_inds):
    """Select classes from an epoched data object.

    This method selects the classes with the specified indices.

    Parameters
    ----------
    dat : Data
        epoched Data object
    indices : array of ints
        The indices of the classes to select.
    invert : Boolean, optional
        if true keep all classes except the ones defined by ``indices``.
    classaxis : int, optional
        the axis along which the classes are selected

    Returns
    -------
    dat : Data
        a copy of the epoched data with only the selected classes
        included.

    Raises
    ------
    AssertionError
        if ``dat`` has no ``.class_names`` attribute.

    See Also
    --------
    remove_classes

    Examples
    --------

    Get the classes 1 and 2.

    >>> dat.axes[0]
    [0, 0, 1, 2, 2]
    >>> dat = select_classes(dat, [1, 2])
    >>> dat.axes[0]
    [1, 2, 2]

    Remove class 2

    >>> dat.axes[0]
    [0, 0, 1, 2, 2]
    >>> dat = select_classes(dat, [2], invert=True)
    >>> dat.axes[0]
    [0, 0, 1]

    """

    trial_inds = np.flatnonzero(np.in1d(epo.trials.data, class_inds))
    epo = epo.isel(trials=trial_inds)
    return epo


def select_channels(dat, regexp_list, invert=False):
    """Select channels from data.

    The matching is case-insensitive and locale-aware (as in
    ``re.IGNORECASE`` and ``re.LOCALE``). The regular expression always
    has to match the whole channel name string

    Parameters
    ----------
    dat : Data
    regexp_list : list of regular expressions
        The regular expressions provided, are used directly by Python's
        :mod:`re` module, so all regular expressions which are understood
        by this module are allowed.

        Internally the :func:`re.match` method is used, additionally to
        check for a match (which also matches substrings), it is also
        checked if the whole string matched the pattern.
    invert : Boolean, optional
        If True the selection is inverted. Instead of selecting specific
        channels, you are removing the channels. (default: False)
    chanaxis : int, optional
        the index of the channel axis in ``dat`` (default: -1)

    Returns
    -------
    dat : Data
        A copy of ``dat`` with the channels, matched by the list of
        regular expressions.

    Examples
    --------
    Select all channels Matching 'af.*' or 'fc.*'

    >>> dat_new = select_channels(dat, ['af.*', 'fc.*'])

    Remove all channels Matching 'emg.*' or 'eog.*'

    >>> dat_new = select_channels(dat, ['emg.*', 'eog.*'], invert=True)

    Even if you only provide one Regular expression, it has to be in an
    array:

    >>> dat_new = select_channels(dat, ['af.*'])

    See Also
    --------
    remove_channels : Remove Channels
    re : Python's Regular Expression module for more information about
        regular expressions.

    """
    dat = deepcopy_xarr(dat)
    all_channels = dat.channels.data
    matched_channels = []
    for c in all_channels:
        for regexp in regexp_list:
            matched = re.match(regexp, c, re.IGNORECASE | re.LOCALE)
            if matched and matched.group() == c:
                matched_channels.append(c)
                # no need to look any further for matches for this channel
                break
    if not invert:
        wanted_channels = [c for c in all_channels if c in matched_channels]
    else:
        wanted_channels = [c for c in all_channels if c not in matched_channels]

    return dat.sel(channels=wanted_channels)


def resample_cnt(cnt, new_fs):
    cnt = deepcopy_xarr(cnt)
    if new_fs == cnt.attrs['fs']:
        log.info(
            "Just copying data, no resampling, since new sampling rate same.")
        return deepcopy_xarr(cnt)
    log.warn("This is not causal, uses future data....")
    log.info("Resampling from {:f} to {:f} Hz.".format(
        cnt.attrs['fs'], new_fs
    ))

    time_axis = list(cnt.dims).index('time')
    new_data = resampy.resample(cnt.data, cnt.attrs['fs'],
                                new_fs, axis=time_axis, filter='kaiser_fast')
    # take times starting backwards to take rather later timestamps
    # so any offset is towards the back not towards the start
    inds = np.linspace(len(cnt.time.data),
                       0, new_data.shape[time_axis], endpoint=False)[::-1]
    new_times = np.interp(inds, np.arange(len(cnt.time.data)), cnt.time.data)
    cnt = xr.DataArray(new_data,
                       coords={'channels': cnt.channels, 'time': new_times},
                       dims=cnt.dims,
                       attrs=cnt.attrs)
    old_fs = cnt.attrs['fs']
    event_samples_old = cnt.attrs['events'][:,0]
    event_samples = event_samples_old * new_fs / float(old_fs)
    event_samples = np.uint32(np.ceil(event_samples))
    cnt.attrs['events'][:,0] = event_samples
    cnt.attrs['fs'] = new_fs
    return cnt


def lfilter(dat, b, a, zi=None):
    """Filter data using the filter defined by the filter coefficients.

    This method mainly delegates the call to
    :func:`scipy.signal.lfilter`.

    Parameters
    ----------
    dat : Data
        the data to be filtered
    b : 1-d array
        the numerator coefficient vector
    a : 1-d array
        the denominator coefficient vector
    zi : nd array, optional
        the initial conditions for the filter delay. If zi is ``None``
        or not given, initial rest is assumed.
    timeaxis : int, optional
        the axes in ``data`` to filter along to

    Returns
    -------
    dat : Data
        the filtered output

    See Also
    --------
    :func:`lfilter_zi`, :func:`filtfilt`, :func:`scipy.signal.lfilter`,
    :func:`scipy.signal.butter`, :func:`scipy.signal.butterord`

    Examples
    --------

    Generate and use a Butterworth bandpass filter for complete
    (off-line data):

    >>> # the sampling frequency of our data in Hz
    >>> dat.fs
    100
    >>> # calculate the nyquist frequency
    >>> fn = dat.fs / 2
    >>> # the desired low and high frequencies in Hz
    >>> f_low, f_high = 2, 13
    >>> # the order of the filter
    >>> butter_ord = 4
    >>> # calculate the filter coefficients
    >>> b, a = signal.butter(butter_ord, [f_low / fn, f_high / fn], btype='band')
    >>> filtered = lfilter(dat, b, a)

    """
    dat = deepcopy_xarr(dat)
    time_axis = list(dat.dims).index('time')
    if zi is None:
        new_data = scipy.signal.lfilter(b, a, dat.data, axis=time_axis)
    else:
        new_data, zo = scipy.signal.lfilter(b, a, dat.data, zi=zi, axis=time_axis)

    new_dat = xr.DataArray(new_data,
                       coords={'channels': dat.channels, 'time': dat.time},
                       dims=dat.dims,
                       attrs=dat.attrs)
    if zi is None:
        return new_dat
    else:
        return new_dat, zo


def exponential_standardize_cnt(cnt, init_block_size=1000, factor_new=1e-3,
        eps=1e-4):
    cnt = deepcopy_xarr(cnt)
    cnt_data = cnt.data
    time_axis = list(cnt.dims).index('time')
    cnt_data = cnt_data.swapaxes(0, time_axis)
    standardized_data = exponential_running_standardize(cnt_data,
        factor_new=factor_new, init_block_size=init_block_size,eps=eps)
    standardized_data = standardized_data.swapaxes(0, time_axis)
    cnt.data = standardized_data
    return cnt


def exponential_demean_cnt(cnt, init_block_size=1000, factor_new=1e-3):
    cnt = deepcopy_xarr(cnt)
    cnt_data = cnt.data
    time_axis = list(cnt.dims).index('time')
    cnt_data = cnt_data.swapaxes(0, time_axis)
    demeaned_data = exponential_running_demean(cnt_data,
        factor_new=factor_new, init_block_size=init_block_size)
    demeaned_data = demeaned_data.swapaxes(0, time_axis)
    cnt.data = demeaned_data
    return cnt


def exponential_running_standardize(data, factor_new=0.001,
                                    init_block_size=None, eps=1e-4):
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(data[0:init_block_size], axis=other_axis,
                            keepdims=True)
        init_std = np.std(data[0:init_block_size], axis=other_axis,
                          keepdims=True)
        init_block_standardized = (data[0:init_block_size] - init_mean) / \
                                  np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized



def exponential_running_demean(data, factor_new=0.001, init_block_size=None):
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    demeaned = np.array(demeaned)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(data[0:init_block_size], axis=other_axis,
                            keepdims=True)
        demeaned[0:init_block_size] = (data[0:init_block_size] - init_mean)
    return demeaned


def calculate_csp(epo, classes=None):
    """Calculate the Common Spatial Pattern (CSP) for two classes.
    Now with pattern computation as in matlab bbci toolbox
    https://github.com/bbci/bbci_public/blob/c7201e4e42f873cced2e068c6cbb3780a8f8e9ec/processing/proc_csp.m#L112
    
    This method calculates the CSP and the corresponding filters. Use
    the columns of the patterns and filters.
    Examples
    --------
    Calculate the CSP for the first two classes::
    >>> w, a, d = calculate_csp(epo)
    >>> # Apply the first two and the last two columns of the sorted
    >>> # filter to the data
    >>> filtered = apply_spatial_filter(epo, w[:, [0, 1, -2, -1]])
    >>> # You'll probably want to get the log-variance along the time
    >>> # axis, this should result in four numbers (one for each
    >>> # channel)
    >>> filtered = np.log(np.var(filtered, 0))
    Select two classes manually::
    >>> w, a, d = calculate_csp(epo, [2, 5])
    Parameters
    ----------
    epo : epoched Data object
        this method relies on the ``epo`` to have three dimensions in
        the following order: class, time, channel
    classes : list of two ints, optional
        If ``None`` the first two different class indices found in
        ``epo.axes[0]`` are chosen automatically otherwise the class
        indices can be manually chosen by setting ``classes``
    Returns
    -------
    v : 2d array
        the sorted spatial filters
    a : 2d array
        the sorted spatial patterns. Column i of a represents the
        pattern of the filter in column i of v.
    d : 1d array
        the variances of the components
    Raises
    ------
    AssertionError :
        If:
          * ``classes`` is not ``None`` and has less than two elements
          * ``classes`` is not ``None`` and the first two elements are
            not found in the ``epo``
          * ``classes`` is ``None`` but there are less than two
            different classes in the ``epo``
    See Also
    --------
    :func:`apply_spatial_filter`, :func:`apply_csp`, :func:`calculate_spoc`
    References
    ----------
    http://en.wikipedia.org/wiki/Common_spatial_pattern
    """
    n_channels = len(epo.channels.data)
    if classes is None:
        # automagically find the first two different classidx
        # we don't use uniq, since it sorts the classidx first
        # first check if we have a least two diffeent idxs:
        unique_classes = np.unique(epo.trials.data)
        assert len(unique_classes) == 2
        cidx1 = unique_classes[0]
        cidx2 = unique_classes[1]
    else:
        assert (len(classes) == 2 and
            classes[0] in epo.trials.data and
            classes[1] in epo.trials.data)
        cidx1 = classes[0]
        cidx2 = classes[1]
    epoc1 = select_classes(epo, [cidx1])
    epoc2 = select_classes(epo, [cidx2])
    # we need a matrix of the form (observations, channels) so we stack trials
    # and time per channel together

    x1 = epoc1.transpose('trials', 'time', 'channels').data.reshape(-1, n_channels)
    x2 = epoc2.transpose('trials', 'time', 'channels').data.reshape(-1, n_channels)
    # compute covariance matrices of the two classes
    c1 = np.cov(x1.transpose())
    c2 = np.cov(x2.transpose())
    # solution of csp objective via generalized eigenvalue problem
    # in matlab the signature is v, d = eig(a, b)
    d, v = sp.linalg.eig(c1-c2, c1+c2)
    d = d.real
    # make sure the eigenvalues and -vectors are correctly sorted
    indx = np.argsort(d)
    # reverse
    indx = indx[::-1]
    d = d.take(indx)
    v = v.take(indx, axis=1)
    
    # Now compute patterns
    #old pattern computation
    #a = sp.linalg.inv(v).transpose()
    c_avg = (c1 + c2) / 2.0
    
    # compare 
    # https://github.com/bbci/bbci_public/blob/c7201e4e42f873cced2e068c6cbb3780a8f8e9ec/processing/proc_csp.m#L112
    # with W := v
    v_with_cov = np.dot(c_avg, v)
    source_cov = np.dot(np.dot(v.T, c_avg), v)
    # matlab-python comparison
    """
    v_with_cov = np.array([[1,2,-2],
             [3,-2,4],
             [5,1,0.3]])

    source_cov = np.array([[1,2,0.5],
                  [2,0.6,4],
                  [0.5,4,2]])
    
    sp.linalg.solve(source_cov.T, v_with_cov.T).T
    # for matlab
    v_with_cov = [[1,2,-2],
                 [3,-2,4],
                 [5,1,0.3]]
    
    source_cov = [[1,2,0.5],
                  [2,0.6,4],
                  [0.5,4,2]]
    v_with_cov / source_cov"""

    a = sp.linalg.solve(source_cov.T, v_with_cov.T).T
    return v, a, d


def apply_csp_fast(epo, filt, columns=[0, -1]):
    """Apply the CSP filter.

    Apply the spacial CSP filter to the epoched data.

    Parameters
    ----------
    epo : epoched ``Data`` object
        this method relies on the ``epo`` to have three dimensions in
        the following order: class, time, channel
    filt : 2d array
        the CSP filter (i.e. the ``v`` return value from
        :func:`calculate_csp`)
    columns : array of ints, optional
        the columns of the filter to use. The default is the first and
        the last one.

    Returns
    -------
    epo : epoched ``Data`` object
        The channels from the original have been replaced with the new
        virtual CSP channels.

    Examples
    --------

    >>> w, a, d = calculate_csp(epo)
    >>> epo = apply_csp_fast(epo, w)

    See Also
    --------
    :func:`calculate_csp`
    :func:`apply_csp`

    """
    epo = deepcopy_xarr(epo)
    assert epo.dims == ('trials', 'time', 'channels')
    f = filt[:, columns]
    data = np.empty((epo.data.shape[0], epo.data.shape[1], f.shape[1]))
    for trial_i in range(epo.data.shape[0]):
        data[trial_i] = np.dot(epo.data[trial_i], f)

    csp_filter_names = np.array(['csp %i' % i for i in range(data.shape[-1])])

    epo = xr.DataArray(data,
                       coords={'trials': epo.trials, 'time': epo.time,
                               'channels': csp_filter_names,},
                       dims=('trials', 'time','channels',),
                       attrs=epo.attrs)

    return epo


def apply_csp_var_log(epo, filters, columns):
    csp_filtered = apply_csp_fast(epo, filters, columns)
    assert csp_filtered.dims == ('trials', 'time','channels',)
    feature_data = np.log(np.var(csp_filtered.data, axis=1))
    csp_filter_names = np.array(['csp %i' % i
                                 for i in range(feature_data.shape[-1])])
    features = xr.DataArray(feature_data,
                       coords={'trials': csp_filtered.trials,
                               'CSP filter': csp_filter_names,},
                       dims=('trials', 'CSP filter',),
                       attrs=csp_filtered.attrs)
    return features


def lda_train_scaled(fv, shrink=False):
    """Train the LDA classifier.

    Parameters
    ----------
    fv : ``Data`` object
        the feature vector must have 2 dimensional data, the first
        dimension being the class axis. The unique class labels must be
        0 and 1 otherwise a ``ValueError`` will be raised.
    shrink : Boolean, optional
        use shrinkage

    Returns
    -------
    w : 1d array
    b : float

    Raises
    ------
    ValueError : if the class labels are not exactly 0s and 1s

    Examples
    --------

    >>> clf = lda_train(fv_train)
    >>> out = lda_apply(fv_test, clf)

    See Also
    --------
    lda_apply

    """
    assert shrink is True
    assert fv.dims[0] == 'trials'
    assert len(fv.dims) == 2
    x = fv.data
    y = fv.trials.data
    if len(np.unique(y)) != 2:
        raise ValueError('Should only have two unique class labels, instead got'
            ': {labels}'.format(labels=np.unique(y)))
    # Use sorted labels
    labels = np.sort(np.unique(y))
    mu1 = np.mean(x[y == labels[0]], axis=0)
    mu2 = np.mean(x[y == labels[1]], axis=0)
    # x' = x - m
    m = np.empty(x.shape)
    m[y == labels[0]] = mu1
    m[y == labels[1]] = mu2
    x2 = x - m
    # w = cov(x)^-1(mu2 - mu1)
    if shrink:
        estimator = LW()
        covm = estimator.fit(x2).covariance_
    else:
        covm = np.cov(x2.T)
    w = np.dot(np.linalg.pinv(covm), (mu2 - mu1))

    #  From matlab bbci toolbox:
    # https://github.com/bbci/bbci_public/blob/fe6caeb549fdc864a5accf76ce71dd2a926ff12b/classification/train_RLDAshrink.m#L133-L134
    #C.w= C.w/(C.w'*diff(C_mean, 1, 2))*2;
    #C.b= -C.w' * mean(C_mean,2);
    w = (w / np.dot(w.T, (mu2 - mu1))) * 2
    b = np.dot(-w.T, np.mean((mu1, mu2), axis=0))
    assert not np.any(np.isnan(w))
    assert not np.isnan(b)
    return w, b


def lda_apply(fv, clf):
    """Apply feature vector to LDA classifier.

    Parameters
    ----------
    fv : ``Data`` object
        the feature vector must have a 2 dimensional data, the first
        dimension being the class axis.
    clf : (1d array, float)

    Returns
    -------

    out : 1d array
        The projection of the data on the hyperplane.

    Examples
    --------

    >>> clf = lda_train(fv_train)
    >>> out = lda_apply(fv_test, clf)


    See Also
    --------
    lda_train

    """
    assert fv.dims[0] == 'trials'
    x = fv.data
    w, b = clf
    return np.dot(x, w) + b


def select_marker_classes(cnt, classes, copy_data=False):
    if copy_data is True:
        cnt = deepcopy_xarr(cnt)

    event_mask = [(ev_code in classes) for ev_code in cnt.attrs['events'][:,1]]
    cnt.attrs['events'] = cnt.attrs['events'][event_mask]
    return cnt


def select_marker_epochs(cnt, epoch_inds, copy_data=False):
    # Restrict markers to only the correct epoch inds..
    # use list comprehension and not conversion to numpy array
    # + indexing to preserve types (type of first part of each marker,
    # the time can be different from type of second part, the label)
    # transforming to numpy array
    # can lead to upcasting of labels from int to float for example....
    if copy_data is True:
        cnt = deepcopy_xarr(cnt)
    event_mask = [(i in epoch_inds) for i in range(len(cnt.attrs['events']))]

    cnt.attrs['events'] = cnt.attrs['events'][event_mask]
    return cnt


#### OLD
def set_channel_to_zero(cnt, chan_name):
    assert chan_name in cnt.axes[1]
    data = cnt.data.copy()
    i_chan = cnt.axes[1].tolist().index(chan_name)
    data[:,i_chan] = 0
    return cnt.copy(data=data)

def select_marker_classes_epoch_range(cnt, classes, start,stop,copy_data=False):
    cnt = select_marker_classes(cnt, classes, copy_data)
    cnt = select_marker_epoch_range(cnt, start, stop, copy_data)
    return cnt

def select_marker_epoch_range(cnt, start, stop, copy_data=False):
    if start is None:
        start = 0
    if stop is None:
        stop = len(cnt.markers)
    epoch_inds = range(start,stop)
    return select_marker_epochs(cnt, epoch_inds, copy_data)

def select_ival_with_markers(cnt, segment_ival):
    """Select the ival of the data that has markers inside.
    Respect segment ival.
    Keeps data from 2 sec before first marker + segment_ival[0] to
    2 sec after last marker + segment_ival[1]"""
    ms_first_marker = cnt.markers[0][0]
    
    # allow 2 sec before first marker and after last marker
    start = max(0, ms_first_marker + segment_ival[0] -2000)
    ms_last_marker = cnt.markers[-1][0]
    stop = ms_last_marker + segment_ival[1] + 2000
    
    cnt = select_ival(cnt, [start,stop])
    # possibly subtract first element of timeaxis so timeaxis starts at 0 again?
    return cnt

def create_cnt_y_start_end_marker(cnt, start_marker_def, end_marker_def,
    segment_ival, timeaxis=-2):
    """Segment ival is : (offset to start marker, offset to end marker)"""
    start_to_end_value = dict()
    for class_name in start_marker_def:
        start_marker_vals = start_marker_def[class_name]
        end_marker_vals = end_marker_def[class_name]
        assert len(start_marker_vals) == 1
        assert len(end_marker_vals) == 1
        start_to_end_value[start_marker_vals[0]] = end_marker_vals[0] 
            
        
    assert len(cnt.markers) % 2 == 0
    start_markers = cnt.markers[::2]
    end_markers = cnt.markers[1::2]

    # Assuming start marker vals are 1 ... n_classes
    # Otherwise change code...
    all_start_marker_vals = start_to_end_value.keys()
    n_classes = np.max(all_start_marker_vals)
    assert np.array_equal(np.sort(all_start_marker_vals),
                         range(1, n_classes+1)), (
        "Assume start marker values are from 1...n_classes")
    
    y = np.zeros((cnt.data.shape[0], np.max(all_start_marker_vals)), dtype= np.int32)
    for i_event in xrange(len(start_markers)):
        start_marker_ms, start_marker_val  = start_markers[i_event]
        end_marker_ms, end_marker_val = end_markers[i_event]
        assert end_marker_val == start_to_end_value[start_marker_val], (
            "Expect the end marker value to be corresponding, but have"
            "start marker: {:f}, end marker: {:f}".format(
            start_marker_val, end_marker_val))
        first_index = np.searchsorted(cnt.axes[timeaxis], start_marker_ms + segment_ival[0])
        last_index = np.searchsorted(cnt.axes[timeaxis], end_marker_ms+segment_ival[1])
        # -1 to transform 1-based to 0-based indexing
        y[first_index:last_index, int(start_marker_val) - 1] = 1 
        
    return y
        
def create_cnt_y(cnt, segment_ival, marker_def=None, timeaxis=-2,
        trial_classes=None):
    """ Create a one-hot-encoded signal for all the markers in cnt.
    Dimensions will be #samples x #classes(i.e. marker types)"""
    if marker_def is None:
        marker_def = {'1': [1], '2': [2], '3': [3], '4': [4]}
    n_classes = len(marker_def)
    assert np.all([len(labels) == 1 for labels in 
        marker_def.values()]), (
        "Expect only one label per class, otherwise rewrite...")

    classes = sorted([labels[0] for labels in marker_def.values()])
    # restrict to only those markers in marker def
    cnt = select_marker_classes(cnt, classes)
    event_samples_and_classes = get_event_samples_and_classes(cnt,
        timeaxis=timeaxis)
    # In case classes are not from 1...n_classes
    # lets map them to be from 1 .. n_classes
    if classes != range(1,n_classes+1) and (trial_classes is None):
        for i_marker in xrange(len(event_samples_and_classes)):
            old_class = event_samples_and_classes[i_marker][1]
            new_class = classes.index(old_class) + 1 #+1 for matlab-based indexing
            event_samples_and_classes[i_marker][1] = new_class
    
    if trial_classes is not None:
        old_class_to_new_class = create_old_class_to_new_class(marker_def,
            trial_classes)
        for i_marker in xrange(len(event_samples_and_classes)):
            old_class = event_samples_and_classes[i_marker][1]
            new_class = old_class_to_new_class[old_class]
            event_samples_and_classes[i_marker][1] = new_class
    
    return get_y_signal(event_samples_and_classes,n_samples=len(cnt.data),
                       n_classes=n_classes, segment_ival=segment_ival,
                       fs=cnt.fs)
    
def create_old_class_to_new_class(marker_def, trial_classes):
    """Maps marker codes to class indices according to the order given
    by trial classes."""
    old_class_to_new_class = dict()
    for new_class, class_name in enumerate(trial_classes):
        old_class = marker_def[class_name]
        assert len(old_class) == 1, "Expect only one marker per class, else rewrite below"
        old_class = old_class[0]
        new_class += 1 # for matlab based indexing
        old_class_to_new_class[old_class] = new_class
    return old_class_to_new_class

def get_event_samples_and_classes(cnt, timeaxis=-2):
    event_samples_and_classes = np.ones(
        (len(cnt.markers), 2),dtype=np.int32)
    
    for i_marker in xrange(len(cnt.markers)):
        marker_ms = cnt.markers[i_marker][0]
        marker_class = cnt.markers[i_marker][1]
        i_sample = np.searchsorted(cnt.axes[timeaxis], marker_ms)
        event_samples_and_classes[i_marker] = [i_sample, marker_class]

    return event_samples_and_classes


def get_y_signal(event_samples_and_classes, n_samples, n_classes, segment_ival, fs):
    i_samples, labels = zip(*event_samples_and_classes)
    """ Expects classes in event_samples_and_classes to be from
    1 to n_classes (inclusive)"""
    # Create y "signal", first zero everywhere, in loop assign 
    #  1 to where a trial for the respective class happened
    # (respect segmentation interval for this)
    y = np.zeros((n_samples, n_classes),
        dtype=np.int32)
    trial_start_offset = int(segment_ival[0] * fs / 1000.0)
    trial_stop_offset = int(segment_ival[1] * fs / 1000.0)

    unique_labels = sorted(np.unique(labels))
    assert np.array_equal(unique_labels, range(1, n_classes+1)), (
        "Expect labels to be from 1 to n_classes...")

    for i_trial in xrange(len(labels)):
        i_start_sample = i_samples[i_trial]
        i_class = labels[i_trial]-1 # -1 due to 1-based matlab indexing
        # make sure trial is within bounds
        if ((i_start_sample + trial_start_offset >= 0) and
            (i_start_sample + trial_stop_offset <= len(y))):
            y[i_start_sample+trial_start_offset:i_start_sample+trial_stop_offset, 
                i_class] = 1
    return y



def online_standardize_epo(epo_train, epo_test):
    # standard dim inds are trials and samples
    # so compute means/vars for channels
    # (epo data is #trialsx#samplesx#channels)
    standard_dim_inds=(0,1)
    std_eps = 1e-4
    train_mean = np.mean(epo_train.data, axis=standard_dim_inds, keepdims=True)
    train_std = np.std(epo_train.data, axis=standard_dim_inds, keepdims=True)
    new_epo_train_data = (epo_train.data - train_mean) / (train_std + std_eps)
    n_old_trials = len(epo_train.axes[0])
    assert len(epo_train.axes[0]) == len(epo_train.data)

    new_epo_test_data = online_standardize(epo_test.data, 
                                        old_mean=train_mean,
                                        old_std=train_std, 
                                        n_old_trials=n_old_trials, 
                                        dims_to_squash=standard_dim_inds[1:], # 0th batch dim should be ignored for this call
                                        std_eps=std_eps)
    assert np.array_equal(epo_train.data.shape, new_epo_train_data.shape)
    assert np.array_equal(epo_test.data.shape, new_epo_test_data.shape)
    return epo_train.copy(data=new_epo_train_data), epo_test.copy(data=new_epo_test_data)

def select_channels_epo(epo, regexp_list, invert=False, chanaxis=-1):
    """Select channels from data.

    The matching is case-insensitive and locale-aware (as in
    ``re.IGNORECASE`` and ``re.LOCALE``). The regular expression always
    has to match the whole channel name string

    Parameters
    ----------
    dat : Data
    regexp_list : list of regular expressions
        The regular expressions provided, are used directly by Python's
        :mod:`re` module, so all regular expressions which are understood
        by this module are allowed.

        Internally the :func:`re.match` method is used, additionally to
        check for a match (which also matches substrings), it is also
        checked if the whole string matched the pattern.
    invert : Boolean, optional
        If True the selection is inverted. Instead of selecting specific
        channels, you are removing the channels. (default: False)
    chanaxis : int, optional
        the index of the channel axis in ``dat`` (default: -1)

    Returns
    -------
    dat : Data
        A copy of ``dat`` with the channels, matched by the list of
        regular expressions.

    Examples
    --------
    Select all channels Matching 'af.*' or 'fc.*'

    >>> dat_new = select_channels(dat, ['af.*', 'fc.*'])

    Remove all channels Matching 'emg.*' or 'eog.*'

    >>> dat_new = select_channels(dat, ['emg.*', 'eog.*'], invert=True)

    Even if you only provide one Regular expression, it has to be in an
    array:

    >>> dat_new = select_channels(dat, ['af.*'])

    See Also
    --------
    remove_channels : Remove Channels
    re : Python's Regular Expression module for more information about
        regular expressions.

    """
    chan_mask = np.array([False for _ in range(len(epo.axes[chanaxis]))])
    for c_idx, c in enumerate(epo.axes[chanaxis]):
        for regexp in regexp_list:
            m = re.match(regexp, c, re.IGNORECASE | re.LOCALE)
            if m and m.group() == c:
                chan_mask[c_idx] = True
                # no need to look any further for matches for this channel
                break
    if invert:
        chan_mask = ~chan_mask
    data = epo.data.compress(chan_mask, chanaxis)
    channels = epo.axes[chanaxis][chan_mask]
    axes = epo.axes[:]
    axes[chanaxis] = channels
    return epo.copy(data=data, axes=axes)





def apply_csp_one_dot(epo, filt, columns=[0, -1]):
    """Apply the CSP filter.
    Version with just one dot product.Surprisingly in my tests,
    this is slower than the version looping over the trials
     (~270 ms vs ~210ms for normal sized dataset)

    Apply the spacial CSP filter to the epoched data.

    Parameters
    ----------
    epo : epoched ``Data`` object
        this method relies on the ``epo`` to have three dimensions in
        the following order: class, time, channel
    filt : 2d array
        the CSP filter (i.e. the ``v`` return value from
        :func:`calculate_csp`)
    columns : array of ints, optional
        the columns of the filter to use. The default is the first and
        the last one.

    Returns
    -------
    epo : epoched ``Data`` object
        The channels from the original have been replaced with the new
        virtual CSP channels.

    Examples
    --------

    >>> w, a, d = calculate_csp(epo)
    >>> epo = apply_csp_fast(epo, w)

    See Also
    --------
    :func:`calculate_csp`
    :func:`apply_csp`

    """
    f = filt[:, columns]
    data = np.dot(epo.data, f)
    axes = epo.axes[:]
    axes[-1] = np.array(['csp %i' % i for i in range(data.shape[-1])])
    names = epo.names[:]
    names[-1] = 'CSP Channel'
    dat = epo.copy(data=data, axes=axes, names=names)
    return dat



def common_average_reference_cnt(cnt):
    assert cnt.data.ndim == 2
    car = np.mean(cnt.data, axis=1, keepdims=True)
    newdata = cnt.data - car
    return cnt.copy(data=newdata)

def rereference_to(cnt, sensor_name):
    assert cnt.data.ndim == 2
    sensor_ind = np.flatnonzero(np.array(cnt.axes[1]) == sensor_name)[0]
    newdata = cnt.data - cnt.data[:,sensor_ind:sensor_ind+1]
    return cnt.copy(data=newdata)



def subsample_cnt(cnt, newfs, timeaxis=-2):
    if newfs == cnt.fs:
        log.info("Just copying data, no resampling, since new sampling rate same.")
        return cnt.copy()
    assert (float(cnt.fs) / float(newfs)).is_integer(), ("Only allow "
        "subsamplings for integer ratios")
    subsample_factor = int(float(cnt.fs) / float(newfs))
    resampled_data = cnt.data[::subsample_factor]
    newaxes= deepcopy(cnt.axes)
    timesteps = cnt.axes[timeaxis][::subsample_factor]
    newaxes[timeaxis] = timesteps
    return cnt.copy(data=resampled_data, fs=newfs, axes=newaxes)



def lda_train(fv, shrink=False):
    """Train the LDA classifier. Fixed to work with all pairs of
    class labels, also different from 0/1.

    Parameters
    ----------
    fv : ``Data`` object
        the feature vector must have 2 dimensional data, the first
        dimension being the class axis. The unique class labels must be
        0 and 1 otherwise a ``ValueError`` will be raised.
    shrink : Boolean, optional
        use shrinkage

    Returns
    -------
    w : 1d array
    b : float

    Raises
    ------
    ValueError : if the class labels are not exactly 0s and 1s

    Examples
    --------

    >>> clf = lda_train(fv_train)
    >>> out = lda_apply(fv_test, clf)

    See Also
    --------
    lda_apply

    """
    x = fv.data
    y = fv.axes[0]
    if len(np.unique(y)) != 2:
        raise ValueError('Should only have two unique class labels, instead got'
            ': {labels}'.format(labels=np.unique(y)))
    # Use sorted labels
    labels = np.sort(np.unique(y))
    mu1 = np.mean(x[y == labels[0]], axis=0)
    mu2 = np.mean(x[y == labels[1]], axis=0)
    # x' = x - m
    m = np.empty(x.shape)
    m[y == labels[0]] = mu1
    m[y == labels[1]] = mu2
    x2 = x - m
    # w = cov(x)^-1(mu2 - mu1)
    if shrink:
        covm = LW().fit(x2).covariance_
    else:
        covm = np.cov(x2.T)
    w = np.dot(np.linalg.pinv(covm), (mu2 - mu1))
    # b = 1/2 x'(mu1 + mu2)
    b = -0.5 * np.dot(w.T, (mu1 + mu2))
    return w, b

def resample_epo(epo, newfs, timeaxis=-2):
    assert epo.data.ndim == 3, "Expect 3 dimensions/epoched data"
    old_data = epo.data
    newnumsamples = int(old_data.shape[timeaxis] * newfs / float(epo.fs))
    new_data = scipy.signal.resample(old_data, num=newnumsamples, axis=timeaxis)
    return epo.copy(data=new_data, fs=newfs)


def running_standardize_epo(epo, factor_new=0.9, init_block_size=50):
    """ Running standardize channelwise."""
    assert factor_new <= 1.0 and factor_new >= 0.0
    running_means = exponential_running_mean(epo.data, factor_new=factor_new, 
        init_block_size=init_block_size, axis=1)
    running_means = np.expand_dims(running_means, 1)
    demeaned_data = epo.data - running_means
    running_vars = exponential_running_var_from_demeaned(demeaned_data,
        running_means, factor_new=factor_new, init_block_size=init_block_size,
        axis=1)
    
    running_vars = np.expand_dims(running_vars, 1)
    running_std = np.sqrt(running_vars)
    
    standardized_epo_data = demeaned_data / running_std
    return epo.copy(data=standardized_epo_data)

def highpass_cnt(cnt, low_cut_off_hz, filt_order=3):
    if (low_cut_off_hz is None) or (low_cut_off_hz == 0):
        log.info("Not doing any highpass, since low 0 or None")
        return cnt.copy()
    b,a = scipy.signal.butter(filt_order, low_cut_off_hz/(cnt.fs/2.0),
        btype='highpass')
    assert filter_is_stable(a)
    cnt_highpassed = lfilter(cnt,b,a)
    return cnt_highpassed

def highpass_filt_filt_cnt(cnt, low_cut_off_hz, filt_order=3):
    if (low_cut_off_hz is None) or (low_cut_off_hz == 0):
        log.info("Not doing any highpass, since low 0 or None")
        return cnt.copy()
    b,a = scipy.signal.butter(filt_order, low_cut_off_hz/(cnt.fs/2.0),
        btype='highpass')
    assert filter_is_stable(a)
    cnt_highpassed = filtfilt(cnt,b,a)
    return cnt_highpassed

def lowpass_cnt(cnt, high_cut_off_hz, filt_order=3):
    if (high_cut_off_hz is None) or (high_cut_off_hz == cnt.fs):
        log.info("Not doing any lowpass, since ince high cut hz is None or current fs")
        return cnt.copy()
    b,a = scipy.signal.butter(filt_order, high_cut_off_hz/(cnt.fs/2.0),
        btype='lowpass')
    assert filter_is_stable(a)
    cnt_lowpassed = lfilter(cnt,b,a)
    return cnt_lowpassed

def lowpass_filt_filt_cnt(cnt, high_cut_off_hz, filt_order=3):
    if (high_cut_off_hz is None) or (high_cut_off_hz == cnt.fs):
        log.info("Not doing any lowpass, since ince high cut hz is None or current fs")
    b,a = scipy.signal.butter(filt_order, high_cut_off_hz/(cnt.fs/2.0),
        btype='lowpass')
    assert filter_is_stable(a)
    cnt_lowpassed = filtfilt(cnt,b,a)
    return cnt_lowpassed

def bandpass_cnt(cnt, low_cut_hz, high_cut_hz, filt_order=3):
    """Bandpass cnt signal using butterworth filter.
    Uses lowpass in case low cut hz is exactly zero."""
    if (low_cut_hz == 0 or low_cut_hz is None) and (
        high_cut_hz == None or high_cut_hz == cnt.fs):
        log.info("Not doing any bandpass, since low 0 or None and "
            "high None or current fs")
        return cnt.copy()
    if low_cut_hz == 0 or low_cut_hz == None:
        log.info("Using lowpass filter since low cut hz is 0 or None")
        return lowpass_cnt(cnt, high_cut_hz, filt_order=filt_order)
    if high_cut_hz == None or high_cut_hz == cnt.fs:
        log.info("Using highpass filter since high cut hz is None or current fs")
        return highpass_cnt(cnt, low_cut_hz, filt_order=filt_order)
        
    nyq_freq = 0.5 * cnt.fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype='bandpass')
    assert filter_is_stable(a), "Filter should be stable..."
    cnt_bandpassed = lfilter(cnt,b,a)
    return cnt_bandpassed


def bandpass_filt_filt_cnt(cnt, low_cut_hz, high_cut_hz, filt_order=3):
    """Bandpass cnt signal using butterworth filter.
    Uses lowpass in case low cut hz is exactly zero."""
    if (low_cut_hz == 0 or low_cut_hz == None) and (
        high_cut_hz == None or high_cut_hz == cnt.fs):
        log.info("Not doing any bandpass, since low 0 or None and "
            "high None or current fs")
        return cnt.copy()
    if low_cut_hz == 0 or low_cut_hz == None:
        log.info("Using lowpass filter since low cut hz is 0 or None")
        return lowpass_filt_filt_cnt(cnt, high_cut_hz, filt_order=filt_order)
    if high_cut_hz == None or high_cut_hz == cnt.fs:
        log.info("Using highpass filter since high cut hz is None or current fs")
        return highpass_filt_filt_cnt(cnt, low_cut_hz, filt_order=filt_order)
        
    nyq_freq = 0.5 * cnt.fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype='bandpass')
    assert filter_is_stable(a), "Filter should be stable..."
    cnt_bandpassed = filtfilt(cnt,b,a)
    return cnt_bandpassed


def bandstop_cnt(cnt, low_cut_hz, high_cut_hz, filt_order=3):
    nyq_freq = 0.5 * cnt.fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype='bandstop')
    
    assert filter_is_stable(a)
    cnt_bandpassed = lfilter(cnt,b,a)
    return cnt_bandpassed