import scipy
from wyrm.processing import lfilter, filtfilt
import numpy as np
from copy import deepcopy
from braindecode.datahandling.preprocessing import (exponential_running_mean, 
    exponential_running_var_from_demeaned, OnlineAxiswiseStandardize)
from sklearn.covariance import LedoitWolf as LW
import scikits.samplerate
import re
import wyrm.types

def exponential_standardize_cnt(cnt):
    cnt_data = cnt.data
    init_block_size=8000
    factor_new = 0.001
    means = exponential_running_mean(cnt_data, factor_new=factor_new,
        init_block_size=init_block_size, axis=None)
    demeaned = cnt_data - means
    stds = np.sqrt(exponential_running_var_from_demeaned(
        demeaned, factor_new, init_block_size=init_block_size, axis=None))
    eps = 1e-4
    standardized_data = demeaned / np.maximum(stds, eps)
    return cnt.copy(data=standardized_data)

def online_standardize_epo(epo_train, epo_test):
    standard_dim_inds=(0,1)
    std_eps = 1e-4
    train_mean = np.mean(epo_train.data, axis=standard_dim_inds, keepdims=True)
    train_std = np.std(epo_train.data, axis=standard_dim_inds, keepdims=True)
    new_epo_train_data = (epo_train.data - train_mean) / (train_std + std_eps)
    n_old_trials = len(epo_train.axes[0])
    assert len(epo_train.axes[0]) == len(epo_train.data)

    new_epo_test_data = OnlineAxiswiseStandardize.standardize(epo_test.data, 
                                        standard_dim_inds, 
                                        old_mean=train_mean,
                                        old_std=train_std, 
                                        n_old_trials=n_old_trials, 
                                        use_only_new=False,
                                        new_factor=1.,
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
    chan_mask = np.array([False for i in range(len(epo.axes[chanaxis]))])
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
    f = filt[:, columns]
    data = np.empty((epo.data.shape[0], epo.data.shape[1], f.shape[1]))
    for trial_i in range(epo.data.shape[0]):
        data[trial_i] = np.dot(epo.data[trial_i], f)
    axes = epo.axes[:]
    axes[-1] = np.array(['csp %i' % i for i in range(data.shape[-1])])
    names = epo.names[:]
    names[-1] = 'CSP Channel'
    dat = epo.copy(data=data, axes=axes, names=names)
    return dat

def apply_csp_var_log(epo,filters, columns):
    feature = apply_csp_fast(epo, filters, columns)
    feature_data = np.log(np.var(feature.data, axis=1))
    feature = wyrm.types.Data(feature_data, axes = [feature.axes[0], columns], 
        names=['class', 'CSP filter'], units=['#', 'logvarHz'])
    assert feature.data.ndim == 2, ("Feature should only have "
        "one dimension for trials and another for features")
    return feature

def common_average_reference_cnt(cnt):
    assert cnt.data.ndim == 2
    car = np.mean(cnt.data, axis=1, keepdims=True)
    newdata = cnt.data - car
    return cnt.copy(data=newdata)

def resample_cnt(cnt, newfs, timeaxis=-2):
    resampled_data = scikits.samplerate.resample(cnt.data, newfs/float(cnt.fs), 
        type='sinc_fastest')
    # add sensor dim if only having one sensor...
    if resampled_data.ndim == 1:
        resampled_data = resampled_data[:, np.newaxis]
    newaxes= deepcopy(cnt.axes)
    timesteps = scikits.samplerate.resample(cnt.axes[timeaxis], 
        newfs/float(cnt.fs), type='linear')
    newaxes[timeaxis] = timesteps
    return cnt.copy(data=resampled_data, fs=newfs, axes=newaxes)

def segment_dat_fast(dat, marker_def, ival, newsamples=None, timeaxis=-2):
    """Convert a continuous data object to an epoched one.

    Given a continuous data object, a definition of classes, and an
    interval, this method looks for markers as defined in ``marker_def``
    and slices the dat according to the time interval given with
    ``ival`` along the ``timeaxis``. The returned ``dat`` object stores
    those slices and the class each slice belongs to.

    Epochs that are too close to the borders and thus too short are
    ignored.

    If the segmentation does not result in any epochs (i.e. the markers
    in ``marker_def`` could not be found in ``dat``, the resulting
    dat.data will be an empty array.

    This method is also suitable for **online processing**, please read
    the documentation for the ``newsamples`` parameter and have a look
    at the Examples below.

    Parameters
    ----------
    dat : Data
        the data object to be segmented
    marker_def : dict
        The keys are class names, the values are lists of markers
    ival : [int, int]
        The interval in milliseconds to cut around the markers. I.e. to
        get the interval starting with the marker plus the remaining
        100ms define the interval like [0, 100]. The start point is
        included, the endpoint is not (like: ``[start, end)``).  To get
        200ms before the marker until 100ms after the marker do:
        ``[-200, 100]`` Only negative or positive values are possible
        (i.e. ``[-500, -100]``)
    newsamples : int, optional
        consider the last ``newsamples`` samples as new data and only
        return epochs which are possible with the old **and** the new
        data (i.e. don't include epochs which where possible without the
        new data).

        If this parameter is ``None`` (default) ``segment_dat`` will
        always process the whole ``dat``, this is what you want for
        offline experiments where you process the whole data from a file
        at once. In online experiments however one usually gets the data
        incrementally, stores it in a ringbuffer to get the last n
        milliseconds. Consequently ``segment_dat`` gets overlapping data
        in each iteration (the amount of overlap is exactly the data -
        the new samples. To make sure each epoch appears only once
        within all iterations, ``segment_dat`` needs to know the number
        of new samples.


    timeaxis : int, optional
        the axis along which the segmentation will take place

    Returns
    -------
    dat : Data
        a copy of the resulting epoched data.

    Raises
    ------
    AssertionError
        * if ``dat`` has not ``.fs`` or ``.markers`` attribute or if
          ``ival[0] > ival[1]``.
        * if ``newsamples`` is not ``None`` or positive

    Examples
    --------

    Offline Experiment

    >>> # Define the markers belonging to class 1 and 2
    >>> md = {'class 1': ['S1', 'S2'],
    ...       'class 2': ['S3', 'S4']
    ...      }
    >>> # Epoch the data -500ms and +700ms around the markers defined in
    >>> # md
    >>> epo = segment_dat(cnt, md, [-500, 700])

    Online Experiment

    >>> # Define the markers belonging to class 1 and 2
    >>> md = {'class 1': ['S1', 'S2'],
    ...       'class 2': ['S3', 'S4']
    ...      }
    >>> # define the interval to epoch around a marker
    >>> ival = [0, 300]
    >>> while 1:
    ...     dat, mrk = amp.get_data()
    ...     newsamples = len(dat)
    ...     # the ringbuffer shall keep the last 2000 milliseconds,
    ...     # which is way bigger than our ival...
    ...     ringbuffer.append(dat, mrk)
    ...     cnt, mrk = ringbuffer.get()
    ...     # cnt contains now data up to 2000 millisecons, to make sure
    ...     # we don't see old markers again and again until they where
    ...     # pushed out of the ringbuffer, we need to tell segment_dat
    ...     # how many samples of cnt are actually new
    ...     epo = segment_dat(cnt, md, ival, newsamples=newsamples)

    """
    assert hasattr(dat, 'fs')
    assert hasattr(dat, 'markers')
    assert ival[0] <= ival[1]
    if newsamples is not None:
        assert newsamples >= 0
        # the times of the `newsamples`
        new_sample_times = dat.axes[timeaxis][-newsamples:] if newsamples > 0 else []
    # the expected length of each cnt in the resulted epo
    expected_samples = dat.fs * (ival[1] - ival[0]) / 1000
    data = []
    classes = []
    class_names = sorted(marker_def.keys())
    masks = []
    for t, m in dat.markers:
        for class_idx, classname in enumerate(class_names):
            if m in marker_def[classname]:
                first_index = np.searchsorted(dat.axes[timeaxis], t+ival[0])
                # as at last index will already be sth bigger or equal,
                # mask should be exclusive this index!
                last_index = np.searchsorted(dat.axes[timeaxis], t+ival[1])
                mask = range(first_index, last_index)
                if len(mask) != expected_samples:
                    # result is too short or too long, ignore it
                    continue
                # check if the new cnt shares at least one timepoint
                # with the new samples. attention: we don't only have to
                # check the ival but also the marker if it is on the
                # right side of the ival!
                if newsamples is not None:
                    times = dat.axes[timeaxis].take(mask)
                    if newsamples == 0:
                        continue
                    if (len(np.intersect1d(times, new_sample_times)) == 0 and
                        t < new_sample_times[0]):
                        continue
                masks.append(mask)
                classes.append(class_idx)
    if len(masks) == 0:
        data = np.array([])
    else:
        # np.take inserts a new dimension at `axis`...
        data = dat.data.take(masks, axis=timeaxis)
        # we want that new dimension at axis 0 so we have to swap it.
        # before that we have to convert the netagive axis indices to
        # their equivalent positive one, otherwise swapaxis will be one
        # off.
        if timeaxis < 0:
            timeaxis = dat.data.ndim + timeaxis
        data = data.swapaxes(0, timeaxis)
    axes = dat.axes[:]
    time = np.linspace(ival[0], ival[1], float(ival[1] - ival[0]) / 1000 * dat.fs, endpoint=False)
    axes[timeaxis] = time
    classes = np.array(classes)
    axes.insert(0, classes)
    names = dat.names[:]
    names.insert(0, 'class')
    units = dat.units[:]
    units.insert(0, '#')
    return dat.copy(data=data, axes=axes, names=names, units=units, class_names=class_names)


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

def highpass_cnt(cnt, low_cut_off_hz, filt_order=3):
    b,a = scipy.signal.butter(filt_order, low_cut_off_hz/(cnt.fs/2.0),btype='highpass')
    cnt_highpassed = lfilter(cnt,b,a)
    return cnt_highpassed

def lowpass_cnt(cnt, cut_off_hz, filt_order=3):
    b,a = scipy.signal.butter(filt_order, cut_off_hz/(cnt.fs/2.0),btype='lowpass')
    cnt_highpassed = lfilter(cnt,b,a)
    return cnt_highpassed


def highpass_filt_filt_cnt(cnt, low_cut_off_hz, filt_order=3):
    b,a = scipy.signal.butter(filt_order, low_cut_off_hz/(cnt.fs/2.0),btype='highpass')
    cnt_highpassed = filtfilt(cnt,b,a)
    return cnt_highpassed


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

def bandpass_cnt(cnt, low_cut_hz, high_cut_hz, filt_order=3):
    nyq_freq = 0.5 * cnt.fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype='band')
    
    cnt_bandpassed = lfilter(cnt,b,a)
    return cnt_bandpassed
