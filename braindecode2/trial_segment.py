import numpy as np
import xarray as xr


def create_target_series(cnt, marker_def, ival):
    """
    Compute one-hot encoded target series.
    Parameters
    ----------
    cnt : DateArray
    marker_def: OrderedDict (str -> list int)
        Dictionary mapping class names -> list of marker codes. Order of keys
         is used to determine indices of classes in one-hot encoded result. 
    ival : (number, number) tuple
        Segmentation interval for each trial.
    Returns
    -------
    2darray
        One-hot encoded target series.

    """
    assert 'fs' in cnt.attrs
    assert 'events' in cnt.attrs
    assert ival[0] <= ival[1]
    expected_samples = int(
        np.ceil(cnt.attrs['fs'] * (ival[1] - ival[0]) / 1000.0))
    class_names = marker_def.keys()
    targets = np.zeros((len(cnt.data), len(class_names)), dtype=np.int32)
    for t, m in cnt.attrs['events']:
        for class_idx, classname in enumerate(class_names):
            if m in marker_def[classname]:
                first_index = np.searchsorted(cnt.coords['time'], t + ival[0])
                # as at last index will already be sth bigger or equal,
                # mask should be exclusive this index!
                last_index = np.searchsorted(cnt.coords['time'], t + ival[1])
                n_samples = last_index - first_index
                if n_samples != expected_samples:
                    # result is too short or too long, ignore it
                    log.warn("ignoring trial")
                    log.warn(
                        "expected samples in trial segmentation {:d}".format(
                            expected_samples))
                    log.warn("actual samples {:d}".format(n_samples))
                    continue
                targets[first_index:last_index, class_idx] = 1
    return targets


def compute_trial_start_end_samples(y, check_trial_lengths_equal=True,
                                    input_time_length=None):
    """Computes trial start and end samples (end is inclusive) from
    one-hot encoded y-matrix.
    Specify input time length to kick out trials that are too short after
    signal start.

    Parameters
    ----------
    y : 2darray
    check_trial_lengths_equal : bool
         (Default value = True)
    input_time_length : int, optional
         (Default value = None)

    Returns
    -------

    """
    trial_part = np.sum(y, 1) == 1
    boundaries = np.diff(trial_part.astype(np.int32))
    i_trial_starts = np.flatnonzero(boundaries == 1) + 1
    i_trial_ends = np.flatnonzero(boundaries == -1)
    # it can happen that a trial is only partially there since the
    # cnt signal was split in the middle of a trial
    # for now just remove these
    # use that start marker should always be before or equal to end marker
    if i_trial_starts[0] > i_trial_ends[0]:
        # cut out first trial which only has end marker
        i_trial_ends = i_trial_ends[1:]
    if i_trial_starts[-1] > i_trial_ends[-1]:
        # cut out last trial which only has start marker
        i_trial_starts = i_trial_starts[:-1]

    assert (len(i_trial_starts) == len(i_trial_ends))
    assert (np.all(i_trial_starts <= i_trial_ends))
    # possibly remove first trials if they are too early
    if input_time_length is not None:
        while i_trial_starts[0] < (input_time_length - 1):
            i_trial_starts = i_trial_starts[1:]
            i_trial_ends = i_trial_ends[1:]
    if check_trial_lengths_equal:
        # just checking that all trial lengths are equal
        all_trial_lens = np.array(i_trial_ends) - np.array(i_trial_starts)
        assert all(all_trial_lens == all_trial_lens[0]), (
            "All trial lengths should be equal...")
    return i_trial_starts, i_trial_ends


def segment_dat(cnt, marker_def, ival):
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
    epo.data will be an empty array.

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

    Returns
    -------
    epo : Data
        a copy of the resulting epoched data.

    Raises
    ------
    AssertionError
        * if ``dat`` has not ``.fs`` or ``.markers`` attribute or if
          ``ival[0] > ival[1]``.

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

    """

    assert 'fs' in cnt.attrs
    assert 'events' in cnt.attrs
    y = create_target_series(cnt, marker_def, ival=ival)
    # Create classes per trial
    # and sample inds per trial from the target series
    starts, ends = compute_trial_start_end_samples(y)
    classes = [np.argmax(y[i_s]) for i_s in starts]
    sample_inds_per_trial = [range(i_s, i_e + 1) for (i_s, i_e) in
                             zip(starts, ends)]

    if len(sample_inds_per_trial) == 0:
        assert "No epochs not tested yet"
        data = np.array([])
    else:
        timeaxis = list(cnt.dims).index('time')
        # np.take inserts a new dimension at `axis`...
        data = cnt.data.take(sample_inds_per_trial, axis=timeaxis)
        # we want that new dimension at axis 0 so we have to swap it.
        # before that we have to convert the negative axis indices to
        # their equivalent positive one, otherwise swapaxis will be one
        # off.
        data = data.swapaxes(0, timeaxis)
    time = np.linspace(ival[0], ival[1],
                       int(cnt.attrs['fs'] * float(ival[1] - ival[0]) / 1000.0),
                       endpoint=False)
    epo = xr.DataArray(data,
                       coords={'trials': classes, 'time': time,
                               'channels': cnt.channels,},
                       dims=('trials', 'time','channels',))
    epo.attrs = cnt.attrs.copy()
    # remove events part
    epo.attrs.pop('events')
    return epo

