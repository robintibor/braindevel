from pylearn2.datasets.preprocessing import Preprocessor
from pylearn2.datasets.preprocessing import Pipeline
import numpy as np
import os
from pylearn2.utils import serial
from copy import deepcopy
import scipy.ndimage

def exponential_running_mean(data, factor_new, start_mean=None,
    init_block_size=None, axis=None):
    """ Compute the running mean across axis 0.
    For each datapoint in axis 0 its "running exponential mean" is computed as:
    Its mean * factor_new + so far computed mean * (1-factor_new).
    You can either specify a start mean or an init_block_size to 
    compute the start mean of. 
    In any case one mean per datapoint in axis 0 is returned."""
    assert not (start_mean is None and init_block_size is None), (""
        "Need either an init block or a start mean")
    assert start_mean is None or init_block_size is None, ("Can only use start mean"
                                                           "or init block size")
    assert factor_new <= 1.0    
    assert factor_new >= 0.0
    if axis is None:
        axis = tuple(range(1,np.array(data).ndim))
    if isinstance(axis, int):
        axis = (axis,)
    #running_means = np.ones
    factor_old = 1 - factor_new
    if start_mean is None:
        start_data = data[0:init_block_size]
        axes_for_start_mean = (0,) + axis # also average across init trials
        current_mean = np.mean(start_data, axis=axes_for_start_mean)
        data = data[init_block_size:]
        # repeat mean for running means
        running_means = [current_mean.copy() for _ in range(init_block_size)]
        running_means = np.array(running_means)
    else:
        # do first iteraiton of loop to init running means with correct shape
        # TODELAY: how to do this without this else clause? as it is basically
        # copied code from loop for first iteration
        first_mean = factor_new * np.mean(data[0:1], axis=axis) + factor_old * start_mean
        data = data[1:]
        running_means = np.array(first_mean) # wrap in array with one more dimension
        current_mean = first_mean
        
    for i in range(len(data)):
        next_mean = factor_new * np.mean(data[i:i+1], axis=axis) + factor_old * current_mean
        running_means = np.concatenate((running_means, next_mean), axis=0)
        current_mean = next_mean
    return running_means

def exponential_running_var(data, running_means, factor_new, start_var=None,
    init_block_size=None,   axis=None):
    """ Compute the running var across axis 0.
    For each datapoint in axis 0 its "running exponential var" is computed as:
    Its (datapoint - its running mean)**2 * factor_new + so far computed var * (1-factor_new).
    You can either specify a start mean or an initial block size to 
    compute the start mean of. 
    In any case one mean per datapoint in axis 0 is returned."""
    assert not (start_var is None and init_block_size is None), (""
        "Need either an init block or a start mean")
    assert start_var is None or init_block_size is None, ("Can only use start mean"
        "or init block size, not both")
    assert factor_new <= 1.0    
    assert factor_new >= 0.0
    data = np.array(data)
    if axis is None:
        axis = tuple(range(1,np.array(data).ndim))
    if isinstance(axis, int):
        axis = (axis,)
    # add missing dimensions to running means to prevent incorrect broadcasts
    for i in range(1,np.array(data).ndim):
        if i in axis:
            running_means = np.expand_dims(running_means, i)
    demeaned_data = data - running_means
    return exponential_running_var_from_demeaned(demeaned_data,
        factor_new, start_var, init_block_size, axis)

def exponential_running_var_from_demeaned(demeaned_data, factor_new, start_var=None,
    init_block_size=None,   axis=None):
    # TODELAY: split out if and else case into different functions
    # i.e. split apart a common function hjaving a start value (basically the loop)
    # and then split if and else into different functions
    factor_old = 1 - factor_new
    # prealloc running vars for performance (otherwise much slower)
    running_vars_shape = [len(demeaned_data)]
    for i in range(1, demeaned_data.ndim):
        if i not in axis:
            running_vars_shape.append(demeaned_data.shape[i])
    running_vars = np.ones(running_vars_shape) * np.nan
    if start_var is None:
        start_running_var = np.mean(np.square(demeaned_data[0:init_block_size]),
             axis=axis)
        start_running_var = np.mean(start_running_var, axis=0)
        running_vars[0:init_block_size] = start_running_var
        start_i = init_block_size
    else:
        first_start_var = np.mean(np.square(demeaned_data[0:1]), axis=axis)
        first_start_var = first_start_var[0] # unwrap
        first_var = factor_new * first_start_var + factor_old * start_var 
        running_vars[0] = first_var
        start_i = 1
    current_var = running_vars[start_i-1]
    for i in range(start_i, len(demeaned_data)):
        this_var = np.mean(np.square(demeaned_data[i:i+1]),axis=axis)
        next_var = factor_new * this_var + factor_old * current_var
        running_vars[i] = next_var
        current_var = next_var
    return running_vars


def compute_combined_mean(num_old, num_new, old_mean, new_mean):
    # formula according to http://stats.stackexchange.com/a/43183
    return (num_old*old_mean + num_new*new_mean) / \
                (num_old + num_new)

def compute_combined_std(num_old, num_new, old_mean, new_mean, 
    combined_mean, old_std, new_std):
    # formulas according to http://stats.stackexchange.com/a/43183
    combined_var = ((num_old * (old_std**2 + old_mean**2) + \
                    num_new * (new_std**2 + new_mean**2)) / \
                    (num_old + num_new)) - \
                    combined_mean ** 2
    combined_std = np.sqrt(combined_var)
    return combined_std

class RestrictTrials(Preprocessor):
    """ Restrict number of trials to given number or given fraction. """
    def __init__(self, number_of_trials):
        self.number_of_trials = number_of_trials

    def apply(self, dataset, can_fit=False):
        assert self.number_of_trials is not None
        if self.number_of_trials <= 1:
            assert self.number_of_trials > 0
            assert isinstance(self.number_of_trials, float)
            number_of_trials = int(len(dataset.y) * self.number_of_trials)
        else:
            assert isinstance(self.number_of_trials, int)
            number_of_trials = self.number_of_trials
        y = dataset.y[:number_of_trials]
        topo_view = dataset.get_topological_view()
        topo_view = topo_view[:number_of_trials]
        dataset.y = y
        dataset.set_topological_view(topo_view, dataset.view_converter.axes)

class RestrictToTwoClasses(Preprocessor):
    def __init__(self, classes):
        assert len(classes) == 2
        assert len(np.unique(classes)) == 2
        self.classes = classes
    
    def apply(self, dataset, can_fit=False):
        assert len(self.classes) == 2
        assert len(np.unique(self.classes)) == 2
        y = dataset.y
        classes_y = np.argmax(y, axis=1)
        assert self.classes[0] in classes_y
        assert self.classes[1] in classes_y
        wanted_inds = np.logical_or(classes_y == self.classes[0],
            classes_y == self.classes[1])
        topo_view = dataset.get_topological_view()
        topo_view = topo_view[wanted_inds]
        y = y[wanted_inds]
        y = y[:,self.classes] # remove columns of other classes
        dataset.y = y
        dataset.set_topological_view(topo_view, dataset.view_converter.axes)


class MergeFreqBins(Preprocessor):
    """ Merge freq bins, typically in higher frequencies, typically
    to reduce dimensionality and prevent overfitting. """
    def __init__(self, start_bin, num_bins_to_merge):
        self.start_bin = start_bin
        self.num_bins_to_merge = num_bins_to_merge
    
    def apply(self, dataset, can_fit=False):
        topo = dataset.get_topological_view()
        unchanged_topo = topo[:,:,:,:self.start_bin]
        changed_topo  = topo[:,:,:,self.start_bin:]
        changed_topo = changed_topo.reshape(changed_topo.shape[0], 
                          changed_topo.shape[1], 
                          changed_topo.shape[2], 
                          changed_topo.shape[3] // self.num_bins_to_merge, 
                          -1)
        changed_topo = np.mean(changed_topo, axis=4)
        
        new_topo = np.concatenate((unchanged_topo, changed_topo), axis=3)
        dataset.set_topological_view(new_topo, dataset.view_converter.axes)

class NoPreprocessing(Preprocessor):
    def apply(self, dataset, can_fit=False):
        """
        Do nothing...
        """
        return

class CachedSet(object):
    """ Just a container to be able to serialize nicely without overhead"""
    def __init__(self, X, y):
        self.X = X
        self.y = y

class CachedPipeline(Pipeline):
    preprocessed_directory = "data/preprocessed-sets"
    """
    A Preprocessor that sequentially applies a list
    of other Preprocessors.

    Parameters
    ----------
    items : WRITEME
    """

    def apply(self, dataset, can_fit=False):
        
        self._construct_cached_file_name(dataset, can_fit)
        if self._dataset_already_preprocessed():
            cached_set = self._load_cached_preprocessed_set()
            dataset.set_design_matrix(cached_set.X)
            dataset.y = cached_set.y
        else:
            super(CachedPipeline, self).apply(dataset, can_fit=can_fit)
            self._store_cached_preprocessed_set(dataset)

    def _construct_cached_file_name(self, dataset, can_fit):
        hash_dataset_str = str(self._hash_dataset(dataset))
        fitted_str = "fitted" if can_fit else "unfitted"
        preprocessor_names = ["{0}.{1}".format(item.__class__.__module__,
                               item.__class__.__name__) for item in self.items]
        preprocessor_str = "-".join(preprocessor_names)
        file_name = "{:s}.{:s}.{:s}.pkl".format(preprocessor_str,
            hash_dataset_str, fitted_str)
        self._cached_file_name = os.path.join(self.preprocessed_directory,
                                              file_name)
    
    def _hash_dataset(self, dataset):
        # construct hashes of X and y matrices,
        # first have to set write flag to False for this...
        # http://stackoverflow.com/a/16592241/1469195
        dataset.X.flags.writeable = False
        dataset.y.flags.writeable = False
        
        hash_value = hash((dataset.X.data, dataset.y.data))
        
        # allow changing of dataset again...
        dataset.X.flags.writeable = True
        dataset.y.flags.writeable = True
        return hash_value    
        
    def _dataset_already_preprocessed(self):
        return os.path.isfile(self._cached_file_name)
    
    def _load_cached_preprocessed_set(self):
        return serial.load(self._cached_file_name)
    
    def _store_cached_preprocessed_set(self, dataset):
        if not os.path.exists(self.preprocessed_directory):
            os.makedirs(self.preprocessed_directory)
        cached_set = CachedSet(X=dataset.X, y=dataset.y)
        serial.save(self._cached_file_name, cached_set)

class LogPreprocessor(Preprocessor):

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        topo_view = dataset.get_topological_view()
        topo_view = np.log(topo_view)
        dataset.set_topological_view(topo_view, dataset.view_converter.axes)

class RelativeToFirstBin(Preprocessor):

    def apply(self, dataset, can_fit=False):
        """
        Take bandpower relative to first timebin
        """
        # deep copy to amke sure we will not apply preprocessing
        # several times on same data
        topo_view = deepcopy(dataset.get_topological_view())
        topo_view[:,:,1:,:] = topo_view[:,:,1:,:] - topo_view[:,:,0:1,:]
        dataset.set_topological_view(topo_view, dataset.view_converter.axes)

class TranslatePreprocessor(Preprocessor):

    def __init__(self, start_bin, stop_bin):
        # assume 1-based indexing given
        self.start_bin = start_bin - 1
        self.stop_bin = stop_bin - 1

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        if can_fit:
            topo_view = dataset.get_topological_view()
            topo_view = np.copy(topo_view)
            targets = np.copy(dataset.y)
            translated_examples, new_targets = self._create_translated_examples(
               topo_view, targets)
            new_topo_view = np.concatenate((topo_view, 
                                               translated_examples))
            new_targets = np.concatenate((targets, new_targets))
            dataset.set_topological_view(new_topo_view, dataset.view_converter.axes)
            dataset.y = new_targets
        else:
            pass
    
    def _create_translated_examples(self, topo_view, targets):
        # translate back and forward one timestep
        # after the first column (always leave first timestep intact
        # as it should be important for baselining) 
        # pad by repeating last / second timestep 
        start_bin = self.start_bin
        stop_bin = self.stop_bin
        # backwards shift:
        # remove start_bin, duplicate stop_bin
        # until before start bin +
        # after start bin to stop bin + stop bin to end
        translated_backwards = np.concatenate((
                                topo_view[:, :, 0:start_bin,:], 
                                topo_view[:, :, start_bin + 1:stop_bin + 1,:], 
                                topo_view[:,:, stop_bin:, :]), 
                                axis=2)

        # forwards shift:
        # remove stop bin, duplicate start bin
        # until start bin, start bin to before stop bin, after stop bin
        # second column + from second column (repeated(!)) to second-last column
        translated_forwards = np.concatenate((
                                    topo_view[:, :, 0:start_bin + 1,:], 
                                    topo_view[:,:, start_bin:stop_bin,:],
                                    topo_view[:,:, stop_bin + 1:,:]), 
                                    axis=2)
        translated_examples = np.concatenate((translated_backwards,
                                                 translated_forwards))
        # targets remain the same, just replicate...
        targets = np.concatenate((targets, targets))
        return translated_examples, targets
            
class SmoothingPreprocessor(Preprocessor):
    """ Smooth spectrogram along the time axis"""
    def __init__(self, start_bin, stop_bin, filter_deviation):
        # assume 1-based indexing given
        self.start_bin = start_bin - 1
        self.stop_bin = stop_bin - 1
        self._filter_deviation = filter_deviation

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        if can_fit:
            topo_view = dataset.get_topological_view()
            topo_view = np.copy(topo_view)
            targets = np.copy(dataset.y)
            smoothed_examples = self._create_smoothed_examples(
               topo_view)
            new_topo_view = np.concatenate((topo_view, 
                                               smoothed_examples))
            new_targets = np.concatenate((targets, targets))
            dataset.set_topological_view(new_topo_view, dataset.view_converter.axes)
            dataset.y = new_targets
        else:
            pass
    
    def _create_smoothed_examples(self, topo_view):
        # smooth only along time axis third axis = 2
        smoothed = np.copy(topo_view)
        smoothed[:,:,self.start_bin:self.stop_bin,:] = \
            scipy.ndimage.filters.gaussian_filter(
                topo_view[:,:,self.start_bin:self.stop_bin,:],
                sigma=(0,0,self._filter_deviation,0))
        
        return smoothed

class RemoveTrialChannelMean(Preprocessor):
    """ Remove all channel means from all trials """
    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        # too lazy to do this for different axes orderings :)
        zero_axis = dataset.view_converter.axes.index(0)
        one_axis = dataset.view_converter.axes.index(1)
        
        topo_view = dataset.get_topological_view()
        mean = np.mean(topo_view, axis=(zero_axis,one_axis), keepdims=True)
        new_topo_view = topo_view - mean
        dataset.set_topological_view(new_topo_view, 
            dataset.view_converter.axes)
    

class Standardize(Preprocessor):

    """
    Subtracts the mean and divides by the standard deviation.
    Difference to pylearn: std_eps is not added but max taken of eps and actual std
    Parameters
    ----------
    global_mean : bool, optional
        If `True`, subtract the (scalar) mean over every element
        in the design matrix. If `False`, subtract the mean from
        each column (feature) separately. Default is `False`.
    global_std : bool, optional
        If `True`, after centering, divide by the (scalar) standard
        deviation of every element in the design matrix. If `False`,
        divide by the column-wise (per-feature) standard deviation.
        Default is `False`.
    std_eps : float, optional
        Stabilization factor added to the standard deviations before
        dividing, to prevent standard deviations very close to zero
        from causing the feature values to blow up too much.
        Default is `1e-4`.
    """

    def __init__(self, global_mean=False, global_std=False, std_eps=1e-4):
        self._global_mean = global_mean
        self._global_std = global_std
        self._std_eps = std_eps
        self._mean = None
        self._std = None

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        X = dataset.get_design_matrix()
        if can_fit:
            self._mean = X.mean() if self._global_mean else X.mean(axis=0)
            self._std = X.std() if self._global_std else X.std(axis=0)
        else:
            if self._mean is None or self._std is None:
                raise ValueError("can_fit is False, but Standardize object "
                                 "has no stored mean or standard deviation")
        new = (X - self._mean) / np.maximum(self._std_eps, self._std)
        dataset.set_design_matrix(new)

class OnlineStandardize(Preprocessor):

    """
    Subtracts the mean and divides by the standard deviation.
    For the non-fittable datasets, goes over dataset trial by trial
    and uses data of trials seen so far to standardize next trial.
    ----------
    global_mean : bool, optional
        If `True`, subtract the (scalar) mean over every element
        in the design matrix. If `False`, subtract the mean from
        each column (feature) separately. Default is `False`.
    global_std : bool, optional
        If `True`, after centering, divide by the (scalar) standard
        deviation of every element in the design matrix. If `False`,
        divide by the column-wise (per-feature) standard deviation.
        Default is `False`.
    std_eps : float, optional
        Stabilization factor added to the standard deviations before
        dividing, to prevent standard deviations very close to zero
        from causing the feature values to blow up too much.
        Default is `1e-4`.
    new_factor: float, optional
        Factor by how much to weight new variance/means higher, i.e.
        2 means new variances are weighted twice as high as old ones
        Default is 1
    use_only_new: boolean, optional
        Use only the new variances/means for standardization, ignore old values
        from training set
        Default is False
    """

    def __init__(self, global_mean=False, global_std=False, std_eps=1e-4,
        new_factor=1, use_only_new=False):
        self._global_mean = global_mean
        self._global_std = global_std
        self._std_eps = std_eps
        self._mean = None
        self._std = None
        self._new_factor = new_factor
        self._use_only_new = use_only_new

    def apply(self, dataset, can_fit=False, new_factor=1):
        X = dataset.get_design_matrix()
        if can_fit:
            self._mean = X.mean() if self._global_mean else X.mean(axis=0)
            self._std = X.std() if self._global_std else X.std(axis=0)
            self._num_old_examples = X.shape[0]
            newX = (X - self._mean) / (self._std_eps + self._std)
            dataset.set_design_matrix(newX)
        else:
            if self._mean is None or self._std is None:
                raise ValueError("can_fit is False, but Standardize object "
                                 "has no stored mean or standard deviation")
            newX = deepcopy(X)
            for trial_i in xrange(X.shape[0]):
                combined_mean = self._mean
                combined_std = self._std
                if (trial_i > 0): # need two trials to compute a variance :)
                    num_new = trial_i + 1
                    num_old = self._num_old_examples
                    old_mean = self._mean
                    old_std = self._std
                    new_mean = X[0:num_new].mean(axis=0)
                    assert np.all(np.equal(new_mean.shape, old_mean.shape))
                    new_std = X[0:num_new].std(axis=0)
                    assert np.all(np.equal(new_std.shape, old_std.shape))
                    # test: weigh new nums higher: 
                    num_new = round(num_new * self._new_factor)
                    if (not self._use_only_new):
                        combined_mean = compute_combined_mean(num_old,
                            num_new, old_mean, new_mean)
                        combined_std = compute_combined_std(num_old, num_new,
                            old_mean, new_mean, combined_mean, old_std, new_std)
                    else:
                        combined_mean = new_mean
                        combined_std = new_std
                newX[trial_i] = (newX[trial_i] - combined_mean) / (self._std_eps + combined_std)
            dataset.set_design_matrix(newX)
            
        

class ChannelwiseStandardize(Preprocessor):

    """
    Subtracts the mean and divides by the standard deviation channel-wise.
    Difference to pylearn: channel-wise standardization(!), so only compute
    means and standard deviations of channels

    Parameters
    ----------
    std_eps : float, optional
        Stabilization factor added to the standard deviations before
        dividing, to prevent standard deviations very close to zero
        from causing the feature values to blow up too much.
        Default is `1e-4`.
    """

    def __init__(self, std_eps=1e-4):
        self._std_eps = std_eps
        self._mean = None
        self._std = None

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        topo_view = dataset.get_topological_view()
        if can_fit:
            # assuming channel is on axis 1, so only keeping this axis,
            # reducing all others
            self._mean, self._std = self.channelwise_mean_std(dataset)
        else:
            if self._mean is None or self._std is None:
                raise ValueError("can_fit is False, but Standardize object "
                                 "has no stored mean or standard deviation")
        new_topo_view = (topo_view - self._mean) / (self._std_eps + self._std)
        dataset.set_topological_view(new_topo_view,  dataset.view_converter.axes)
        
    @staticmethod
    def channelwise_mean_std(dataset):
        topo_view = dataset.get_topological_view()
        axes = dataset.view_converter.axes
        channel_dim_i = axes.index('c')
        mean,std = ChannelwiseStandardize.axeswise_mean_std(topo_view, 
            channel_dim_i)
        return mean, std
        
    
    @staticmethod
    def axeswise_mean_std(topo_view, channel_dim_i):
        other_dims = range(topo_view.ndim)
        other_dims.remove(channel_dim_i)
        other_dims = tuple(other_dims)
        # assuming channel is on axis 1, so only keeping this axis,
        # reducing all others
        mean = topo_view.mean(axis=other_dims, keepdims=True)
        std = topo_view.std(axis=other_dims, keepdims=True)
        return mean,std
        
    
class SplitTrials(Preprocessor):
    """ Split each trial into several subtrials, e.g.
    4 second trial into three 2 second trials with 1 second overlap """
    
    def apply(self, dataset, can_fit=False):
        assert tuple(dataset.view_converter.axes) == ('b', 'c', 0, 1)
        topo_view = dataset.get_topological_view()
        """
        new_topo_view = np.concatenate((topo_view[:,:,0:300,:], 
                                        topo_view[:,:,150:450,:], 
                                        topo_view[:,:,300:600,:]))"""
        new_topo_view = np.concatenate((topo_view[:,:,120:420,:],
                                        topo_view[:,:,150:450,:],
                                        topo_view[:,:,180:480,:],
                                        topo_view[:,:,210:510,:],
                                        topo_view[:,:,240:540,:], 
                                        topo_view[:,:,270:570,:], 
                                        topo_view[:,:,300:600,:]))
        new_y = np.concatenate((dataset.y, dataset.y, dataset.y, 
            dataset.y, dataset.y, dataset.y, dataset.y))
        dataset.set_topological_view(new_topo_view, 
            dataset.view_converter.axes)
        dataset.y = new_y
    
class OnlineAxiswiseStandardize(Preprocessor):
    """
    Subtracts the mean and divides by the standard deviation, axiswise.
    For the non-fittable datasets, goes over dataset trial by trial
    and uses data of trials seen so far to standardize next trial.
    ----------
    axis: sequence, optional
    axis over which to do the standardization, 
        e.g. ('c', 0) means to compute means and variances for all 
        channelx0 points and subtract/divide by them 
        
        Default is ('c', 0, 1) => all axes (order of axes do not matter), each feature
        is standardized individually

    std_eps : float, optional
        Stabilization factor added to the standard deviations before
        dividing, to prevent standard deviations very close to zero
        from causing the feature values to blow up too much.
        Default is `1e-4`.
    """

    def __init__(self, axis=('c', 0, 1), std_eps=1e-5):
        self._std_eps = std_eps
        self._mean = None
        self._std = None
        self.axis = axis

    def apply(self, dataset, can_fit=False):
        topo_view = dataset.get_topological_view()
        standard_dim_inds = self.determine_standardization_dims(
            dataset.view_converter.axes)
        if can_fit:
            self._mean = np.mean(topo_view, axis=standard_dim_inds,
                 keepdims=True)
            self._std = np.std(topo_view, axis=standard_dim_inds,
                 keepdims=True)
            self._num_old_examples = topo_view.shape[0]
            newTopo = (topo_view - self._mean) / (self._std + self._std_eps)
            dataset.set_topological_view(newTopo, dataset.view_converter.axes)
        else:
            if self._mean is None or self._std is None:
                raise ValueError("can_fit is False, but Standardize object "
                                 "has no stored mean or standard deviation")
            
            assert dataset.view_converter.axes[0] == 'b', ("batch axis should "
                "be first")
            dims_to_squash = None
            if len(standard_dim_inds) > 1:
                # first dim is 'b'-> batch, so needs to be ignored here :))
                # have to subtract 1 since batch dim is gone inside online
                # standardize function
                dims_to_squash = tuple(np.array(standard_dim_inds[1:]) - 1)
            new_topo = online_standardize(topo_view,
                old_mean=self._mean,
                old_std=self._std,
                dims_to_squash=dims_to_squash,
                n_old_trials=self._num_old_examples,
                std_eps=self._std_eps)
            
            dataset.set_topological_view(new_topo, 
                dataset.view_converter.axes)
    
    def determine_standardization_dims(self, dataset_axes):
        """
        >>> preproc = OnlineAxiswiseStandardize(axis=['c',0])
        >>> preproc.determine_standardization_dims(['b','c',0,1])
        (0, 3)
        >>> preproc.determine_standardization_dims(['b',0,1,'c'])
        (0, 2)
        >>> preproc = OnlineAxiswiseStandardize(axis=['c'])
        >>> preproc.determine_standardization_dims(['b','c',0,1])
        (0, 2, 3)
        >>> preproc.determine_standardization_dims(['b',0,1,'c'])
        (0, 1, 2)
        """
        standard_axes = list(self.axis)
        assert 'b' not in standard_axes, ("Just haven't thought abt this case,"
            " if needed assertion can probably be removed")
        for ax in standard_axes:
            assert ax in standard_axes, "all standardization axis should exist"
        # these dims will be given to std/mean functions and
        # therefore removed
        unwanted_dims = [dim_i for dim_i,axis in enumerate(dataset_axes) \
            if axis not in standard_axes]
        return tuple(unwanted_dims)
    
# adapted from https://subluminal.wordpress.com/2008/07/31/running-standard-deviations/
# using degree of freedom 0 which is default for numpy std
# and also simplifies calculations even more
def online_standardize(topo, n_old_trials, old_mean, old_std, dims_to_squash=None, std_eps=1e-5):
    n = n_old_trials
    mean = old_mean
    power_avg = (old_std * old_std + old_mean * old_mean)
    new_topo = deepcopy(topo)
    for i_trial in xrange(len(topo)):
        n += 1
        this_topo = topo[i_trial]
        if dims_to_squash is not None:
            this_topo = np.mean(this_topo, axis=tuple(dims_to_squash), keepdims=True)
        mean = mean + ((this_topo - mean) / n)
        power_avg = power_avg + ((this_topo * this_topo - power_avg) / n)
        std = np.sqrt((power_avg - mean * mean))
        new_topo[i_trial] = ((topo[i_trial] - mean) / (std_eps + std))
    return new_topo    

class OnlineChannelwiseStandardize(ChannelwiseStandardize):
    """
    Subtracts the mean and divides by the standard deviation, channelwise.
    For the non-fittable datasets, goes over dataset trial by trial
    and uses data of trials seen so far to standardize next trial.
    ----------
    global_mean : bool, optional
        If `True`, subtract the (scalar) mean over every element
        in the design matrix. If `False`, subtract the mean from
        each column (feature) separately. Default is `False`.
    global_std : bool, optional
        If `True`, after centering, divide by the (scalar) standard
        deviation of every element in the design matrix. If `False`,
        divide by the column-wise (per-feature) standard deviation.
        Default is `False`.
    std_eps : float, optional
        Stabilization factor added to the standard deviations before
        dividing, to prevent standard deviations very close to zero
        from causing the feature values to blow up too much.
        Default is `1e-4`.
    new_factor: float, optional
        Factor by how much to weight new variance/means higher, i.e.
        2 means new variances are weighted twice as high as old ones
        Default is 1
    use_only_new: boolean, optional
        Use only the new variances/means for standardization, ignore old values
        from training set
        Default is False
    """

    def __init__(self, global_mean=False, global_std=False, std_eps=1e-4,
        new_factor=1, use_only_new=False):
        self._global_mean = global_mean
        self._global_std = global_std
        self._std_eps = std_eps
        self._mean = None
        self._std = None
        self._new_factor = new_factor
        self._use_only_new = use_only_new

    def apply(self, dataset, can_fit=False):
        topo_view = dataset.get_topological_view()
        if can_fit:
            self._mean, self._std = self.channelwise_mean_std(dataset)
            self._num_old_examples = topo_view.shape[0]
            newTopo = (topo_view - self._mean) / (self._std_eps + self._std)
            dataset.set_topological_view(newTopo, dataset.view_converter.axes)
        else:
            if self._mean is None or self._std is None:
                raise ValueError("can_fit is False, but Standardize object "
                                 "has no stored mean or standard deviation")
            newTopo = deepcopy(topo_view)
            axes = dataset.view_converter.axes
            assert axes[0] == 'b', "batch axis should be first"
            channel_dim_i = axes.index('c')
            for trial_i in xrange(newTopo.shape[0]):
                combined_mean = self._mean
                combined_std = self._std
                if (trial_i > 0): # need two trials to compute a variance :)
                    num_new = trial_i + 1
                    num_old = self._num_old_examples
                    old_mean = self._mean
                    old_std = self._std
                    new_mean, new_std = self.axeswise_mean_std(
                        topo_view[0:num_new], channel_dim_i)
                    assert np.all(np.equal(new_mean.shape, old_mean.shape))
                    assert np.all(np.equal(new_std.shape, old_std.shape))
                    # test: weigh new nums higher: 
                    num_new = round(num_new * self._new_factor)
                    if (not self._use_only_new):
                        combined_mean = compute_combined_mean(num_old,
                            num_new, old_mean, new_mean)
                        combined_std = compute_combined_std(num_old, num_new,
                            old_mean, new_mean, combined_mean, old_std, new_std)
                    else:
                        combined_mean = new_mean
                        combined_std = new_std
                newTopo[trial_i] = ((newTopo[trial_i] - combined_mean) /
                    (self._std_eps + combined_std))
            dataset.set_topological_view(newTopo, dataset.view_converter.axes)
            


class TrialwiseStandardize(Preprocessor):
    """
    Subtracts the mean and divides by the standard deviation channel-wise.
    Difference to pylearn: channel-wise standardization(!), so only compute
    means and standard deviations of channels

    Parameters
    ----------
    std_eps : float, optional
        Stabilization factor added to the standard deviations before
        dividing, to prevent standard deviations very close to zero
        from causing the feature values to blow up too much.
        Default is `1e-4`.
    """

    def __init__(self, std_eps=1e-4):
        self._std_eps = std_eps

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        topo_view = dataset.get_topological_view()
        assert tuple(dataset.view_converter.axes) == ('b', 'c', 0, 1)
        mean = np.mean(topo_view, axis=(2,3))
        std = np.std(topo_view, axis=(2,3))
        
        mean = mean[:,:, np.newaxis, np.newaxis]
        std = std[:,:, np.newaxis, np.newaxis]
        new_topo_view = (topo_view - mean) / (self._std_eps + std)
        dataset.set_topological_view(new_topo_view,  dataset.view_converter.axes)
        