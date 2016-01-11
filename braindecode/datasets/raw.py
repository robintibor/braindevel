from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np
from wyrm.processing import select_channels, select_epochs
from braindecode.mywyrm.processing import segment_dat_fast
import logging
from braindecode.datasets.sensor_positions import sort_topologically
from pylearn2.format.target_format import OneHotFormatter
log = logging.getLogger(__name__)

def load_cnt_processed(full_set):
    full_set.load_full_set()
    full_set.determine_clean_trials_and_chans()
    full_set.select_sensors()
    log.info("Preprocessing set...")
    full_set.signal_processor.preprocess_continuous_signal()
    return full_set.signal_processor.cnt

def load_cnt_unprocessed(full_set):
    full_set.load_full_set()
    full_set.determine_clean_trials_and_chans()
    full_set.select_sensors()
    return full_set.signal_processor.cnt

class SignalMatrix(DenseDesignMatrix):
    """ This loads EEG signal datasets and puts them in a Dense Design Matrix.
    Signal processor needs to load """
    def __init__(self, signal_processor,
        sensor_names='all', limits=None, start=None, stop=None,
        axes=('b', 'c', 0, 1),
        unsupervised_preprocessor=None,
        sort_topological=True):

        # sort sensors topologically to allow networks to exploit topology
        if (sensor_names is not None) and (sensor_names  is not 'all') and sort_topological:
            sensor_names = sort_topologically(sensor_names)
        self.__dict__.update(locals())
        del self.self       
        self._data_not_loaded_yet = True # needed for lazy loading
        
    def load(self):
        self.load_signal()
        self.create_dense_design_matrix()
        self.remove_signal_epo()
        if self.unsupervised_preprocessor is not None:
            self.apply_unsupervised_preprocessor()

        # for now format y back to classes
        self.y = np.argmax(self.y, axis=1).astype(np.int32)
        self._data_not_loaded_yet = False # needed for lazy loading
      
    def load_signal(self):  
        raise NotImplementedError("Should not be called anymore, only call"
            " clean dataset loader")

    def create_dense_design_matrix(self):
        # epo has original shape trials x samples x channels x(freq/band?)
        topo_view = self.signal_processor.epo.data.swapaxes(1,2).astype(np.float32)
        # add empty axes if needed
        if topo_view.ndim == 3:
            topo_view = np.expand_dims(topo_view, axis=3)
        topo_view = np.ascontiguousarray(np.copy(topo_view))
        y = self.signal_processor.epo.axes[0] + 1
        other_y = [event_class for time, event_class in self.signal_processor.epo.markers]
        assert np.array_equal(y, other_y[:len(y)]), ("trial axes should"
            "have same event labels as markers "
            "(offset by 1 due to 0 and 1-based indexing), except for out of "
            "bounds trials")

        topo_view, y = self.adjust_for_start_stop_limits(topo_view, y)

        y = format_targets(np.array(y))
        super(SignalMatrix, self).__init__(topo_view=topo_view, y=y, 
                                              axes=self.axes)
        
        log.info("Loaded dataset with shape: {:s}".format(
            str(self.get_topological_view().shape)))
    
    def remove_signal_epo(self):
        """ To save memory, delete bbci set data,
        but keep signal_processor itself to allow reloading. """
        del self.signal_processor.epo

    def adjust_for_start_stop_limits(self, topo_view, y):
        # cant use both limits and start and stop...
        assert(self.limits is None) or \
            (self.start is None and self.stop is None)
        if (self.limits is None):
            start, stop = adjust_start_stop(topo_view.shape[0], 
                self.start, self.stop)
            new_topo_view = topo_view[start:stop]
            new_y = y[start:stop]
        else:
            start = self.limits[0][0]
            stop = self.limits[0][1]
            new_topo_view = topo_view[start:stop]
            new_y = y[start:stop]
            for limit in self.limits[1:]:
                start = limit[0]
                stop = limit[1]
                topo_view_part = topo_view[start:stop]
                y_part = y[start:stop]
                new_topo_view = np.append(new_topo_view, topo_view_part, 0)
                new_y = np.append(new_y, y_part, 0)
        return new_topo_view, new_y

    def apply_unsupervised_preprocessor(self):
        self.unsupervised_preprocessor.apply(self, can_fit=False)
        log.info("Applied unsupervised preprocessing, dataset shape now: {:s}".format(
            str(self.get_topological_view().shape)))


class CleanSignalMatrix(SignalMatrix):
    reloadable=False # No need to reload rawset, it shd be small enough (?)
    def __init__(self, cleaner, **kwargs):
        self.cleaner = cleaner
        super(CleanSignalMatrix, self).__init__(**kwargs)

    def ensure_is_loaded(self):
        if not hasattr(self, 'X'):
            self.load()

    def load_signal(self):  
        """ Only load clean data """
        # Data may be loaded already... e.g. if train set was loaded
        # and now valid set is being loaded as part of same set
        self.clean_and_load_set()
        log.info("Loaded clean data with shape {:s}.".format(
            self.signal_processor.epo.data.shape))           
    
    def clean_and_load_set(self):
        self.load_full_set()
        self.determine_clean_trials_and_chans()
        self.select_sensors()
        self.preproc_and_load_clean_trials()
    
    def load_full_set(self):
        log.info("Loading set...")
        # First load whole set
        self.signal_processor.load_signal_and_markers()
    
    def determine_clean_trials_and_chans(self):
        log.info("Cleaning set...")
        (rejected_chans, rejected_trials, clean_trials) = self.cleaner.clean(
            self.signal_processor.cnt)
        # In case this is not true due to different segment ivals of
        # cleaner and real data, try to standardize variable name for
        # segment ival of cleaner, e.g. to segment_ival
        all_trial_epo = segment_dat_fast(self.signal_processor.cnt,
            self.signal_processor.marker_def,
                 self.signal_processor.segment_ival)
        
        assert np.array_equal(np.union1d(clean_trials, rejected_trials),
            range(all_trial_epo.data.shape[0])), ("All trials should "
                "either be clean or rejected.")
        assert np.intersect1d(clean_trials, rejected_trials).size == 0, ("All "
            "trials should either be clean or rejected.")

        self.rejected_chans = rejected_chans
        self.rejected_trials = rejected_trials
        self.clean_trials = clean_trials # just for later info

    def select_sensors(self):
        if len(self.rejected_chans) > 0:
            self.signal_processor.cnt = select_channels(self.signal_processor.cnt, 
                self.rejected_chans, invert=True)
        if (self.sensor_names is not None) and (self.sensor_names is not 'all'):
            self.signal_processor.cnt = select_channels(
                self.signal_processor.cnt, 
                self.sensor_names)
        cleaned_sensor_names = self.signal_processor.cnt.axes[-1]
        self.sensor_names = cleaned_sensor_names

    def preproc_and_load_clean_trials(self):
        log.info("Preprocessing set...")
        self.signal_processor.preprocess_continuous_signal()
        self.signal_processor.segment_into_trials()
        if len(self.rejected_trials) > 0:
            self.signal_processor.epo = select_epochs(
                self.signal_processor.epo, 
                self.rejected_trials, invert=True)
        # select epochs does not update marker structure...
        clean_markers = [m for i,m in enumerate(self.signal_processor.epo.markers) \
            if i not in self.rejected_trials]
        self.signal_processor.epo.markers = clean_markers
        self.signal_processor.remove_continuous_signal()
        self.signal_processor.preprocess_trials()

def format_targets(y):
    # matlab has one-based indexing and one-based labels
    # have to convert to zero-based labels so subtract 1...
    y = y - 1
    # we need only a 1d-array of integers
    # squeeze in case of 2 dimensions, make sure it is still 1d in case of
    # a single number (can happen for test runs with just one trial)
    y = np.atleast_1d(y.squeeze())
    y = y.astype(int)
    target_formatter = OneHotFormatter(4)
    y = target_formatter.format(y)
    return y

def adjust_start_stop(num_trials, given_start, given_stop):
    # allow to specify start trial as percentage of total dataset
    if isinstance(given_start, float):
        assert given_start >= 0 and given_start <= 1
        given_start = int(num_trials * given_start)
    if isinstance(given_stop, float):
        assert given_stop >= 0 and given_stop <= 1
        # use -1 to ensure that if stop given as e.g. 0.6
        # and next set uses start as 0.6 they are completely
        # seperate/non-overlapping
        given_stop = int(num_trials * given_stop) - 1
    # if start or stop not existing set to 0 and end of dataset :)
    start = given_start or 0
    stop = given_stop or num_trials
    return start, stop