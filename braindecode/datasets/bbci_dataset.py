import numpy as np
import wyrm.types  
from braindecode.mywyrm.processing import segment_dat_fast
from braindecode.datasets.sensor_positions import sort_topologically
import h5py  
import re

class BBCIDataset(object):
    """ Class to load BBCI Dataset in matlab format """
    def __init__(self, filename, sensor_names=None,
        cnt_preprocessors=[], epo_preprocessors=[],
        segment_ival=[0,4000]):
        """ Constructor will not call superclass constructor yet"""
        self.__dict__.update(locals())
        del self.self
        self._data_not_loaded_yet = True # needed for lazy loading

    def load(self):
        """ This function actually loads the data. Will be called by the 
        get dataset lazy loading function""" 
        # TODELAY: Later switch to a wrapper dataset for all files
        self.load_continuous_signal()
        self.add_markers()
        self.preprocess_continuous_signal()
        self.segment_into_trials()
        self.remove_continuous_signal() # not needed anymore
        self.preprocess_trials()

    def load_continuous_signal(self):
        wanted_chan_inds, wanted_sensor_names = self.determine_sensors()
        fs = self.determine_samplingrate()
        with h5py.File(self.filename, 'r') as h5file:
            samples = int(h5file['nfo']['T'][0,0])
            cnt_signal_shape = (samples, len(wanted_chan_inds))
            continuous_signal = np.empty(cnt_signal_shape, dtype=np.float64)
            for chan_ind_arr, chan_ind_set  in enumerate(wanted_chan_inds):
                chan_set_name = 'ch' + str(chan_ind_set + 1)
                # first 0 to unpack into vector, before it is 1xN matrix
                chan_signal = h5file[chan_set_name][0,:] # already load into memory
                continuous_signal[:, chan_ind_arr] = chan_signal
            samplenumbers = np.array(range(continuous_signal.shape[0]))
            timesteps_in_ms = samplenumbers * 1000.0 / fs
        cnt = wyrm.types.Data(continuous_signal, 
            [timesteps_in_ms, wanted_sensor_names],
            ['time', 'channel'], 
            ['ms', '#'])
        cnt.fs = fs
        self.cnt = cnt

    def determine_sensors(self):
        #TODELAY: change to only taking filename? maybe more 
        # clarity where file is opened
        with h5py.File(self.filename, 'r') as h5file:
            clab_set = h5file['dat']['clab'][:,0]
            all_sensor_names = [''.join(chr(c) for c in h5file[obj_ref]) for \
                obj_ref in clab_set]
            if self.sensor_names is None:
                # if no sensor names given, take all EEG-chans
                EEG_sensor_names = filter(lambda s: not s.startswith('E'), all_sensor_names)
                # sort sensors topologically to allow networks to exploit topology
                self.sensor_names = sort_topologically(EEG_sensor_names)
            chan_inds = self.determine_chan_inds(all_sensor_names, 
                self.sensor_names)
        return chan_inds, self.sensor_names

    
    def determine_samplingrate(self):
        with h5py.File(self.filename, 'r') as h5file:
            fs =  h5file['dat']['fs'][0,0]
            assert isinstance(fs, int) or fs.is_integer()
            fs = int(fs)
        return fs

    @staticmethod
    def determine_chan_inds(all_sensor_names, sensor_names):
        assert sensor_names is not None
        chan_inds = [all_sensor_names.index(s) for s in sensor_names]
        assert len(chan_inds) == len(sensor_names), ("All"
            "sensors should be there.")
        assert len(set(chan_inds)) == len(chan_inds), ("No"
            "duplicated sensors wanted.")
        return chan_inds
    
    def add_markers(self):
        with h5py.File(self.filename, 'r') as h5file:
            event_times_in_ms = h5file['mrk']['time'][:][:,0]
            event_classes = h5file['mrk']['event']['desc'][:][0]
        # expect epoched set with always 2000 samples per epoch 
        # compare to matlab samples from tonio lab
        self.cnt.markers =  zip(event_times_in_ms, event_classes)

    def preprocess_continuous_signal(self):
        for func, kwargs in self.cnt_preprocessors:
            self.cnt = func(self.cnt, **kwargs)

    def segment_into_trials(self):
        # adding the numbers at start to force later sort in segment_dat
        # to sort them in given order
        self.epo = segment_dat_fast(self.cnt, 
            marker_def={'1 - Right Hand': [1], '2 - Left Hand': [2], 
                '3 - Rest': [3], '4 - Feet': [4]}, 
            ival=self.segment_ival)

    def remove_continuous_signal(self):
        del self.cnt

    def preprocess_trials(self):
        for func, kwargs in self.epo_preprocessors:
            self.epo = func(self.epo, **kwargs)

    @staticmethod
    def get_all_sensors(filename, pattern):
        # TODELAY: split into two methods?
        with h5py.File(filename, 'r') as h5file:
            clab_set = h5file['dat']['clab'][:,0]
            all_sensor_names = [''.join(chr(c) for c in h5file[obj_ref]) for obj_ref in clab_set]
            if pattern is not None:
                all_sensor_names = filter(lambda sname: re.search(pattern, sname), 
                    all_sensor_names)
        return all_sensor_names