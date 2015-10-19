import numpy as np
import wyrm.types
from braindecode.mywyrm.processing import segment_dat_fast
from braindecode.datasets.sensor_positions import sort_topologically
import h5py
import re

class ProcessedDataset(object):
    """ Class to load BBCI Dataset in matlab format """
    def __init__(self, wyrm_set, sensor_names=None,
            cnt_preprocessors=[], epo_preprocessors=[],
            segment_ival=[0,4000], 
            marker_def={'1': [1], '2': [2], '3': [3], '4': [4]}):
        """ Constructor will not call superclass constructor yet"""
        self.__dict__.update(locals())
        del self.self
        self._data_not_loaded_yet = True # needed for lazy loading

    def load(self):
        """ This function actually loads the data. Will be called by the 
        get dataset lazy loading function""" 
        # TODELAY: Later switch to a wrapper dataset for all files
        self.load_signal_and_markers()
        self.preprocess_continuous_signal()
        self.segment_into_trials()
        self.remove_continuous_signal() # not needed anymore
        self.preprocess_trials()

    def load_signal_and_markers(self):
        self.cnt = self.wyrm_set.load()

    def preprocess_continuous_signal(self):
        for func, kwargs in self.cnt_preprocessors:
            self.cnt = func(self.cnt, **kwargs)

    def segment_into_trials(self):
        # adding the numbers at start to force later sort in segment_dat
        # to sort them in given order
        self.epo = segment_dat_fast(self.cnt, 
            ival=self.segment_ival,
            marker_def=self.marker_def)

    def remove_continuous_signal(self):
        del self.cnt

    def preprocess_trials(self):
        for func, kwargs in self.epo_preprocessors:
            self.epo = func(self.epo, **kwargs)

class BBCIDataset(object):
    def __init__(self, filename, load_sensor_names):
        assert load_sensor_names is not None
        self.__dict__.update(locals())
        del self.self

    def load(self):
        """ This function actually loads the data. Will be called by the 
        get dataset lazy loading function""" 
        # TODELAY: Later switch to a wrapper dataset for all files
        cnt = self.load_continuous_signal()
        self.add_markers(cnt)
        return cnt

    def load_continuous_signal(self):
        wanted_chan_inds, wanted_sensor_names = self.determine_sensors()
        print wanted_chan_inds
        fs = self.determine_samplingrate()
        with h5py.File(self.filename, 'r') as h5file:
            samples = int(h5file['nfo']['T'][0,0])
            cnt_signal_shape = (samples, len(wanted_chan_inds))
            continuous_signal = np.empty(cnt_signal_shape, dtype=np.float64)
            for chan_ind_arr, chan_ind_set  in enumerate(wanted_chan_inds):
                chan_set_name = 'ch' + str(chan_ind_set + 1)
                # first 0 to unpack into vector, before it is 1xN matrix
                chan_signal = h5file[chan_set_name][0,:] # already load into memory
                
                # replace nans by mean value
                continuous_signal[:, chan_ind_arr] = chan_signal
            samplenumbers = np.array(range(continuous_signal.shape[0]))
            timesteps_in_ms = samplenumbers * 1000.0 / fs
        cnt = wyrm.types.Data(continuous_signal, 
            [timesteps_in_ms, wanted_sensor_names],
            ['time', 'channel'], 
            ['ms', '#'])
        cnt.fs = fs
        return cnt

    def determine_sensors(self):
        #TODELAY: change to only taking filename? maybe more 
        # clarity where file is opened
        all_sensor_names = self.get_all_sensors(self.filename, pattern=None)
        chan_inds = self.determine_chan_inds(all_sensor_names, 
            self.load_sensor_names)
        return chan_inds, self.load_sensor_names

    
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
    
    def add_markers(self, cnt):
        with h5py.File(self.filename, 'r') as h5file:
            event_times_in_ms = h5file['mrk']['time'][:,0]
            event_classes = h5file['mrk']['event']['desc'][0]
        # expect epoched set with always 2000 samples per epoch 
        # compare to matlab samples from tonio lab
        cnt.markers =  zip(event_times_in_ms, event_classes)

    @staticmethod
    def get_all_sensors(filename, pattern):
        # TODELAY: split into two methods?
        with h5py.File(filename, 'r') as h5file:
            clab_set = h5file['dat']['clab'][:,0]
            # if saved in matlab, sensor names will be object referecnes
            # otherwise they are stored as strings and directly retrievable
            try:
                all_sensor_names = [''.join(chr(c) 
                    for c in h5file[obj_ref]) for obj_ref in clab_set]
            except KeyError: 
                all_sensor_names = clab_set.tolist()
            if pattern is not None:
                all_sensor_names = filter(lambda sname: re.search(pattern, sname), 
                    all_sensor_names)
        return all_sensor_names
    
class BCICompetition4Set2A(object):
    def __init__(self, filename, load_sensor_names=None):
        """ Constructor will not call superclass constructor yet"""
        self.__dict__.update(locals())
        del self.self

    def load(self):
        """ This function actually loads the data. Will be called by the 
        get dataset lazy loading function""" 
        with h5py.File(self.filename, 'r') as h5file:
            cnt_signal = np.float32(h5file['signal'])
            last_eeg_chan = 22 # remaining are eog chans
            eeg_signal = cnt_signal[:last_eeg_chan,:].T 
            # replace nans
            eeg_signal[np.isnan(eeg_signal)] = np.nanmean(eeg_signal)
        
            chan_names = [''.join(chr(c) for c in h5file[obj_ref]) for 
                            obj_ref in h5file['header']['Label'][0,:]]
            assert np.array_equal(['EEG-Fz', 'EEG', 'EEG', 'EEG', 'EEG', 'EEG',
                                   'EEG', 'EEG-C3', 'EEG', 'EEG-Cz', 'EEG', 
                                   'EEG-C4', 'EEG', 'EEG', 'EEG', 'EEG', 'EEG', 
                                   'EEG', 'EEG', 'EEG-Pz', 'EEG', 'EEG', 
                                   'EOG-left', 'EOG-central', 'EOG-right'],
                                 chan_names)
        
            fs = np.int(h5file['header']['EVENT']['SampleRate'][:])
        
            classes = h5file['header']['Classlabel'][0,:].astype(np.int32)
            event_types = h5file['header']['EVENT']['TYP'][0,:]
            assert ((np.intersect1d(event_types,  [769,770,771,772]).size == 0) 
                or not np.any(event_types == 783)), ("Either only known only "
                "unknown events")
            trial_mask = np.array([ev in [769, 770, 771, 772, 783] for ev in event_types])
            if not np.any(event_types == 783): # chekc labels correct
                assert np.array_equal(event_types[trial_mask] - 768, classes)
            start_samples = h5file['header']['EVENT']['POS'][0,:].astype(np.int64)
            trial_start_samples = start_samples[trial_mask]
            trial_start_times = trial_start_samples * (1000.0 / fs) 
            markers = zip(trial_start_times, classes)
            samplenumbers = np.array(range(eeg_signal.shape[0]))
            timesteps_in_ms = samplenumbers * 1000.0 / fs
            
            wanted_sensor_names = chan_names[:last_eeg_chan]
            cnt = wyrm.types.Data(eeg_signal, 
                        [timesteps_in_ms, wanted_sensor_names],
                        ['time', 'channel'], 
                        ['ms', '#'])
            cnt.fs = fs
            cnt.markers = markers
        return cnt    