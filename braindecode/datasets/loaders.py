import os.path
import re
import numpy as np
import h5py
import wyrm.types
import logging
log = logging.getLogger(__name__)
from braindecode.datasets.sensor_positions import sort_topologically
from wyrm.processing import append_cnt


class BBCIDataset(object):
    def __init__(self, filename, load_sensor_names=None):
        """ Constructor will not call superclass constructor yet"""
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
        fs = self.determine_samplingrate()
        with h5py.File(self.filename, 'r') as h5file:
            samples = int(h5file['nfo']['T'][0,0])
            cnt_signal_shape = (samples, len(wanted_chan_inds))
            continuous_signal = np.ones(cnt_signal_shape, dtype=np.float32) * np.nan
            for chan_ind_arr, chan_ind_set  in enumerate(wanted_chan_inds):
                # + 1 because matlab/this hdf5-naming logic
                # has 1-based indexing
                # i.e ch1,ch2,....
                chan_set_name = 'ch' + str(chan_ind_set + 1)
                # first 0 to unpack into vector, before it is 1xN matrix
                chan_signal = h5file[chan_set_name][0,:] # already load into memory
                continuous_signal[:, chan_ind_arr] = chan_signal
            samplenumbers = np.array(range(continuous_signal.shape[0]))
            timesteps_in_ms = samplenumbers * 1000.0 / fs
            assert not np.any(np.isnan(continuous_signal)), "No NaNs expected in signal"
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
        if self.load_sensor_names is None:
            # if no sensor names given, take all EEG-chans
            EEG_sensor_names = filter(lambda s: not s.startswith('E'), all_sensor_names)
            EEG_sensor_names = filter(lambda s: not s.startswith('Microphone'), EEG_sensor_names)
            EEG_sensor_names = filter(lambda s: not s.startswith('Breath'), EEG_sensor_names)
            EEG_sensor_names = filter(lambda s: not s.startswith('GSR'), EEG_sensor_names)
            assert (len(EEG_sensor_names) == 128 or
                len(EEG_sensor_names) == 64 or
                len(EEG_sensor_names) == 32 or 
                len(EEG_sensor_names) == 16), (
                "Recheck this code if you have different sensors...")
            # sort sensors topologically to allow networks to exploit topology
            self.load_sensor_names = sort_topologically(EEG_sensor_names)
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
            
            # Check whether class names known and correct order
            class_name_set = h5file['nfo']['className'][:,0]
            all_class_names = [''.join(chr(c) for c in h5file[obj_ref]) 
                for obj_ref in class_name_set]
            if all_class_names == ['Right Hand', 'Left Hand', 'Rest', 'Feet']:
                pass
            elif ((all_class_names == ['1', '10', '11', '111', '12', '13', '150',
                '2', '20', '22', '3', '30', '33', '4', '40', '44', '99']) or 
                  (all_class_names == ['1', '10', '11', '12', '13', '150', 
                       '2', '20', '22', '3', '30', '33', '4', '40', '44', '99']) or
                  (all_class_names == ['1', '2', '3', '4'])):
                pass # Semantic classes
            elif  all_class_names == ['Rest', 'Feet', 'Left Hand', 'Right Hand']:
                # Have to swap from
                # ['Rest', 'Feet', 'Left Hand', 'Right Hand']
                # to
                # ['Right Hand', 'Left Hand', 'Rest', 'Feet']
                right_mask = event_classes == 4
                left_mask = event_classes == 3
                rest_mask = event_classes == 1
                feet_mask = event_classes == 2
                event_classes[right_mask] = 1
                event_classes[left_mask] = 2
                event_classes[rest_mask] = 3
                event_classes[feet_mask] = 4
            elif all_class_names == ['Right Hand Start', 'Left Hand Start',
                'Rest Start', 'Feet Start', 'Right Hand End',
                'Left Hand End', 'Rest End', 'Feet End']:
                pass
            elif all_class_names == ['Right Hand', 'Left Hand', 'Rest',
                'Feet', 'Face', 'Navigation', 'Music', 'Rotation',
                'Subtraction', 'Words']:
                pass # robot hall 10 class decoding
            elif (all_class_names == ['RightHand', 'Feet', 'Rotation', 'Words',
                '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00',
                'RightHand_End', '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00',
                '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00',
                'Feet_End', '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00',
                '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00',
                'Rotation_End', '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00',
                '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00',
                'Words_End'] or
                all_class_names == ['RightHand', 'Feet', 'Rotation', 'Words',
                    'Rest', '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00',
                    'RightHand_End', '\x00\x00', '\x00\x00', '\x00\x00',
                    '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00',
                    '\x00\x00', 'Feet_End', '\x00\x00', '\x00\x00', '\x00\x00',
                    '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00',
                    '\x00\x00', 'Rotation_End', '\x00\x00', '\x00\x00',
                    '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00',
                    '\x00\x00', '\x00\x00', 'Words_End', '\x00\x00', '\x00\x00',
                    '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00',
                    '\x00\x00', '\x00\x00', 'Rest_End']):
                pass # weird stuff when we recorded cursor in robot hall
                    # on 2016-09-14 and 2016-09-16 :D
            
            elif len(event_times_in_ms) ==  len(all_class_names):
                pass # weird neuroone(?) logic where class names have event classes
            else:
                # add another clause here for other class names...
                raise ValueError("Unknown class names {:s}".format(
                    all_class_names))
            
        cnt.markers =  zip(event_times_in_ms, event_classes)
        cnt.class_names = all_class_names

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
            chan_names = [''.join(chr(c) for c in h5file[obj_ref]) for 
                            obj_ref in h5file['header']['Label'][0,:]]
            assert np.array_equal(['EEG-Fz', 'EEG', 'EEG', 'EEG', 'EEG', 'EEG',
                                   'EEG', 'EEG-C3', 'EEG', 'EEG-Cz', 'EEG', 
                                   'EEG-C4', 'EEG', 'EEG', 'EEG', 'EEG', 'EEG', 
                                   'EEG', 'EEG', 'EEG-Pz', 'EEG', 'EEG', 
                                   'EOG-left', 'EOG-central', 'EOG-right'],
                                 chan_names)
            last_eeg_chan = 22 # remaining are eog chans
            if self.load_sensor_names is None:
                # Assume all eeg sensors wanted
                eeg_signal = cnt_signal[:last_eeg_chan,:].T 
                wanted_signal = eeg_signal
                wanted_sensor_names = chan_names[:last_eeg_chan]
            else:
                assert (np.array_equal(['EOG-left', 'EOG-central', 'EOG-right'],
                    self.load_sensor_names) or
                    np.array_equal(['C3', 'C4', 'Cz'],
                    self.load_sensor_names)), ("Only implemented loading eeg "
                        " or debug eeg or EOG sensors for now")
                if np.array_equal(self.load_sensor_names, ['C3', 'C4', 'Cz']):
                    self.load_sensor_names = ['EEG-C3', 'EEG-C4', 'EEG-Cz']
                chan_inds = [chan_names.index(name)
                    for name in self.load_sensor_names]
                
                wanted_signal = cnt_signal[chan_inds,:].T
                wanted_sensor_names = np.array(chan_names)[chan_inds].tolist() 
            
            # replace nans  by means of corresponding chans
            for i_chan in xrange(wanted_signal.shape[1]):
                chan_signal = wanted_signal[:, i_chan]
                chan_signal[np.isnan(chan_signal)] = np.nanmean(chan_signal)
                wanted_signal[:, i_chan] = chan_signal
            assert not np.any(np.isnan(wanted_signal))

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
            samplenumbers = np.array(range(wanted_signal.shape[0]))
            timesteps_in_ms = samplenumbers * 1000.0 / fs
                
            cnt = wyrm.types.Data(wanted_signal, 
                        [timesteps_in_ms, wanted_sensor_names],
                        ['time', 'channel'], 
                        ['ms', '#'])
            cnt.fs = fs
            cnt.markers = markers
        return cnt
    
def convert_test_files_add_markers():
    """ Just for documentation purposes put this here ..."""
    for i_subject in xrange(1,10):
        combined_filename = 'data/bci-competition-iv/2a-combined/A0{:d}TE.mat'.format(i_subject)
        test_filename = 'data/bci-competition-iv/2a/A0{:d}E.mat'.format(i_subject)
    
        combined_set = BCICompetition4Set2A(combined_filename)
        combined_cnt = combined_set.load()
    
        test_h5_file = h5py.File(test_filename, 'a')
        class_labels = np.array(combined_cnt.markers)[288:, 1]
        del test_h5_file['header']['Classlabel']
        test_h5_file['header'].create_dataset('Classlabel', data=class_labels[np.newaxis,:].astype(np.float64))
    
        # load and change event type only for class labels
        event_type = test_h5_file['header']['EVENT']['TYP'][:]
        if np.any(event_type[0] == 783):
            event_type[0, event_type[0] == 783] = class_labels.astype(np.float64) + 768
    
        del test_h5_file['header']['EVENT']['TYP']
        test_h5_file['header']['EVENT'].create_dataset('TYP', data=event_type)
    
        test_h5_file.close()
    
    
        test_h5_file = h5py.File(test_filename, 'r')
        assert np.array_equal(class_labels, test_h5_file['header']['Classlabel'][0,:])
    
        event_type_in_file = test_h5_file['header']['EVENT']['TYP'][0,:]
        trial_mask = np.array([ev in [769, 770, 771, 772, 783] for ev in event_type_in_file])
    
        assert np.array_equal(event_type_in_file[trial_mask]- 768, class_labels)
    
        test_h5_file.close()
    
        
    # Final check over datasets
    for i_subject in xrange(1,10):
        combined_filename = 'data/bci-competition-iv/2a-combined/A0{:d}TE.mat'.format(i_subject)
        test_filename = 'data/bci-competition-iv/2a/A0{:d}E.mat'.format(i_subject)
    
        combined_set = BCICompetition4Set2A(combined_filename)
        combined_cnt = combined_set.load()
        test_set = BCICompetition4Set2A(test_filename)
        test_cnt = test_set.load()
        assert np.array_equal(np.array(test_cnt.markers)[:,1],
                   np.array(combined_cnt.markers)[288:,1])

class BCICompetition4Set2aAllSubjects(object):
    """For All Subjects.
    train_or_test should be the string 'train' or the string 'test'"""
    def __init__(self, folder, train_or_test, i_last_subject=9, load_sensor_names=None):
        """ Constructor will not call superclass constructor yet"""
        assert load_sensor_names is None, "Not implemented loading only specific sensors"
        self.folder = folder
        self.train_or_test = train_or_test
        self.load_sensor_names = load_sensor_names
        self.i_last_subject = i_last_subject
        
    def load(self):
        if self.train_or_test == 'train':
            suffix = 'T'
        elif self.train_or_test == 'test':
            suffix = 'E'
        else:
            raise ValueError("Please pass either 'train' or "
                "'test', got {:s}".format(self.train_or_test))
        
        filenames = [os.path.join(self.folder, 'A0{:d}{:s}.mat'.format(
            i_subject,suffix)) 
            for i_subject in xrange(1,self.i_last_subject + 1)]
        cnts = [BCICompetition4Set2A(name).load() for name in filenames]

        combined_cnt = reduce(append_cnt, cnts)
        return combined_cnt

class HDF5StreamedSet(object):
    """Our very own minimalistic format how data is stored when streamed during an online
     experiment."""
    
    def __init__(self, filename, load_sensor_names=None):
        self.__dict__.update(locals())
        del self.self

    def load(self):
        """ This function actually loads the data. Will be called by the 
        get dataset lazy loading function""" 
        with h5py.File(self.filename, 'r') as h5file:
            all_data = np.array(h5file['cnt_samples'])
            sensor_names = h5file['chan_names'][:-1]
            
            if self.load_sensor_names is None:
                cnt_data = all_data[:,:-1]
                wanted_sensor_names = sensor_names
            else:
                log.warn("Load sensor names may lead to different results for "
                    "this set class compared to others")
                chan_inds = [sensor_names.tolist().index(s)
                    for s in self.load_sensor_names]
                cnt_data = all_data[:,chan_inds]
                wanted_sensor_names = self.load_sensor_names
            
            marker = all_data[:,-1]
            fs = 512.0
            time_steps = np.arange(len(cnt_data)) * 1000.0 / fs
            cnt = wyrm.types.Data(cnt_data,axes=[time_steps, 
                wanted_sensor_names], names=['time', 'channel'],
                units=['ms', '#'])
            cnt.fs = fs
            
            # Reconstruct markers
            pause = marker == 0
            boundaries = np.diff(pause)
            inds_boundaries = np.flatnonzero(boundaries)
            # first sample of next class is at inds_boundaries + 1 always..
            event_samples_and_classes = [(i + 1, int(marker[i + 1])) 
                for i in inds_boundaries]
            event_samples_and_classes = [pair for 
                pair in event_samples_and_classes if pair[1] != 0]
            event_ms_and_classes = [((pair[0] * 1000.0 / cnt.fs), pair[1]) for 
                pair in event_samples_and_classes]
            cnt.markers = event_ms_and_classes
        return cnt