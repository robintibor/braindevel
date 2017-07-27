import os.path
import re
from functools import reduce
import numpy as np
import xarray as xr
import h5py
import logging
from scipy.io.matlab.mio import loadmat
import mne
from braindecode2.datasets.sensor_positions import sort_topologically
from braindecode2.mywyrm.processing import concatenate_cnt

log = logging.getLogger(__name__)


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
            samples = int(h5file['nfo']['T'][0, 0])
            cnt_signal_shape = (samples, len(wanted_chan_inds))
            continuous_signal = np.ones(cnt_signal_shape,
                                        dtype=np.float32) * np.nan
            for chan_ind_arr, chan_ind_set in enumerate(wanted_chan_inds):
                # + 1 because matlab/this hdf5-naming logic
                # has 1-based indexing
                # i.e ch1,ch2,....
                chan_set_name = 'ch' + str(chan_ind_set + 1)
                # first 0 to unpack into vector, before it is 1xN matrix
                chan_signal = h5file[chan_set_name][
                              :].squeeze()  # already load into memory
                continuous_signal[:, chan_ind_arr] = chan_signal
            samplenumbers = np.array(range(continuous_signal.shape[0]))
            timesteps_in_ms = samplenumbers * 1000.0 / fs
            assert not np.any(
                np.isnan(continuous_signal)), "No NaNs expected in signal"

        cnt = xr.DataArray(continuous_signal,
                           coords={'time': timesteps_in_ms,
                                   'channels': wanted_sensor_names, },
                           dims=('time', 'channels'))
        cnt.attrs['fs'] = fs
        return cnt

    def determine_sensors(self):
        # TODELAY: change to only taking filename? maybe more
        # clarity where file is opened
        all_sensor_names = self.get_all_sensors(self.filename, pattern=None)
        if self.load_sensor_names is None:
            # if no sensor names given, take all EEG-chans
            eeg_sensor_names = all_sensor_names
            eeg_sensor_names = filter(lambda s: not s.startswith('BIP'),
                                      eeg_sensor_names)
            eeg_sensor_names = filter(lambda s: not s.startswith('E'),
                                      eeg_sensor_names)
            eeg_sensor_names = filter(lambda s: not s.startswith('Microphone'),
                                      eeg_sensor_names)
            eeg_sensor_names = filter(lambda s: not s.startswith('Breath'),
                                      eeg_sensor_names)
            eeg_sensor_names = filter(lambda s: not s.startswith('GSR'),
                                      eeg_sensor_names)
            eeg_sensor_names = list(eeg_sensor_names)
            assert (len(eeg_sensor_names) == 128 or
                    len(eeg_sensor_names) == 64 or
                    len(eeg_sensor_names) == 32 or
                    len(eeg_sensor_names) == 16), (
                "Recheck this code if you have different sensors...")
            # sort sensors topologically to allow networks to exploit topology
            # this is kpe there to ensure reproducibility,
            # rerunning of old results only
            self.load_sensor_names = sort_topologically(eeg_sensor_names)
        chan_inds = self.determine_chan_inds(all_sensor_names,
                                             self.load_sensor_names)
        return chan_inds, self.load_sensor_names

    def determine_samplingrate(self):
        with h5py.File(self.filename, 'r') as h5file:
            fs = h5file['dat']['fs'][0, 0]
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
            event_times_in_ms = h5file['mrk']['time'][:].squeeze()
            event_classes = h5file['mrk']['event']['desc'][:].squeeze()

            # Check whether class names known and correct order
            class_name_set = h5file['nfo']['className'][:].squeeze()
            all_class_names = [''.join(chr(c) for c in h5file[obj_ref])
                               for obj_ref in class_name_set]
            if all_class_names == ['Right Hand', 'Left Hand', 'Rest', 'Feet']:
                pass
            elif ((all_class_names == ['1', '10', '11', '111', '12', '13',
                                       '150',
                                       '2', '20', '22', '3', '30', '33', '4',
                                       '40', '44', '99']) or
                      (all_class_names == ['1', '10', '11', '12', '13', '150',
                                           '2', '20', '22', '3', '30', '33',
                                           '4', '40', '44', '99']) or
                      (all_class_names == ['1', '2', '3', '4'])):
                pass  # Semantic classes
            elif all_class_names == ['Rest', 'Feet', 'Left Hand', 'Right Hand']:
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
                                     'Rest Start', 'Feet Start',
                                     'Right Hand End',
                                     'Left Hand End', 'Rest End', 'Feet End']:
                pass
            elif all_class_names == ['Right Hand', 'Left Hand', 'Rest',
                                     'Feet', 'Face', 'Navigation', 'Music',
                                     'Rotation',
                                     'Subtraction', 'Words']:
                pass  # robot hall 10 class decoding
            elif (all_class_names == ['RightHand', 'Feet', 'Rotation', 'Words',
                                      '\x00\x00', '\x00\x00', '\x00\x00',
                                      '\x00\x00', '\x00\x00',
                                      'RightHand_End', '\x00\x00', '\x00\x00',
                                      '\x00\x00', '\x00\x00',
                                      '\x00\x00', '\x00\x00', '\x00\x00',
                                      '\x00\x00', '\x00\x00',
                                      'Feet_End', '\x00\x00', '\x00\x00',
                                      '\x00\x00', '\x00\x00',
                                      '\x00\x00', '\x00\x00', '\x00\x00',
                                      '\x00\x00', '\x00\x00',
                                      'Rotation_End', '\x00\x00', '\x00\x00',
                                      '\x00\x00', '\x00\x00',
                                      '\x00\x00', '\x00\x00', '\x00\x00',
                                      '\x00\x00', '\x00\x00',
                                      'Words_End'] or
                          all_class_names == ['RightHand', 'Feet', 'Rotation',
                                              'Words',
                                              'Rest', '\x00\x00', '\x00\x00',
                                              '\x00\x00', '\x00\x00',
                                              'RightHand_End', '\x00\x00',
                                              '\x00\x00', '\x00\x00',
                                              '\x00\x00', '\x00\x00',
                                              '\x00\x00', '\x00\x00',
                                              '\x00\x00',
                                              '\x00\x00', 'Feet_End',
                                              '\x00\x00', '\x00\x00',
                                              '\x00\x00',
                                              '\x00\x00', '\x00\x00',
                                              '\x00\x00', '\x00\x00',
                                              '\x00\x00',
                                              '\x00\x00', 'Rotation_End',
                                              '\x00\x00', '\x00\x00',
                                              '\x00\x00', '\x00\x00',
                                              '\x00\x00', '\x00\x00',
                                              '\x00\x00',
                                              '\x00\x00', '\x00\x00',
                                              'Words_End', '\x00\x00',
                                              '\x00\x00',
                                              '\x00\x00', '\x00\x00',
                                              '\x00\x00', '\x00\x00',
                                              '\x00\x00',
                                              '\x00\x00', '\x00\x00',
                                              'Rest_End']):
                pass  # weird stuff when we recorded cursor in robot hall
                # on 2016-09-14 and 2016-09-16 :D

            elif (all_class_names == ['0004', '0016', '0032', '0056', '0064',
                                      '0088', '0095', '0120']):
                pass
            elif (all_class_names == ['0004', '0056', '0088', '0120']):
                pass
            elif (all_class_names == ['0004', '0016', '0032', '0048', '0056',
                                      '0064', '0080', '0088', '0095', '0120']):
                pass
            elif (all_class_names == ['0004', '0016', '0056', '0088', '0120',
                                      '__']):
                pass
            elif (all_class_names == ['0004', '0056', '0088', '0120', '__']):
                pass
            elif (all_class_names == ['0004', '0032', '0048', '0056', '0064',
                                      '0080', '0088', '0095', '0120', '__']):
                pass
            elif (all_class_names == ['0004', '0056', '0080', '0088', '0096',
                                      '0120', '__']):
                pass
            elif (all_class_names == ['0004', '0032', '0056', '0064', '0080',
                                      '0088', '0095', '0120']):
                pass
            elif (all_class_names == ['0004', '0032', '0048', '0056', '0064',
                                      '0080', '0088', '0095', '0120']):
                pass
            elif (all_class_names == ['0004', '0016', '0032', '0048', '0056',
                                      '0064', '0080', '0088', '0095', '0096',
                                      '0120']):
                pass
            elif (all_class_names == ['4', '16', '32', '56', '64', '88', '95',
                                      '120']):
                pass
            elif (all_class_names == ['4', '56', '88', '120']):
                pass
            elif (all_class_names == ['4', '16', '32', '48', '56',
                                      '64', '80', '88', '95', '120']):
                pass
            elif (all_class_names == ['0', '4', '56', '88', '120']):
                pass
            elif (all_class_names == ['0', '4', '16', '56', '88', '120']):
                pass
            elif (all_class_names == ['0', '4', '32', '48', '56', '64', '80',
                                      '88', '95', '120']):
                pass
            elif (all_class_names == ['0', '4', '56', '80', '88', '96', '120']):
                pass
            elif (all_class_names == ['4', '32', '56', '64', '80', '88', '95',
                                      '120']):
                pass
            elif (all_class_names == ['One', 'Two', 'Three', 'Four']):
                pass
            elif (
                all_class_names == ['1', '10', '11', '12', '2', '20', '3', '30',
                                    '4', '40']):
                pass
            elif (
                all_class_names == ['1', '10', '12', '13', '2', '20', '3', '30',
                                    '4', '40']):
                pass
            elif (
                all_class_names == ['1', '10', '13', '2', '20', '3', '30', '4',
                                    '40', '99']):
                pass
            elif (all_class_names == ['1', '10', '11', '14', '18', '20', '21',
                                      '24', '251', '252', '28', '30', '4',
                                      '8']):
                pass
            elif (all_class_names == ['1', '10', '11', '14', '18', '20', '21',
                                      '24', '252', '253', '28', '30', '4',
                                      '8']):
                pass
            elif len(event_times_in_ms) == len(all_class_names):
                pass  # weird neuroone(?) logic where class names have event classes
            elif (all_class_names == ['Right_hand_stimulus_onset',
                                      'Feet_stimulus_onset',
                                      'Rotation_stimulus_onset',
                                      'Words_stimulus_onset',
                                      'Right_hand_stimulus_offset',
                                      'Feet_stimulus_offset',
                                      'Rotation_stimulus_offset',
                                      'Words_stimulus_offset']):
                pass
            # elif (all_class_names == ['Right hand', 'Feet', 'Rotation', 'Words']):
            #    pass
            else:
                # remove this whole if else stuffs?
                log.warn("Unknown class names {:s}".format(
                    all_class_names))

        event_times_in_samples = np.uint32(np.round(event_times_in_samples))
        cnt.attrs['events'] = np.array(
            list(zip(event_times_in_samples, event_classes)))

    @staticmethod
    def get_all_sensors(filename, pattern):
        # TODELAY: split into two methods?
        with h5py.File(filename, 'r') as h5file:
            clab_set = h5file['dat']['clab'][:].squeeze()
            all_sensor_names = [''.join(chr(c) for c in h5file[obj_ref]) for
                                obj_ref in clab_set]
            if pattern is not None:
                all_sensor_names = filter(
                    lambda sname: re.search(pattern, sname),
                    all_sensor_names)
        return all_sensor_names


# for checks, remove:
#from braindecode.datasets.loaders import BCICompetition4Set2B as BCICompetition4Set2BOld
#cnt_old = BCICompetition4Set2BOld('/home/schirrmr/data/bci-competition-iv/2b/signal/B0101T.hdf5').load()
#cnt_new = BCICompetition4Set2B(filename=filename).load()
#np.allclose(cnt_old.data, cnt_new.data)
class BCICompetition4Set2B(object):
    def __init__(self, filename, load_sensor_names=None, labels_filename=None):
        assert load_sensor_names is None
        self.__dict__.update(locals())
        del self.self

    def load(self):
        raw_edf = mne.io.read_raw_edf(self.filename)
        cnt = self.extract_data(raw_edf)
        cnt.attrs['fs'] = 250.0
        events, artifact_trial_mask = self.extract_events(raw_edf)
        cnt.attrs['events'] = events
        cnt.attrs['artifact_trial_mask'] = artifact_trial_mask
        return cnt

    @staticmethod
    def extract_data(raw_edf):
        cnt = xr.DataArray(raw_edf.get_data().T,
                           coords={'time': raw_edf.times * 1000.0,
                                   'channels': raw_edf.ch_names, },
                           # pd.to_timedelta(raw_edf.times,unit='s')},
                           dims=('time', 'channels'))
        cnt = cnt.sel(channels=['EEG:C3', 'EEG:Cz', 'EEG:C4'])

        # correct NaN Values ...
        # are set to lowest negative number
        data = cnt.values
        for i_chan in range(data.shape[1]):
            # first set to nan, than replace nans by nanmean.
            this_chan = data[:, i_chan]
            data[:, i_chan] = np.where(this_chan == np.min(this_chan),
                                 np.nan, this_chan)
            mask = np.isnan(data[:, i_chan])
            chan_mean = np.nanmean(data[:, i_chan])
            data[mask, i_chan] = chan_mean
        # somehow for correct units as before when loading from matlab,
        # need multiplication by 1e6...
        cnt.values = data * 1e6
        return cnt

    def extract_events(self, raw_edf):
        # all events
        events = np.array(list(zip(
            raw_edf.get_edf_events()[1],
            raw_edf.get_edf_events()[2])))

        # only trial onset events
        trial_events = events[(events[:, 1] == 769) | (events[:, 1] == 770) |
                        (events[:, 1] == 783)]
        assert (len(trial_events) == 120) or (len(trial_events) == 160) or (
            len(trial_events) == 140), (
            "Got {:d} markers".format(len(trial_events)))
        # event markers 769,770 -> 1,2
        trial_events[:, 1] = trial_events[:, 1] - 768
        # possibly overwrite with markers from labels file
        if self.labels_filename is not None:
            classes = loadmat(self.labels_filename)['classlabel'].squeeze()
            trial_events[:, 1] = classes
        unique_classes = np.unique(trial_events[:, 1])
        assert np.array_equal([1, 2], unique_classes), (
            "Expect only 1 and 2 as class labels, got {:s}".format(
                str(unique_classes))
        )
        # now also create 0-1 vector for rejected trials
        trial_start_events = events[events[:,1] == 768]
        assert len(trial_start_events) == len(trial_events)
        artifact_trial_mask = np.zeros(len(trial_events), dtype=np.uint8)
        artifact_events = events[events[:,1] == 1023]
        for artifact_time in artifact_events[:,0]:
            i_trial = trial_start_events[:,0].tolist().index(artifact_time)
            artifact_trial_mask[i_trial] = 1

        return trial_events, artifact_trial_mask


class MultipleBCICompetition4Set2B(object):
    def __init__(self, subject_id, session_ids, data_folder):
        self.subject_id = subject_id
        self.session_ids = session_ids
        self.data_folder = data_folder

    def load(self):
        signal_folder = os.path.join(self.data_folder, 'signal')
        labels_folder = os.path.join(self.data_folder, 'labels')
        all_cnts = []
        for session_id in self.session_ids:
            labels_file_path = self.create_labels_file_path(self.subject_id,
                                                            session_id,
                                                            labels_folder)
            signal_file_path = self.create_signal_file_path(self.subject_id,
                                                            session_id,
                                                            signal_folder)
            all_cnts.append(BCICompetition4Set2B(
                signal_file_path,labels_filename=labels_file_path).load())
        artifact_masks = [cnt.attrs['artifact_trial_mask'] for cnt in all_cnts]
        merged_cnt = reduce(concatenate_cnt, all_cnts[1:], all_cnts[0])
        merged_cnt.attrs['artifact_trial_mask'] = np.concatenate(artifact_masks)
        assert merged_cnt.data.shape[0] == np.sum(
            [cnt.data.shape[0] for cnt in all_cnts])
        assert len(merged_cnt.attrs['artifact_trial_mask']) == len(
            merged_cnt.attrs['events'])
        return merged_cnt

    @staticmethod
    def create_signal_file_path(subject_id, session_id, signal_folder):
        train_or_eval = ("T" if session_id <= 3 else "E")
        filename = "B{:02d}{:02d}{:s}.gdf".format(subject_id, session_id,
                                                  train_or_eval)
        file_path = os.path.join(signal_folder, filename)
        return file_path

    @staticmethod
    def create_labels_file_path(subject_id, session_id, labels_folder):
        train_or_eval = ("T" if session_id <= 3 else "E")
        filename = "B{:02d}{:02d}{:s}.mat".format(subject_id, session_id,
                                                  train_or_eval)
        file_path = os.path.join(labels_folder, filename)
        return file_path


class BCICompetition4Set2A(object):
    def __init__(self, filename, load_sensor_names=None,
                 labels_filename=None):
        assert load_sensor_names is None
        self.__dict__.update(locals())
        del self.self

    def load(self):
        raw_edf = mne.io.read_raw_edf(self.filename)
        cnt = self.extract_data(raw_edf)
        cnt.attrs['fs'] = 250.0
        events, artifact_trial_mask = self.extract_events(raw_edf)
        cnt.attrs['events'] = events
        cnt.attrs['artifact_trial_mask'] = artifact_trial_mask
        return cnt

    @staticmethod
    def extract_data(raw_edf):
        cnt = xr.DataArray(raw_edf.get_data().T,
                           coords={'time': raw_edf.times * 1000.0,
                                   'channels': raw_edf.ch_names, },
                           # pd.to_timedelta(raw_edf.times,unit='s')},
                           dims=('time', 'channels'))
        assert np.array_equal(
            ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5',
             'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8',
             'EEG-9',
             'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz',
             'EEG-15', 'EEG-16', 'EOG-left', 'EOG-central', 'EOG-right',
             'STI 014'], raw_edf.ch_names)
        cnt = cnt.sel(channels=[
            'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5',
            'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8',
            'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14',
            'EEG-Pz', 'EEG-15', 'EEG-16',])


        # correct NaN Values ...
        # are set to lowest negative number
        data = cnt.values
        for i_chan in range(data.shape[1]):
            # first set to nan, than replace nans by nanmean.
            this_chan = data[:, i_chan]
            data[:, i_chan] = np.where(this_chan == np.min(this_chan),
                                       np.nan, this_chan)
            mask = np.isnan(data[:, i_chan])
            chan_mean = np.nanmean(data[:, i_chan])
            data[mask, i_chan] = chan_mean
        # somehow for correct units as before when loading from matlab,
        # need multiplication by 1e6...
        cnt.values = data * 1e6
        return cnt

    def extract_events(self, raw_edf):
        # all events
        events = np.array(list(zip(
            raw_edf.get_edf_events()[1],
            raw_edf.get_edf_events()[2])))

        # only trial onset events
        trial_mask = [ev_code in [769, 770, 771, 772, 783]
                      for ev_code in events[:,1]]
        trial_events = events[trial_mask]
        assert (len(trial_events) == 288), (
            "Got {:d} markers".format(len(trial_events)))
        # event markers 769,770 -> 1,2
        trial_events[:, 1] = trial_events[:, 1] - 768
        # possibly overwrite with markers from labels file
        if self.labels_filename is not None:
            classes = loadmat(self.labels_filename)['classlabel'].squeeze()
            trial_events[:, 1] = classes
        unique_classes = np.unique(trial_events[:, 1])
        assert np.array_equal([1, 2, 3 ,4], unique_classes), (
            "Expect 1,2,3,4 as class labels, got {:s}".format(
                str(unique_classes))
        )
        # now also create 0-1 vector for rejected trials
        trial_start_events = events[events[:, 1] == 768]
        assert len(trial_start_events) == len(trial_events)
        artifact_trial_mask = np.zeros(len(trial_events), dtype=np.uint8)
        artifact_events = events[events[:, 1] == 1023]
        for artifact_time in artifact_events[:, 0]:
            i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
            artifact_trial_mask[i_trial] = 1

        return trial_events, artifact_trial_mask

### old code


"""class BCICompetition4Set2A(object):
    def __init__(self, filename, load_sensor_names=None):
        self.__dict__.update(locals())
        del self.self

    def load(self):
        This function actually loads the data. Will be called by the 
        get dataset lazy loading function
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
                raise ValueError ("Only implemented loading all sensors")
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
            # -1 to account for matlab 1-based indexing
            start_samples = h5file['header']['EVENT']['POS'][0,:].astype(np.int64) - 1
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
            cnt.artefact_trial_mask = h5file['header']['ArtifactSelection'][:].squeeze()
        return cnt"""
    
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
    
class MultipleSetLoader(object):
    def __init__(self, set_loaders):
        self.set_loaders = set_loaders
        
    def load(self):
        cnt = self.set_loaders[0].load()
        for loader in self.set_loaders[1:]:
            next_cnt = loader.load()
            # always sample down to lowest common denominator
            if next_cnt.fs > cnt.fs:
                log.warn("Next set has larger sampling rate ({:d}) "
                    "than before ({:d}), resampling next set".format(
                        next_cnt.fs, cnt.fs))
                next_cnt = resample_cnt(next_cnt, cnt.fs)
            if next_cnt.fs < cnt.fs:
                log.warn("Next set has smaller sampling rate ({:d}) "
                    "than before ({:d}), resampling set so far".format(
                        next_cnt.fs, cnt.fs))
                cnt = resample_cnt(cnt, next_cnt.fs)
            cnt = append_cnt(cnt, next_cnt)
        return cnt
    
class MultipleBBCIDataset(object):
    def __init__(self, filenames, load_sensor_names=None):
        self.filenames = filenames
        self.load_sensor_names = load_sensor_names
    
    def load(self):
        bbci_sets = [BBCIDataset(fname, load_sensor_names=self.load_sensor_names)
                     for fname in self.filenames]
        return MultipleSetLoader(bbci_sets).load()

def get_strings_from_refs(h5file,refs):
    refs = refs.squeeze()
    strings = [''.join(chr(c) for c in h5file[obj_ref]) for obj_ref in refs]
    return strings





class BCICompetition4Set1(object):
    def __init__(self, subject_id, train_or_test, folder, load_sensor_names=None):
        assert load_sensor_names is None
        self.__dict__.update(locals())
        del self.self
        
    def load(self):
        if self.train_or_test == 'train':
            train_folder = os.path.join(self.folder,'signal/train/')

            filename = 'BCICIV_calib_ds1{:s}_1000Hz.mat'.format(self.subject_id)
            file_path = os.path.join(train_folder, filename)
            label_file_path = None
        else:
            test_folder = os.path.join(self.folder,'signal/test/')
            filename = 'BCICIV_eval_ds1{:s}_1000Hz.mat'.format(self.subject_id)
            file_path = os.path.join(test_folder, filename)
            label_folder = os.path.join(self.folder,'labels/')
            label_filename = 'BCICIV_eval_ds1{:s}_1000Hz_true_y.mat'.format(
                self.subject_id)
            label_file_path = os.path.join(label_folder, label_filename)

        return BCICompetition4Set1FromFile(
            file_path, label_file_path=label_file_path).load()


class BCICompetition4Set1FromFile(object):
    expected_chan_names = [u'AF3', u'AF4', u'F5', u'F3', u'F1', u'Fz', u'F2',
                       u'F4', u'F6', u'FC5', u'FC3', u'FC1', u'FCz', u'FC2',
                       u'FC4', u'FC6', u'CFC7', u'CFC5', u'CFC3', u'CFC1', u'CFC2',
                       u'CFC4', u'CFC6', u'CFC8', u'T7', u'C5', u'C3', u'C1', u'Cz',
                       u'C2', u'C4', u'C6', u'T8', u'CCP7', u'CCP5', u'CCP3', u'CCP1',
                       u'CCP2', u'CCP4', u'CCP6', u'CCP8', u'CP5', u'CP3', u'CP1',
                       u'CPz', u'CP2', u'CP4', u'CP6', u'P5', u'P3', u'P1', u'Pz',
                       u'P2', u'P4', u'P6', u'PO1', u'PO2', u'O1', u'O2']

    def __init__(self, filename, label_file_path=None, load_sensor_names=None):
        assert load_sensor_names is None
        self.__dict__.update(locals())
        del self.self
        
    def load(self):
        matfile = loadmat(self.filename)
        cnt, fs = self.load_signal(matfile)

        # load markers
        if self.label_file_path is None:
            mrk_in_ms, mrk_codes = self.load_markers_from_signal_file(
                matfile, fs)
        else:
            mrk_in_ms, mrk_codes = self.load_markers_from_label_file(
                self.label_file_path, fs)

        cnt.markers = zip(mrk_in_ms, mrk_codes)
        assert fs == 1000
        cnt.fs = fs
        return cnt

    def load_signal(self, matfile):
        fs = float(matfile['nfo']['fs'][0,0].squeeze())
        # load signal
        eeg_signal = matfile['cnt']
        samplenumbers = np.array(range(eeg_signal.shape[0]))
        timesteps_in_ms = samplenumbers * 1000.0 / fs
        chan_names = [chan_name[0] for chan_name in matfile['nfo']['clab'][0][0][0]]
        assert np.array_equal(self.expected_chan_names, chan_names)
        cnt = wyrm.types.Data(eeg_signal,
                    [timesteps_in_ms, chan_names],
                    ['time', 'channel'],
                    ['ms', '#'])
        return cnt, fs

    @staticmethod
    def load_markers_from_signal_file(matfile, fs):
        mrk_in_samples = matfile['mrk']['pos'][0, 0].squeeze()
        mrk_in_ms = mrk_in_samples * 1000.0 / fs
        mrk_code = matfile['mrk']['y'][0, 0].squeeze()
        mrk_code = ((mrk_code + 1) / 2) + 1
        return mrk_in_ms, mrk_code

    @staticmethod
    def load_markers_from_label_file(label_file_path, fs):
        labelfile = loadmat(label_file_path)
        y_signal = labelfile['true_y'].squeeze()
        modified_y_signal = y_signal.copy()
        modified_y_signal[np.isnan(modified_y_signal)] = 0

        one_hot_y_signal = np.zeros((len(modified_y_signal), 2), dtype=np.int32)
        one_hot_y_signal[modified_y_signal == -1, 0] = 1
        one_hot_y_signal[modified_y_signal == 1, 1] = 1

        starts, ends = compute_trial_start_end_samples(one_hot_y_signal,
                                                       check_trial_lengths_equal=False)
        mrk_codes = []
        mrk_ms = []
        for start, end in zip(starts, ends):
            assert np.isnan(y_signal[start - 1])
            assert not np.isnan(y_signal[start])
            assert np.isnan(y_signal[end + 1])
            assert not np.isnan(y_signal[end])
            mrk_codes.append(((y_signal[start] + 1) / 2) + 1)
            eval_start = start * 1000.0 / fs
            mrk_ms.append(eval_start - 1000) # first 1000 ms removed from eval..
            mrk_codes.append(((y_signal[end] + 1) / 2) + 11) # -> 11,12 for end
            mrk_ms.append(end * 1000.0 / fs)
        return mrk_ms, mrk_codes


class BCICompetition3Set4aFromFile(object):
    expected_chan_names = [
        'Fp1', 'AFp1', 'Fpz', 'AFp2', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'FAF5',
        'FAF1', 'FAF2', 'FAF6', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
        'F8', 'FFC7', 'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4', 'FFC6', 'FFC8',
        'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
        'FT10', 'CFC7', 'CFC5', 'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8',
        'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'CCP7', 'CCP5',
        'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'CCP8', 'TP9', 'TP7', 'CP5',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'PCP7', 'PCP5',
        'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6', 'PCP8', 'P9', 'P7', 'P5', 'P3',
        'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PPO7', 'PPO5', 'PPO1',
        'PPO2', 'PPO6', 'PPO8', 'PO7', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO8',
        'OPO1', 'OPO2', 'O1', 'Oz', 'O2', 'OI1', 'OI2', 'I1', 'I2']
    def __init__(self, filename, label_file_path, load_sensor_names=None):
        assert load_sensor_names is None
        self.__dict__.update(locals())
        del self.self

    def load(self):
        matfile = loadmat(self.filename)
        cnt, fs = self.load_signal(matfile)

        # load markers
        mrk_in_ms, mrk_codes = self.load_markers(
            matfile, self.label_file_path, fs)

        cnt.markers = zip(mrk_in_ms, mrk_codes)
        assert fs == 1000
        cnt.fs = fs
        return cnt

    def load_signal(self, matfile):
        fs = float(matfile['nfo']['fs'][0, 0].squeeze())
        # load signal
        eeg_signal = matfile['cnt']
        samplenumbers = np.array(range(eeg_signal.shape[0]))
        timesteps_in_ms = samplenumbers * 1000.0 / fs
        chan_names = [chan_name[0] for chan_name in matfile['nfo']['clab'][0][0][0]]
        assert np.array_equal(self.expected_chan_names, chan_names)
        cnt = wyrm.types.Data(eeg_signal,
                              [timesteps_in_ms, chan_names],
                              ['time', 'channel'],
                              ['ms', '#'])
        return cnt, fs

    @staticmethod
    def load_markers(matfile, label_file_path, fs):
        mrk_in_samples = matfile['mrk']['pos'][0, 0].squeeze()
        mrk_in_ms = mrk_in_samples * 1000.0 / fs
        mrk_code = matfile['mrk']['y'][0, 0].squeeze()
        labels = loadmat(label_file_path)
        true_y = labels['true_y'].squeeze()
        assert len(true_y) == len(mrk_code)
        mrk_code = true_y
        return mrk_in_ms, mrk_code


class BCICompetition3Set4a(object):
    def __init__(self, subject_id, folder, load_sensor_names=None):
        assert load_sensor_names is None
        self.__dict__.update(locals())
        del self.self

    def load(self):
        signal_folder = os.path.join(self.folder, 'signal/')
        filename = 'data_set_IVa_a{:s}.mat'.format(self.subject_id)
        file_path = os.path.join(signal_folder, filename)
        label_folder = os.path.join(self.folder, 'labels/')
        label_filename = 'true_labels_a{:s}.mat'.format(self.subject_id)
        label_file_path = os.path.join(label_folder, label_filename)
        return BCICompetition3Set4aFromFile(
            file_path, label_file_path=label_file_path).load()


class BCICompetition3Set5(object):
    def __init__(self, subject_id, folder, train_or_test, load_sensor_names=None):
        assert load_sensor_names is None
        self.__dict__.update(locals())
        del self.self

    def load(self):
        signal_folder = os.path.join(self.folder, 'signal/')
        if self.train_or_test == 'train':
            label_file_path = None
            all_sets = []
            for i_session in range(1,4):
                filename = 'train_subject{:d}_raw0{:d}.mat'.format(
                    self.subject_id, i_session)
                file_path = os.path.join(signal_folder, filename)
                all_sets.append(BCICompetition3Set5FromFile(
                    file_path, label_file_path=label_file_path,
                    load_sensor_names=self.load_sensor_names))
            return MultipleSetLoader(all_sets).load()
        else:
            assert self.train_or_test == 'test'
            label_folder = os.path.join(self.folder, 'labels/')
            filename = 'test_subject{:d}_raw04.mat'.format(self.subject_id)
            file_path = os.path.join(signal_folder, filename)
            label_file_name = 'labels8_subject{:d}_raw.asc'.format(
                self.subject_id)
            label_file_path = os.path.join(label_folder, label_file_name)
            return BCICompetition3Set5FromFile(
                filename=file_path, label_file_path=label_file_path,
                load_sensor_names=self.load_sensor_names).load()


class BCICompetition3Set5FromFile(object):
    expected_chan_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3',
                           'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2',
                           'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6',
                           'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

    def __init__(self, filename, label_file_path, load_sensor_names=None):
        assert load_sensor_names is None
        self.__dict__.update(locals())
        del self.self

    def load(self):
        matfile = loadmat(self.filename)
        cnt, fs = self.load_signal(matfile)

        # load markers
        mrk_in_ms, mrk_codes = self.load_markers(
            matfile, self.label_file_path, fs, len(cnt.data))

        cnt.markers = zip(mrk_in_ms, mrk_codes)
        assert fs == 512
        cnt.fs = fs
        return cnt

    def load_signal(self, matfile):
        signal = matfile['X']
        fs = float(matfile['nfo']['fs'][0, 0][0, 0])
        samplenumbers = np.array(range(signal.shape[0]))
        timesteps_in_ms = samplenumbers * 1000.0 / fs
        chan_names = [str(s[0])
                      for s in matfile['nfo']['clab'][0,0].squeeze()]
        assert np.array_equal(self.expected_chan_names, chan_names)
        cnt = wyrm.types.Data(signal,
                              [timesteps_in_ms, chan_names],
                              ['time', 'channel'],
                              ['ms', '#'])
        return cnt, fs

    @staticmethod
    def load_markers(matfile, label_file_path, fs, signal_len):
        """Signal len necessary for label file path...
        Otherwise end will not be correct"""
        if label_file_path is not None:
            y = np.loadtxt(label_file_path).astype(np.int32).squeeze()
            # have to repeat since given only eery 500ms= every 256 samples
            y = np.repeat(y, 256)
            # some y may be missing at end of signal, but we don't
            # need to pad since anyways signal len is added below
            # as final boundary...
        else:
            y = matfile['Y'].squeeze()
        boundaries = np.flatnonzero(np.diff(y)) + 1
        boundaries = np.concatenate((boundaries,  [signal_len]))
        start = 0
        mrk_code = []
        mrk_ms = []
        for next_start in boundaries:
            # add start marker
            if start > 0:
                assert y[start - 1] != y[start]
            if next_start < len(y):
                assert y[next_start] != y[next_start-1]
            assert np.all(y[start:next_start] == y[start])
            mrk_code.append(y[start])
            mrk_ms.append(start * 1000.0 / fs)
            # add end marker
            mrk_code.append(y[start] + 10)
            end_ms = (next_start - 1) * 1000.0 / fs
            mrk_ms.append(end_ms)
            start = next_start
        return mrk_ms, mrk_code
