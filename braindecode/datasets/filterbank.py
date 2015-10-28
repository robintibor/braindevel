import numpy as np
import logging
from braindecode.mywyrm.processing import (bandpass_cnt,  segment_dat_fast)
from wyrm.processing import select_epochs
from wyrm.types import Data

from braindecode.datasets.raw import CleanSignalMatrix
log = logging.getLogger()

def generate_filterbank(min_freq, max_freq, last_low_freq,
        low_width, high_width):
    assert isinstance(min_freq, int) or min_freq.is_integer()
    assert isinstance(max_freq, int) or max_freq.is_integer()
    assert isinstance(last_low_freq, int) or last_low_freq.is_integer()
    assert isinstance(low_width, int) or low_width.is_integer()
    assert isinstance(high_width, int) or high_width.is_integer()
    
    assert high_width % 2  == 0
    assert low_width % 2  == 0
    assert (last_low_freq - min_freq) % low_width  == 0, ("last low freq "
        "needs to be exactly the center of a low_width filter band")
    assert max_freq >= last_low_freq
    assert (max_freq == last_low_freq or  
            (max_freq - (last_low_freq + low_width/2 + high_width/2)) % 
        high_width == 0), ("max freq needs to be exactly the center "
            "of a filter band")
    low_centers = range(min_freq,last_low_freq+1, low_width)
    high_start = last_low_freq + low_width/2 + high_width/2
    high_centers = range(high_start, max_freq+1, high_width)
    
    low_band = np.array([np.array(low_centers) - low_width/2, 
                         np.array(low_centers) + low_width/2]).T
    low_band = np.maximum(0.5, low_band)
    high_band = np.array([np.array(high_centers) - high_width/2, 
                         np.array(high_centers) + high_width/2]).T
    filterbank = np.concatenate((low_band, high_band))
    return filterbank

class FilterbankCleanSignalMatrix(CleanSignalMatrix):
    reloadable=True
    def __init__(self, min_freq, max_freq,
            last_low_freq, low_width, high_width,
            **kwargs):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.last_low_freq = last_low_freq
        self.low_width = low_width
        self.high_width = high_width
        super(FilterbankCleanSignalMatrix, self).__init__(**kwargs)
        
    def load_signal(self):
        self.load_full_set()
        self.determine_clean_trials_and_chans()
        self.select_sensors()
        log.info("Preprocessing continuous signal...")
        self.signal_processor.preprocess_continuous_signal()
        self.create_filterbank()
 
    def create_filterbank(self):
        log.info("Creating filterbank...")
        # Create filterbands and array for holding
        # filterband trials 
        self.filterbands = generate_filterbank(self.min_freq, self.max_freq,
            self.last_low_freq, self.low_width, self.high_width)
        segment_length =  (self.signal_processor.segment_ival[1] -
            self.signal_processor.segment_ival[0])
        num_samples = segment_length  * self.signal_processor.cnt.fs / 1000.0
        assert num_samples.is_integer()
        num_samples = int(num_samples)
        full_epo_data = np.empty((len(self.clean_trials), num_samples, 
            len(self.sensor_names), len(self.filterbands)), dtype=np.float32)
        # Fill filterbank
        self.fill_filterbank_data(full_epo_data)
        # Transform to wyrm epoched dataset
        clean_markers = [m for i,m in enumerate(self.signal_processor.cnt.markers) \
            if i not in self.rejected_trials]
        del self.signal_processor.cnt
        new_epo = Data(data=full_epo_data, 
            axes=self.filterband_axes, names=self.filterband_names,
            units=self.filterband_names)
        new_epo.markers = clean_markers
        self.signal_processor.epo = new_epo

    def fill_filterbank_data(self, full_epo_data):
        for filterband_i in xrange(len(self.filterbands)): 
            low_freq, high_freq= self.filterbands[filterband_i]
            log.info("Filterband {:d} of {:d}, from {:5.2f} to {:5.2f}".format(
                filterband_i + 1, len(self.filterbands), low_freq, high_freq))
            bandpassed_cnt = bandpass_cnt(self.signal_processor.cnt, 
                low_freq, high_freq, filt_order=3)
            epo = segment_dat_fast(bandpassed_cnt, 
                   marker_def={'1 - Right Hand': [1], '2 - Left Hand': [2], 
                       '3 - Rest': [3], '4 - Feet': [4]}, 
                   ival=self.signal_processor.segment_ival)
            epo.data = np.float32(epo.data)
            epo = select_epochs(epo, self.rejected_trials, invert=True)
            full_epo_data[:,:,:,filterband_i] = epo.data
            del epo.data
            del bandpassed_cnt
        self.filterband_axes = epo.axes + [self.filterbands.tolist()]
        self.filterband_names = epo.names + ['filterband']
        self.filterband_units = epo.units + ['Hz']

    def free_memory(self):
        del self.X
