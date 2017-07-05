from wyrm.types import Data
from braindecode.datasets.signal_processor import SignalProcessor
from braindecode.mywyrm.clean import Cleaner
import numpy as np

class FakeLoader():
    def __init__(self, cnt):
        self.cnt = cnt
    def load(self):
        return self.cnt
        

def compute_cleaner(data, eog_data,marker_positions, ival, max_min=2,
                   whisker_percent=5, whisker_length=3):
    """For Cleaner tests..."""
    assert eog_data.shape[0] == data.shape[0]
    
    axes = [range(data.shape[0]), range(data.shape[1])]
    markers = zip(marker_positions, [0] * len(marker_positions))
    marker_def={'0':[0]}
    cnt = Data(data,axes=axes,names=['time', 'channels'], units=['ms', '#'])
    cnt.fs = 1000
    cnt.markers = markers
    
    
    eog_axes = [range(eog_data.shape[0]), range(eog_data.shape[1])]
    eog_cnt = Data(eog_data,axes=eog_axes,names=['time', 'channels'], units=['ms', '#'])
    eog_cnt.fs = 1000
    eog_cnt.markers = markers
    eog_proc = SignalProcessor(FakeLoader(eog_cnt),segment_ival=ival,marker_def=marker_def)
                
    cleaner = Cleaner(cnt,eog_proc,rejection_blink_ival=ival,
       max_min=max_min,rejection_var_ival=ival, whisker_percent=whisker_percent, 
                      whisker_length=whisker_length,
       low_cut_hz=None, high_cut_hz=None, filt_order=None,marker_def=marker_def)
    cleaner.clean()
    return cleaner

def test_nothing_cleaned():
    data_shape = [20,1]
    marker_positions = [0.0,4.0,8.0,12.0,16.0]
    data = np.zeros(data_shape)
    eog_data = np.zeros(data_shape)
    cleaner = compute_cleaner(data,eog_data,
        marker_positions,ival=[0,2], max_min=2, whisker_percent=20)
    assert(len(cleaner.rejected_trials) == 0)
    assert(len(cleaner.rejected_max_min) == 0)
    assert(len(cleaner.rejected_var) == 0)
    assert(len(cleaner.rejected_chan_names) == 0)
    assert(np.array_equal([0,1,2,3,4], cleaner.clean_trials))
    
def test_max_min_cleaned():
    data_shape = [20,1]
    marker_positions = [0.0,4.0,8.0,12.0,16.0]
    data = np.zeros(data_shape)
    eog_data = np.zeros(data_shape)
    eog_data[4:6,0] = [4,1]
    cleaner = compute_cleaner(data,eog_data,
        marker_positions,ival=[0,2], max_min=2, whisker_percent=20)
    assert(np.array_equal([1], cleaner.rejected_trials))
    assert(np.array_equal([1], cleaner.rejected_max_min))
    assert(np.array_equal([], cleaner.rejected_var))
    assert(np.array_equal([], cleaner.rejected_chan_names))
    assert(np.array_equal([0,2,3,4], cleaner.clean_trials))

def test_var_cleaned():
    data_shape = [20,1]
    marker_positions = [0.0,4.0,8.0,12.0,16.0]
    data = np.zeros(data_shape)
    eog_data = np.zeros(data_shape)
    data[4:6,0] = [4,1]
    cleaner = compute_cleaner(data,eog_data,
        marker_positions,ival=[0,2], max_min=2, whisker_percent=20)
    assert(np.array_equal([1], cleaner.rejected_trials))
    assert(np.array_equal([], cleaner.rejected_max_min))
    assert(np.array_equal([1], cleaner.rejected_var))
    assert(np.array_equal([], cleaner.rejected_chan_names))
    assert(np.array_equal([0,2,3,4], cleaner.clean_trials))
    
def test_max_min_cleaned_var_ignored():
    data_shape = [20,1]
    marker_positions = [0.0,4.0,8.0,12.0,16.0]
    data = np.zeros(data_shape)
    eog_data = np.zeros(data_shape)
    data[4:6,0] = [4,1]
    eog_data[4:6,0] = [4,1]
    cleaner = compute_cleaner(data,eog_data,
        marker_positions,ival=[0,2], max_min=2, whisker_percent=20)
    assert(np.array_equal([1], cleaner.rejected_trials))
    assert(np.array_equal([1], cleaner.rejected_max_min))
    assert(np.array_equal([], cleaner.rejected_var))
    assert(np.array_equal([], cleaner.rejected_chan_names))
    assert(np.array_equal([0,2,3,4], cleaner.clean_trials))
    
def test_chan_cleaned():
    data_shape = [20,5]
    marker_positions = [0.0,4.0,8.0,12.0,16.0]
    data = np.zeros(data_shape)
    eog_data = np.zeros(data_shape)
    data[4:6,1] = [4,1]
    cleaner = compute_cleaner(data,eog_data,
        marker_positions,ival=[0,2], max_min=2, whisker_percent=20)
    assert(np.array_equal([], cleaner.rejected_trials))
    assert(np.array_equal([], cleaner.rejected_max_min))
    assert(np.array_equal([], cleaner.rejected_var))
    assert(np.array_equal([1], cleaner.rejected_chan_names))
    assert(np.array_equal([0,1,2,3,4], cleaner.clean_trials))