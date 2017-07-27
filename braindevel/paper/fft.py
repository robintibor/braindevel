import logging
import numpy as np
from glob import glob
from braindevel.mywyrm.clean import MaxAbsCleaner, restrict_cnt
from braindevel.mywyrm.processing import (set_channel_to_zero,
    common_average_reference_cnt, exponential_standardize_cnt,
    exponential_demean_cnt, resample_cnt)
from braindevel.datasets.loaders import BBCIDataset
from braindevel.datasets.sensor_positions import get_EEG_sensors_sorted
from braindevel.paper import unclean_sets
log = logging.getLogger(__name__)

def load_file(file_name, standardize, demean):
    bbci_set = BBCIDataset(file_name,
                          load_sensor_names=get_EEG_sensors_sorted())
    log.info("Loading...")
    cnt = bbci_set.load()
    log.info("Set cz to zero and remove high absolute value trials")
    marker_def = dict([(str(i_class), [i_class])  for i_class in xrange(1,5)])
    clean_result = MaxAbsCleaner(threshold=800,
                                                  marker_def=marker_def,
                                                  segment_ival=[0,4000]).clean(cnt)
    
    cnt = restrict_cnt(cnt, marker_def.values(), clean_result.clean_trials, 
        clean_result.rejected_chan_names, copy_data=False)
    cnt = set_channel_to_zero(cnt, 'Cz')
    
    
    log.info("Resampling...")
    cnt = resample_cnt(cnt, newfs=250.0)
    
    log.info("Car filtering...")
    cnt = common_average_reference_cnt(cnt)
    if standardize:
        log.info("Standardizing...")
        cnt = exponential_standardize_cnt(cnt)
    if demean: 
        log.info("Demeaning...")
        cnt = exponential_demean_cnt(cnt)
    
    return cnt

def load_amps(file_pattern='data/fft-analysis-standardized-cnt/*base_others.npy'):
    all_file_names = sorted(glob(file_pattern))
    clean_mask = []

    all_relative_class_amps = []
    for file_name in all_file_names:
        is_clean_set = not np.any([name in file_name for name in unclean_sets])
        clean_mask.append(is_clean_set)
        all_relative_class_amps.append(np.load(file_name))

    all_relative_class_amps = np.array(all_relative_class_amps)
    clean_mask = np.array(clean_mask)
    assert len(all_relative_class_amps) == 20
    # 1750 samples, 500 sampling rate.. apparently this was done without resampling originally
    freqs_relative = np.fft.rfftfreq(1750,1/500.0)
    return all_relative_class_amps, clean_mask, freqs_relative
