import logging
from braindecode.mywyrm.clean import MaxAbsCleaner, restrict_cnt
from braindecode.mywyrm.processing import (set_channel_to_zero,
    common_average_reference_cnt, exponential_standardize_cnt)
from braindecode.datasets.loaders import BBCIDataset
from braindecode.datasets.sensor_positions import get_EEG_sensors_sorted
log = logging.getLogger(__name__)

def load_file(file_name, standardize):
    bbci_set = BBCIDataset(file_name,
                          load_sensor_names=get_EEG_sensors_sorted())
    log.info("Loading...")
    cnt = bbci_set.load()
    log.info("Set cz to zero and remove high absolute value trials")
    marker_def = dict([(str(i_class), [i_class])  for i_class in xrange(1,5)])
    clean_result = MaxAbsCleaner(threshold=800,
                                                  marker_def=marker_def,
                                                  segment_ival=[0,4000]).clean(cnt)
    cnt = set_channel_to_zero(cnt, 'Cz')
    cnt = restrict_cnt(cnt, marker_def.values(), clean_result.clean_trials, 
        clean_result.rejected_chan_names, copy_data=False)
    
    log.info("Car filtering...")
    cnt = common_average_reference_cnt(cnt)
    if standardize:
        log.info("Standardizing...")
        cnt = exponential_standardize_cnt(cnt)
    return cnt