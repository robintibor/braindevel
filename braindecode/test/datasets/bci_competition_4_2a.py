from braindecode.datasets.set_loaders import BCICompetition4Set2A
from braindecode.datasets.signal_processor import SignalProcessor
from glob import glob
import h5py
import numpy as np
import logging
log = logging.getLogger(__name__)

# Test A: check if loaded sets signal is correct for train test 
# and labels correct for train, labels pos correct for test
# Test B: check if loaded through bci competition set 
# with wyrm set train trials are same as first 288 trials,
# test trials same as later 288 trials

def check_file(train_file, test_file, combined_file):
    train_signal = train_file['signal'][:]
    test_signal = test_file['signal'][:]
    combined_signal = combined_file['signal'][:]
    # Replace nans with inf to allow "allclose" comparison
    # (np.allclose(np.nan, np.nan) => False!)
    assert not np.any(np.isinf(train_signal))
    assert not np.any(np.isinf(test_signal))
    assert not np.any(np.isinf(combined_signal))
    train_signal[np.isnan(train_signal)] = np.inf
    test_signal[np.isnan(test_signal)] = np.inf
    combined_signal[np.isnan(combined_signal)] = np.inf
    assert np.allclose(train_signal, combined_signal[:, :train_signal.shape[1]])
    assert np.allclose(test_signal, combined_signal[:, train_signal.shape[1]:])
    train_labels = train_file['header']['Classlabel'][0,:]
    train_event_type = train_file['header']['EVENT']['TYP'][0,:]
    train_trial_mask = np.array([ev in [769,770,771,772] for ev in train_event_type])
    assert np.array_equal(train_event_type[train_trial_mask] - 768, train_labels)
    
    test_labels = test_file['header']['Classlabel'][0,:]
    
    combined_labels = combined_file['header']['Classlabel'][0,:]
    combined_event_type = combined_file['header']['EVENT']['TYP'][0,:]
    combined_trial_mask = np.array([ev in [769,770,771,772] for ev in combined_event_type])
    assert np.array_equal(combined_event_type[combined_trial_mask] - 768, combined_labels)
    
    assert len(train_labels) == 288
    assert len(test_labels) == 288
    assert len(combined_labels) == 288 * 2
    assert np.array_equal(combined_labels[:288], train_labels)
    train_event_pos = train_file['header']['EVENT']['POS'][0,:]
    test_event_pos = test_file['header']['EVENT']['POS'][0,:]
    combined_event_pos = combined_file['header']['EVENT']['POS'][0,:]
    
    assert np.array_equal(train_event_pos,
        combined_event_pos[:len(train_event_pos)])
    assert np.array_equal(test_event_pos,
        combined_event_pos[len(train_event_pos):] - train_signal.shape[1])
    log.info("File ok")

def check_as_sets(train_file_name, test_file_name, combined_file_name):
    train_set = BCICompetition4Set2A(train_file_name)
    train_wyrm_set = SignalProcessor(train_set)
    train_wyrm_set.load()
    
    test_set = BCICompetition4Set2A(test_file_name)
    test_wyrm_set = SignalProcessor(test_set,
        marker_def={'Unknown':[-2147483648]})
    test_wyrm_set.load()
    
    combined_set = BCICompetition4Set2A(combined_file_name)
    combined_wyrm_set = SignalProcessor(combined_set)
    combined_wyrm_set.load()
    # nans were made to be means, so ignore that some values are not equal
    train_epo = train_wyrm_set.epo.data
    test_epo = test_wyrm_set.epo.data
    combined_epo = combined_wyrm_set.epo.data
    train_part = combined_epo[:288]
    assert (np.sum(train_epo - train_part!= 0) /
        float(np.prod(train_epo.shape))) < 1e-2
    test_part = combined_epo[288:]
    assert (np.sum(test_epo - test_part) /
        float(np.prod(test_epo.shape))) < 1e-2
    log.info("Set ok")

def test_all():
    """ Extra function with test_ so pytest will also run it """
    train_files = sorted(glob('data/bci-competition-iv/2a/*T.mat'))
    test_files = sorted(glob('data/bci-competition-iv/2a/*E.mat'))
    combined_files = sorted(glob('data/bci-competition-iv/2a-combined/*TE.mat'))
    
    train_test_combined = zip(train_files, test_files, combined_files)
    for train_file_name, test_file_name, combined_file_name in train_test_combined:
        log.info("Checking {:s}".format(combined_file_name))
        with h5py.File(
                train_file_name, 'r') as train_file, h5py.File(
                test_file_name, 'r') as test_file, h5py.File(
                combined_file_name, 'r') as combined_file:
            check_file(train_file, test_file, combined_file)
        check_as_sets(train_file_name, test_file_name, combined_file_name)

if __name__ == '__main__':
    test_all()