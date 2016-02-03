import numpy as np
from wyrm.processing import select_channels
from braindecode.mywyrm.clean import SetCleaner
from braindecode.datasets.loaders import BBCIDataset
from braindecode.mywyrm.processing import select_marker_classes,\
    select_marker_epochs
from braindecode.datasets.raw import CleanSignalMatrix
from braindecode.datasets.cnt_signal_matrix import CntSignalMatrix
import logging
log = logging.getLogger(__name__)

class CombinedSet(object):
    reloadable=False
    def __init__(self, sets):
        self.sets = sets

    def ensure_is_loaded(self):
        for dataset in self.sets:
            dataset.ensure_is_loaded()
    def load(self):
        for dataset in self.sets:
            dataset.load()
        # hack to have correct y dimensions
        self.y = self.sets[-1].y[0:1]

def restrict_cnt(cnt, classes, clean_trials, rejected_chan_names, copy_data=False):
    cleaned_cnt = select_marker_classes(cnt, classes,
                                       copy_data)
    cleaned_cnt = select_marker_epochs(cleaned_cnt, clean_trials,
                                      copy_data)
    cleaned_cnt = select_channels(cleaned_cnt, rejected_chan_names, invert=True)
    return cleaned_cnt

class CombinedCleanedSet(object):
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set

    def load(self):
        # Loading both sets, cleaning cnts and finished
        self.train_set.signal_processor.load_signal_and_markers()
        self.test_set.signal_processor.load_signal_and_markers()
        train_filename = self.train_set.signal_processor.set_loader.filename
        test_filename = self.test_set.signal_processor.set_loader.filename
        train_cnt = self.train_set.signal_processor.cnt
        test_cnt = self.test_set.signal_processor.cnt
        train_cleaner = SetCleaner(eog_set=BBCIDataset(train_filename,
                                     load_sensor_names=['EOGh', 'EOGv']))
        test_cleaner = SetCleaner(eog_set=BBCIDataset(test_filename,
                                             load_sensor_names=['EOGh', 'EOGv']))

        train_clean_result = train_cleaner.clean(train_cnt)
        test_clean_result = test_cleaner.clean(test_cnt, 
            preremoved_chans=train_clean_result.rejected_chan_names)
        assert np.array_equal(test_clean_result.rejected_chan_names,
                      train_clean_result.rejected_chan_names)
        clean_train_cnt = restrict_cnt(train_cnt, train_cleaner.marker_def.values(),
                                      train_clean_result.clean_trials, train_clean_result.rejected_chan_names,
                                      copy_data=False)
        clean_test_cnt = restrict_cnt(test_cnt, train_cleaner.marker_def.values(),
                                      test_clean_result.clean_trials, test_clean_result.rejected_chan_names,
                                      copy_data=False)
        self.train_set.signal_processor.cnt = clean_train_cnt
        self.test_set.signal_processor.cnt = clean_test_cnt
        
        # in case of cnt signal matrix:
        if isinstance(self.train_set, CntSignalMatrix):
            self.train_set.load_from_cnt()
            self.test_set.load_from_cnt()
        # in case of raw set:
        elif isinstance(self.train_set, CleanSignalMatrix):
            for one_set in [self.train_set, self.test_set]:
                one_set.load_from_cnt()
                one_set.create_dense_design_matrix()
                one_set.remove_signal_epo()
                one_set.create_dense_design_matrix()
                if one_set.unsupervised_preprocessor is not None:
                    one_set.apply_unsupervised_preprocessor()
                one_set.y = np.argmax(self.y, axis=1).astype(np.int32)
        else:
            raise ValueError("Unknown type of train set {:s}".format(
                self.train_set.__class__.__name__))
        
        self.sets = [self.train_set, self.test_set]
        log.info("Loaded clean train data with shape {:s}.".format(
            self.train_set.signal_processor.epo.data.shape))    
        log.info("Loaded clean test data with shape {:s}.".format(
            self.test_set.signal_processor.epo.data.shape))         

        self.y = self.sets[-1].y[0:1]