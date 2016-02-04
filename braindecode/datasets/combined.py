import numpy as np
from wyrm.processing import select_channels
from braindecode.mywyrm.processing import select_marker_classes,\
    select_marker_epochs
from braindecode.datasets.raw import CleanSignalMatrix
from braindecode.datasets.cnt_signal_matrix import CntSignalMatrix
from braindecode.mywyrm.clean import log_clean_result, clean_train_test_cnt
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

        
class CombinedCleanedSet(object):
    reloadable=False
    def __init__(self, train_set, test_set, train_cleaner, test_cleaner):
        self.train_set = train_set
        self.test_set = test_set
        self.train_cleaner = train_cleaner
        self.test_cleaner = test_cleaner

    def ensure_is_loaded(self):
        if not hasattr(self, 'sets'):
            self.load()

    def load(self):
        # Loading both sets, cleaning cnts and finished
        log.info("Load Training Set...")
        self.train_set.signal_processor.load_signal_and_markers()
        log.info("Load Test Set...")
        self.test_set.signal_processor.load_signal_and_markers()
        train_cnt = self.train_set.signal_processor.cnt
        test_cnt = self.test_set.signal_processor.cnt
        
        clean_train_cnt, clean_test_cnt = clean_train_test_cnt(train_cnt,
            test_cnt,self.train_cleaner, self.test_cleaner)
        self.train_set.signal_processor.cnt = clean_train_cnt
        self.test_set.signal_processor.cnt = clean_test_cnt
        
        assert np.array_equal(self.train_set.signal_processor.cnt.axes[1], 
            self.train_set.signal_processor.cnt.axes[1]), ("Sensor names should "
                "be the same for train and test...")
        log.info("Create sets from cleaned cnt...")
        # in case of cnt signal matrix:
        if isinstance(self.train_set, CntSignalMatrix):
            self.train_set.load_from_cnt()
            self.test_set.load_from_cnt()
        # in case of raw set:
        elif isinstance(self.train_set, CleanSignalMatrix):
            for one_set in [self.train_set, self.test_set]:
                # this is very fragile... as changes to
                # the original class logic will affect this :(
                one_set.load_from_cnt()
                one_set.create_dense_design_matrix()
                one_set.remove_signal_epo()
                if one_set.unsupervised_preprocessor is not None:
                    one_set.apply_unsupervised_preprocessor()
                one_set.y = np.argmax(one_set.y, axis=1).astype(np.int32)

        else:
            raise ValueError("Unknown type of train set {:s}".format(
                self.train_set.__class__.__name__))
        log.info("Loaded clean train data with shape {:s}.".format(
            self.train_set.get_topological_view().shape))    
        log.info("Loaded clean test data with shape {:s}.".format(
            self.test_set.get_topological_view().shape))            

        self.sets = [self.train_set, self.test_set]
        
        self.y = self.sets[-1].y[0:1]
        