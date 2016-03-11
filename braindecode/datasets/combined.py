import numpy as np
from braindecode.datasets.raw import CleanSignalMatrix
from braindecode.datasets.cnt_signal_matrix import CntSignalMatrix
from braindecode.mywyrm.clean import clean_train_test_cnt
import logging
from braindecode.datasets.loaders import BBCIDataset
from braindecode.datasets.signal_processor import SignalProcessor
from braindecode.util import FuncAndArgs
from braindecode.mywyrm.processing import select_marker_classes,\
    select_marker_epoch_range, select_relevant_ival
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

class CombinedCntSets(object):
    reloadable=False
    def __init__(self, set_args, load_sensor_names,
        sensor_names, segment_ival, 
        cnt_preprocessors, marker_def):
        self.__dict__.update(locals())
        del self.self
    
    def ensure_is_loaded(self):
        if not hasattr(self, 'sets'):
            self.load()
    
    def load(self):
        self.construct_sets()
        for dataset in self.sets:
            dataset.load()
        # hack to have correct y dimensions
        self.y = self.sets[-1].y[0:1]
    
    def construct_sets(self):
        self.sets = []
        for set_arg in self.set_args:
            filename, constructor, start_stop = set_arg
            if constructor == 'bbci':
                constructor = BBCIDataset
            loader= constructor(filename, 
                load_sensor_names=self.load_sensor_names)
            additional_cnt_preprocs = []
            if start_stop is not None:
                start, stop = start_stop
                assert np.all([len(labels) == 1 for labels in 
                    self.marker_def.values()]), (
                    "Expect only one label per class, otherwise rewrite...")
        
                classes = sorted([labels[0] for labels in self.marker_def.values()])
                select_class = [select_marker_classes,
                    dict(classes=classes, copy_data=False)]
                additional_cnt_preprocs.append(select_class)
                
                select_epochs = [select_marker_epoch_range,
                    dict(start=start, stop=stop)]
                additional_cnt_preprocs.append(select_epochs)
                
                # HERE 
                select_ival = [select_relevant_ival,
                     dict(segment_ival=self.segment_ival)]
                additional_cnt_preprocs.append(select_ival)
                
            this_cnt_preprocs = self.cnt_preprocessors + additional_cnt_preprocs
            signal_proc= SignalProcessor(
                set_loader=loader,
                segment_ival=self.segment_ival,
                cnt_preprocessors=this_cnt_preprocs,
                marker_def=self.marker_def)
            this_set = CntSignalMatrix(signal_processor=signal_proc,
                sensor_names=self.sensor_names)
            self.sets.append(this_set)
            

        
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
        