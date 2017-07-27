import numpy as np
from copy import deepcopy, copy
from glob import glob
import os.path
from braindevel.datasets.raw import CleanSignalMatrix
from braindevel.datasets.cnt_signal_matrix import CntSignalMatrix,\
    SetWithMarkers
from braindevel.mywyrm.clean import clean_train_test_cnt
import logging
from braindevel.datasets.loaders import BBCIDataset, MultipleBBCIDataset
from braindevel.datasets.signal_processor import SignalProcessor
from braindevel.mywyrm.processing import select_marker_classes,\
    select_marker_epoch_range, select_ival_with_markers
from braindevel.datasets.trial_segmenter import MarkerSegmenter, AddTrialBreaks,\
    PipelineSegmenter, RestrictTrialRange
log = logging.getLogger(__name__)

class CombinedSet(object):
    reloadable=False
    def __init__(self, sets):
        self.sets = sets

    def ensure_is_loaded(self):
        for dataset in self.sets:
            dataset.ensure_is_loaded()
    def load(self):
        for i_set, dataset in enumerate(self.sets):
            log.info("Loading set {:d} of {:d}...".format(
                i_set + 1, len(self.sets)))
            dataset.load()
        # hack to have correct y dimensions
        self.y = self.sets[-1].y[0:1]

class CombinedCntSets(object):
    reloadable=False
    def __init__(self, set_args, load_sensor_names,
        sensor_names, 
        cnt_preprocessors, marker_def):
        """ Per Set, set_args should be (filename, constructor, start_stop, 
                segment_ival, end_marker_def)"""
        self.__dict__.update(locals())
        del self.self
        if self.load_sensor_names == 'all':
            self.load_sensor_names = None
    
    def ensure_is_loaded(self):
        if not hasattr(self, 'sets'):
            self.load()
    
    def load(self):
        self.construct_sets()
        for i_set, dataset in enumerate(self.sets):
            log.info("Loading set {:d} of {:d}".format(i_set + 1,
                len(self.sets)))
            dataset.load()
        # hack to have correct y dimensions
        # TODO: remove
        self.y = self.sets[-1].y[0:1]
    
    def construct_sets(self):
        self.sets = []
        for set_arg in self.set_args:
            #Thinkabout: maybe do additional preprocs before?
            (filename, constructor, start_stop, 
                segment_ival, end_marker_def) = set_arg
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
        
                classes = sorted([labels[0]
                    for labels in self.marker_def.values()])
                if end_marker_def is not None:
                    end_classes = sorted(
                        [labels[0]
                            for labels in end_marker_def.values()])
                    classes.extend(end_classes)
                select_class = [select_marker_classes,
                    dict(classes=classes, copy_data=False)]
                additional_cnt_preprocs.append(select_class)
                if end_marker_def is not None:
                    # since there are start and end markers,
                    # need to multiply indices by 2
                    if start is not None:
                        start = start * 2
                    if stop is not None:
                        stop = stop * 2
                select_epochs = [select_marker_epoch_range,
                    dict(start=start, stop=stop)]
                    
                additional_cnt_preprocs.append(select_epochs)
                
                select_ival = [select_ival_with_markers,
                     dict(segment_ival=segment_ival)]
                additional_cnt_preprocs.append(select_ival)
                
            
            this_cnt_preprocs = (list(self.cnt_preprocessors) +
                additional_cnt_preprocs)
            signal_proc= SignalProcessor(
                set_loader=loader,
                segment_ival=segment_ival,
                cnt_preprocessors=this_cnt_preprocs,
                marker_def=self.marker_def)
            this_set = CntSignalMatrix(signal_processor=signal_proc,
                sensor_names=self.sensor_names,
                end_marker_def = end_marker_def)
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

def construct_combined_set(filenames, sensor_names, cnt_preprocessors,
                            marker_def, end_marker_def, trial_classes,
                          trial_start_offset_ms, trial_stop_offset_ms,
                           min_break_length_ms, max_break_length_ms,
                          break_start_offset_ms, break_stop_offset_ms,
                          last_set_split_trial, add_trial_breaks=True,
                          filename_to_extra_args=None):
    """ extra_args should be dict filename -> args that are different for
    this file."""
    default_args = deepcopy(locals())
    sets = []

    if filename_to_extra_args is not None:
        for filename_with_args in filename_to_extra_args:
            assert filename_with_args in filenames

    for i_file, filename in enumerate(filenames):
        this_args = copy(default_args)
        if filename_to_extra_args is not None and (
                filename in filename_to_extra_args):
            for key in filename_to_extra_args[filename]:
                assert key in this_args
                this_args[key] = filename_to_extra_args[filename][key]
                assert key != 'last_set_split_trial', "Does not make sense :)"
        marker_segmenter = MarkerSegmenter(segment_ival=[
            this_args['trial_start_offset_ms'], 
            this_args['trial_stop_offset_ms']],
            marker_def=this_args['marker_def'],
            trial_classes=this_args['trial_classes'],
            end_marker_def=this_args['end_marker_def'])
        trial_break_adder = AddTrialBreaks(min_length_ms=this_args['min_break_length_ms'],
            max_length_ms=this_args['max_break_length_ms'], 
            start_offset_ms=this_args['break_start_offset_ms'], 
            stop_offset_ms=this_args['break_stop_offset_ms'],
            start_marker_def=this_args['marker_def'],
            end_marker_def=this_args['end_marker_def'])
        if (i_file < len(filenames) - 1) or (
                this_args['last_set_split_trial'] is None):
            segmenters = [marker_segmenter,]
        else:
            segmenters = [marker_segmenter,
             RestrictTrialRange(0,this_args['last_set_split_trial'])]
        if this_args['add_trial_breaks']:
            segmenters.append(trial_break_adder)
        segmenter  = PipelineSegmenter(segmenters)
        cnt_set = SetWithMarkers(BBCIDataset(filename,
                              load_sensor_names=this_args['sensor_names']),
                  this_args['cnt_preprocessors'],
                  segmenter)        
        sets.append(cnt_set)

    # add last set last part as test set if you split apart last set
    # we use that this_args is now from last set already
    if last_set_split_trial is not None:
        segmenters = [marker_segmenter,
             RestrictTrialRange(last_set_split_trial,None),]
        if this_args['add_trial_breaks']:
            segmenters.append(trial_break_adder)
        segmenter  = PipelineSegmenter(segmenters)
        cnt_set = SetWithMarkers(BBCIDataset(filenames[-1], # again last file needed
                              load_sensor_names=this_args['sensor_names']),
                  this_args['cnt_preprocessors'],
                  segmenter)
        sets.append(cnt_set)
    dataset = CombinedSet(sets)
    return dataset

def construct_folder_combined_set(folder_names, sensor_names, cnt_preprocessors,
                            marker_def, end_marker_def, trial_classes,
                          trial_start_offset_ms, trial_stop_offset_ms,
                           min_break_length_ms, max_break_length_ms,
                          break_start_offset_ms, break_stop_offset_ms,
                          last_set_split_trial, add_trial_breaks=True,
                          folder_name_to_extra_args=None):
    """ extra_args should be dict filename -> args that are different for
    this file."""
    default_args = deepcopy(locals())
    sets = []

    if folder_name_to_extra_args is not None:
        for folder_name_with_args in folder_name_to_extra_args:
            assert folder_name_with_args in folder_names

    for i_folder, folder_name in enumerate(folder_names):
        this_args = copy(default_args)
        if folder_name_to_extra_args is not None and (
                folder_name in folder_name_to_extra_args):
            for key in folder_name_to_extra_args[folder_name]:
                assert key in this_args
                this_args[key] = folder_name_to_extra_args[folder_name][key]
                assert key != 'last_set_split_trial', "Does not make sense :)"
        marker_segmenter = MarkerSegmenter(segment_ival=[
            this_args['trial_start_offset_ms'], 
            this_args['trial_stop_offset_ms']],
            marker_def=this_args['marker_def'],
            trial_classes=this_args['trial_classes'],
            end_marker_def=this_args['end_marker_def'])
        trial_break_adder = AddTrialBreaks(min_length_ms=this_args['min_break_length_ms'],
            max_length_ms=this_args['max_break_length_ms'], 
            start_offset_ms=this_args['break_start_offset_ms'], 
            stop_offset_ms=this_args['break_stop_offset_ms'],
            start_marker_def=this_args['marker_def'],
            end_marker_def=this_args['end_marker_def'])
        if (i_folder < len(folder_names) - 1) or (
                this_args['last_set_split_trial'] is None):
            segmenters = [marker_segmenter,]
        else:
            segmenters = [marker_segmenter,
             RestrictTrialRange(0,this_args['last_set_split_trial'])]
        if this_args['add_trial_breaks']:
            segmenters.append(trial_break_adder)
        segmenter  = PipelineSegmenter(segmenters)
        filenames = sorted(glob(os.path.join(folder_name, '*.BBCI.mat')))
        cnt_set = SetWithMarkers(MultipleBBCIDataset(filenames,
                              load_sensor_names=this_args['sensor_names']),
                  this_args['cnt_preprocessors'],
                  segmenter)        
        sets.append(cnt_set)

    # add last set last part as test set if you split apart last set
    # we use that this_args is now from last set already
    if last_set_split_trial is not None:
        segmenters = [marker_segmenter,
             RestrictTrialRange(last_set_split_trial,None),]
        if this_args['add_trial_breaks']:
            segmenters.append(trial_break_adder)
        segmenter  = PipelineSegmenter(segmenters)
        cnt_set = SetWithMarkers(BBCIDataset(filenames[-1], # again last file needed
                              load_sensor_names=this_args['sensor_names']),
                  this_args['cnt_preprocessors'],
                  segmenter)
        sets.append(cnt_set)
    dataset = CombinedSet(sets)
    return dataset