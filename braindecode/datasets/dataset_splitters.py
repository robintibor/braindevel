from abc import ABCMeta, abstractmethod
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from sklearn.cross_validation import KFold
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from numpy.random import RandomState

class DatasetTrainValidTestSplitter(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def split_into_train_valid_test(self):
        raise NotImplementedError("Subclass needs to implement this")

    @abstractmethod
    def ensure_dataset_is_loaded(self):
        raise NotImplementedError("Subclass needs to implement this")

    @abstractmethod
    def free_memory_if_reloadable(self):
        raise NotImplementedError("Subclass needs to implement this")

    @abstractmethod
    def reload_data(self):
        raise NotImplementedError("Subclass needs to implement this")
    
    def set_dataset(self, dataset):
        self.dataset = dataset

class SingleDatasetSplitter(DatasetTrainValidTestSplitter):
    """Base class for single dataset splitters."""
    def ensure_dataset_is_loaded(self):
        self.dataset.ensure_is_loaded()
        
    def free_memory_if_reloadable(self):
        if self.dataset.reloadable:
            self.dataset.free_memory()
            
    def reload_data(self):
        if self.dataset.reloadable:
            self.dataset.reload()

class DatasetFixedTrialSplitter(SingleDatasetSplitter):
    def __init__(self, dataset, n_train_trials, valid_set_fraction):
        self.dataset = dataset
        self.n_train_trials = n_train_trials
        self.valid_set_fraction = valid_set_fraction
        
    def split_into_train_valid_test(self):
        """ Split into train valid test by splitting 
        dataset into num folds, test fold nr should be given, 
        valid fold will be the one immediately before the test fold, 
        train folds the remaining 8 folds"""
        assert self.dataset.view_converter.axes[0] == 'b'
        assert hasattr(self.dataset, 'X') # needs to be loaded already
        num_trials = self.dataset.get_topological_view().shape[0]
        assert num_trials > self.n_train_trials
        
        # split train into train and valid
        # valid is at end, just subtract -1 because of zero-based indexing
        i_last_valid_trial = self.n_train_trials - 1
        i_last_train_trial = self.n_train_trials - 1 - int(
            self.valid_set_fraction * self.n_train_trials)
        # always +1 since ranges are exclusive the end index(!)
        train_fold = range(i_last_train_trial+1)
        valid_fold = range(i_last_train_trial+1,i_last_valid_trial+1)
        test_fold = range(i_last_valid_trial+1, num_trials)
        

        # Make sure there are no overlaps and we have all possible trials
        # assigned
        assert np.intersect1d(valid_fold, test_fold).size == 0
        assert np.intersect1d(train_fold, test_fold).size == 0
        assert np.intersect1d(train_fold, valid_fold).size == 0
        assert set(np.concatenate((train_fold, valid_fold, test_fold))) == set(range(num_trials))
        train_set= DenseDesignMatrix(
            topo_view=self.dataset.get_topological_view()[train_fold], 
            y=self.dataset.y[train_fold], 
            axes=self.dataset.view_converter.axes)
        valid_set= DenseDesignMatrix(
            topo_view=self.dataset.get_topological_view()[valid_fold],
            y=self.dataset.y[valid_fold], 
            axes=self.dataset.view_converter.axes)
        test_set= DenseDesignMatrix(
            topo_view=self.dataset.get_topological_view()[test_fold], 
            y=self.dataset.y[test_fold],
            axes=self.dataset.view_converter.axes)
        # make ordered dict to make it easier to iterate, i.e. for logging
        return OrderedDict([('train', train_set),
            ('valid', valid_set), 
            ('test', test_set)])

class DatasetSingleFoldSplitter(SingleDatasetSplitter):
    def __init__(self, dataset, num_folds=10, i_test_fold=-1,
            shuffle=False):
        self.dataset = dataset
        self.num_folds = num_folds
        self.i_test_fold = i_test_fold
        self.shuffle=shuffle
        
    def split_into_train_valid_test(self):
        """ Split into train valid test by splitting 
        dataset into num folds, test fold nr should be given, 
        valid fold will be the one immediately before the test fold, 
        train folds the remaining 8 folds"""
        assert self.dataset.view_converter.axes[0] == 'b'
        assert hasattr(self.dataset, 'X') # needs to be loaded already
        num_trials = self.dataset.get_topological_view().shape[0]
        # also works in case test fold nr is 0 as it will just take -1 
        # which is fine last fold)
        i_valid_fold = self.i_test_fold - 1
        
        if self.shuffle:
            rng = RandomState(729387987) #TODO: check it rly leads to same split when being called twice
            folds = list(KFold(num_trials, n_folds=self.num_folds,
                shuffle=self.shuffle, random_state=rng))
        else:
            folds = list(KFold(num_trials, n_folds=self.num_folds,
                shuffle=False))
        # [1] needed as it is always a split of whole dataset into train/test
        # indices
        test_fold = folds[self.i_test_fold][1]
        valid_fold = folds[i_valid_fold][1]
        full_fold = range(num_trials)
        train_fold = np.setdiff1d(full_fold, 
            np.concatenate((valid_fold, test_fold)))

        # Make sure there are no overlaps and we have all possible trials
        # assigned
        assert np.intersect1d(valid_fold, test_fold).size == 0
        assert np.intersect1d(train_fold, test_fold).size == 0
        assert np.intersect1d(train_fold, valid_fold).size == 0
        assert set(np.concatenate((train_fold, valid_fold, test_fold))) == set(range(num_trials))
        train_set= DenseDesignMatrix(
            topo_view=self.dataset.get_topological_view()[train_fold], 
            y=self.dataset.y[train_fold], 
            axes=self.dataset.view_converter.axes)
        valid_set= DenseDesignMatrix(
            topo_view=self.dataset.get_topological_view()[valid_fold],
            y=self.dataset.y[valid_fold], 
            axes=self.dataset.view_converter.axes)
        test_set= DenseDesignMatrix(
            topo_view=self.dataset.get_topological_view()[test_fold], 
            y=self.dataset.y[test_fold], 
            axes=self.dataset.view_converter.axes)
        # make ordered dict to make it easier to iterate, i.e. for logging
        return OrderedDict([('train', train_set),
            ('valid', valid_set), 
            ('test', test_set)])

class DatasetTwoFileSingleFoldSplitter(DatasetTrainValidTestSplitter):
    def __init__(self, train_set, test_set, num_folds):
        """ num_folds here just determines size of valid set and train set.
        E.g. if num_folds is 5, 
        valid set will have 20% of the train set trials"""
        self.train_set = train_set
        self.test_set = test_set
        self.num_folds = num_folds

    def ensure_dataset_is_loaded(self):
        for dataset in [self.train_set, self.test_set]:
            if (hasattr(dataset, '_data_not_loaded_yet') and 
                    dataset._data_not_loaded_yet):
                dataset.load()

    def split_into_train_valid_test(self):
        """ Split into train valid test by splitting 
        train dataset into num folds, test fold nr should be given, 
        valid fold will be the one immediately before the test fold, 
        train folds the remaining 8 folds"""
        assert self.train_set.view_converter.axes[0] == 'b'
        assert hasattr(self.train_set, 'X') # needs to be loaded already
        num_trials = self.train_set.get_topological_view().shape[0]
        folds = list(KFold(num_trials, n_folds=self.num_folds, shuffle=False))
        i_valid_fold = -1 # always use last fold as validation fold
        valid_fold = folds[i_valid_fold][1]
        full_fold = range(num_trials)
        train_fold = np.setdiff1d(full_fold, valid_fold)

        # Make sure there are no overlaps and we have all possible trials
        # assigned
        assert np.intersect1d(train_fold, valid_fold).size == 0
        assert set(np.concatenate((train_fold, valid_fold))) == set(range(num_trials))
        train_set= DenseDesignMatrix(
            topo_view=self.train_set.get_topological_view()[train_fold], 
            y=self.train_set.y[train_fold], 
            axes=self.train_set.view_converter.axes)
        valid_set= DenseDesignMatrix(
            topo_view=self.train_set.get_topological_view()[valid_fold],
            y=self.train_set.y[valid_fold], 
            axes=self.train_set.view_converter.axes)
        test_set= deepcopy(self.test_set) # test set maybe preprocessed=modified by caller
        return {'train': train_set, 
            'valid': valid_set, 
            'test': test_set}

    def free_memory_if_reloadable(self):
        for dataset in [self.train_set, self.test_set]:
            if hasattr(dataset, 'reload'):
                del dataset.X

    def reload_data(self):
        for dataset in [self.train_set, self.test_set]:
            if hasattr(dataset, 'reload'):
                dataset.reload()
        
class PreprocessedSplitter(object):
    def __init__(self, dataset_splitter, preprocessor):
        self.dataset_splitter = dataset_splitter
        self.preprocessor = preprocessor

    def get_train_valid_test(self):
        datasets = self.dataset_splitter.split_into_train_valid_test()
        self.dataset_splitter.free_memory_if_reloadable()
        if self.preprocessor is not None:
            self.preprocessor.apply(datasets['train'], can_fit=True)
            self.preprocessor.apply(datasets['valid'], can_fit=False)
            self.preprocessor.apply(datasets['test'], can_fit=False)
        return datasets

    def get_train_merged_valid_test(self):
        self.dataset_splitter.reload_data()
        this_datasets = self.dataset_splitter.split_into_train_valid_test()
        self.dataset_splitter.free_memory_if_reloadable()
        train_valid_set = self.concatenate_sets(this_datasets['train'],
            this_datasets['valid'])
        test_set = this_datasets['test']
        train_set_num_trials = len(this_datasets['train'].y)
        del this_datasets['train']
        if self.preprocessor is not None:
            self.preprocessor.apply(train_valid_set, can_fit=True)
            self.preprocessor.apply(test_set, can_fit=False)
        _, valid_set = self.split_sets(train_valid_set, 
            train_set_num_trials, len(this_datasets['valid'].y))
        # train valid is the new train set!!
        return {'train': train_valid_set, 'valid': valid_set, 
            'test': test_set}

    def concatenate_sets(self, first_set, second_set):
        """ Concatenates topo views and y(targets)"""
        assert first_set.view_converter.axes == second_set.view_converter.axes,\
            "first set and second set should have same axes ordering"
        assert first_set.view_converter.axes[0] == 'b', ("Expect batch axis "
            "as first axis")
        merged_topo_view = np.concatenate((first_set.get_topological_view(),
            second_set.get_topological_view()))
        merged_y = np.concatenate((first_set.y, second_set.y)) 
        merged_set = DenseDesignMatrix(
            topo_view=merged_topo_view,
            y=merged_y,
            axes=first_set.view_converter.axes)
        return merged_set
    
    def split_sets(self, full_set, split_index, split_to_end_num):
        """ Assumes that full set may be doubled or tripled in size
        and split index refers to original size. So
        if we originally had 100 trials (set1) + 20 trials (set2) 
        merged to 120 trials, we get a split index of 100.
        If we later have 360 trials we assume that the 360 trials 
        consist of:
        100 trials set1 + 20 trials set2 + 100 trials set1 + 20 trials set2
        + 100 trials set1 + 20 trials set2
        (and not 300 trials set1 + 60 trials set2)"""
        full_topo = full_set.get_topological_view()
        full_y = full_set.y
        original_full_len = split_index + split_to_end_num
        topo_first = full_topo[:split_index]
        y_first = full_y[:split_index]
        topo_second = full_topo[split_index:original_full_len]
        y_second = full_y[split_index:original_full_len]
        next_start = original_full_len
        # Go through possibly appended transformed copies of dataset
        # If preprocessors did not change dataset size, this is not 
        # necessary
        for next_split in xrange(next_start + split_index, 
                len(full_set.y), original_full_len):
            assert False, "Please check/test this code again if you need it"
            next_end = next_split + split_to_end_num
            topo_first = np.concatenate((topo_first, 
                full_topo[next_start:next_split]))
            y_first = np.concatenate((y_first, full_y[next_start:next_split]))
            topo_second = np.concatenate((topo_second, 
                full_topo[next_split:next_end]))
            y_second =  np.concatenate((y_second, full_y[next_split:next_end]))
            next_start = next_end
        first_set = DenseDesignMatrix(
            topo_view=topo_first,
            y=y_first,
            axes=full_set.view_converter.axes)
        second_set = DenseDesignMatrix(
            topo_view=topo_second,
            y=y_second,
            axes=full_set.view_converter.axes)
        return first_set, second_set

    def set_dataset(self, dataset):
        self.dataset_splitter.set_dataset(dataset)