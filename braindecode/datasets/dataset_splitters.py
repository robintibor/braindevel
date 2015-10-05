from abc import ABCMeta, abstractmethod
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from sklearn.cross_validation import KFold
import numpy as np
from copy import deepcopy
from collections import OrderedDict

class DatasetTrainValidTestSplitter():
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

class DatasetSingleFoldSplitter(DatasetTrainValidTestSplitter):
    def __init__(self, dataset, num_folds, test_fold_nr):
        self.dataset = dataset
        self.num_folds = num_folds
        self.test_fold_nr = test_fold_nr
        
    def ensure_dataset_is_loaded(self):
        if (hasattr(self.dataset, '_data_not_loaded_yet') and 
            self.dataset._data_not_loaded_yet):
            self.dataset.load()
        
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
        valid_fold_nr = self.test_fold_nr - 1
        folds = list(KFold(num_trials, n_folds=self.num_folds, shuffle=False))
        # [1] needed as it is always a split of whole dataset into train/test
        # indices
        test_fold = folds[self.test_fold_nr][1]
        valid_fold = folds[valid_fold_nr][1]
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
        
    def free_memory_if_reloadable(self):
        if hasattr(self.dataset, 'reload'):
            del self.dataset.X
            
    def reload_data(self):
        if hasattr(self.dataset, 'reload'):
            self.dataset.reload()
        

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
        valid_fold_nr = -1 # always use last fold as validation fold
        valid_fold = folds[valid_fold_nr][1]
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
        