from abc import ABCMeta, abstractmethod
from sklearn.cross_validation import KFold
import numpy as np
from collections import OrderedDict
from numpy.random import RandomState
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
from copy import deepcopy
import logging
log = logging.getLogger(__name__)

class TrainValidTestSplitter(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def split_into_train_valid_test(self, dataset):
        raise NotImplementedError("Subclass needs to implement this")
    
class SeveralSetsSplitter(TrainValidTestSplitter):
    """
    Split multiple sets, last set as test, second last (if exists) as valid,
    first as training. In case of only two sets, split off part of first set
    as valid set.
    Parameters
    ----------
    valid_set_fraction : int
        In case of only two sets, how much of the
        first (training set) should be split off for validation.
    """
    def __init__(self, valid_set_fraction=0.1,
        use_test_as_valid=False):
        self.use_test_as_valid = use_test_as_valid
        self.valid_set_fraction = 0.1

    def split_into_train_valid_test(self, sets_container):
        # Otherwise rewrite code should not be too hard..
        assert len(sets_container.sets) > 1, "Expect atlest 2 sets here..."
        # merge all sets before last into one...
        # left out, final set, is test set
        # if we have more than 2 sets use secondlastset as valid set

        # remember python indexing is 0-based..
        i_test_set = len(sets_container.sets) - 1 
        # in case of just 2 sets valid will get splitted in train and valid
        # in else clause later
        if self.use_test_as_valid:
            i_valid_set = i_test_set
        else:
            i_valid_set = i_test_set - 1
        if len(sets_container.sets) > 2 or self.use_test_as_valid:
            # can split off train and valid as individual sets
            topos = [s.get_topological_view() for s in sets_container.sets[:i_valid_set]]
            ys = [s.y for s in sets_container.sets[:i_valid_set]]
            train_topo = np.concatenate(topos, axis=0)
            train_y = np.concatenate(ys, axis=0)
            train = DenseDesignMatrixWrapper(
                topo_view=train_topo, y=train_y,
                axes=sets_container.sets[0].view_converter.axes)
            valid = sets_container.sets[i_valid_set]
            test = sets_container.sets[i_test_set]
        else:
            # have to split off valid from first=train set
            topos = [s.get_topological_view() for s in sets_container.sets[:i_valid_set+1]]
            ys = [s.y for s in sets_container.sets[:i_valid_set+1]]
            full_topo = np.concatenate(topos, axis=0)
            full_y = np.concatenate(ys, axis=0)
            train_valid_set = DenseDesignMatrixWrapper(
                topo_view=full_topo, y=full_y,
                axes=sets_container.sets[0].view_converter.axes)
            train, valid = split_into_two_sets(train_valid_set, 
                last_set_fraction=self.valid_set_fraction)
            test = sets_container.sets[i_test_set]
        return OrderedDict([('train', train), ('valid',valid),('test', test)])

def split_into_two_sets(full_set, last_set_fraction):
    topo = full_set.get_topological_view()
    y = full_set.y
    n_trials = len(full_set.y)
    
    i_last_first_set_trial = n_trials - 1 - int(last_set_fraction * n_trials)
    
    first_set = DenseDesignMatrixWrapper(
        topo_view=topo[0:i_last_first_set_trial+1], 
        y=y[0:i_last_first_set_trial+1], 
        axes=full_set.view_converter.axes)
    second_set = DenseDesignMatrixWrapper(
        topo_view=topo[i_last_first_set_trial+1:], 
        y=y[i_last_first_set_trial+1:], 
        axes=full_set.view_converter.axes)
    
    return first_set, second_set

class FixedTrialSplitter(TrainValidTestSplitter):
    def __init__(self, n_train_trials, valid_set_fraction):
        self.n_train_trials = n_train_trials
        self.valid_set_fraction = valid_set_fraction
        
    def split_into_train_valid_test(self, dataset):
        """Split into train valid test by splitting
        dataset into num folds, test fold nr should be given,
        valid fold will be the one immediately before the test fold,
        train folds the remaining 8 folds
        """
        assert dataset.view_converter.axes[0] == 'b'
        assert hasattr(dataset, 'X') # needs to be loaded already
        n_trials = dataset.get_topological_view().shape[0]
        assert n_trials > self.n_train_trials
        
        # split train into train and valid
        # valid is at end, just subtract -1 because of zero-based indexing
        i_last_valid_trial = self.n_train_trials - 1
        i_last_train_trial = self.n_train_trials - 1 - int(
            self.valid_set_fraction * self.n_train_trials)
        # always +1 since ranges are exclusive the end index(!)
        train_fold = range(i_last_train_trial+1)
        valid_fold = range(i_last_train_trial+1,i_last_valid_trial+1)
        test_fold = range(i_last_valid_trial+1, n_trials)
        
        datasets = split_set_by_indices(dataset, train_fold, valid_fold,
            test_fold)

        return datasets

class SingleFoldSplitter(TrainValidTestSplitter):
    def __init__(self, n_folds=10, i_test_fold=-1,
            shuffle=False):
        self.n_folds = n_folds
        self.i_test_fold = i_test_fold
        self.shuffle=shuffle

    def split_into_train_valid_test(self, dataset):
        """Split into train valid test by splitting
        dataset into num folds, test fold nr should be given,
        valid fold will be the one immediately before the test fold,
        train folds the remaining 8 folds
        """
        assert dataset.view_converter.axes[0] == 'b'
        assert hasattr(dataset, 'X') # needs to be loaded already
        n_trials = dataset.get_topological_view().shape[0]
        # also works in case test fold nr is 0 as it will just take -1 
        # which is fine last fold)
        i_valid_fold = self.i_test_fold - 1
        
        if self.shuffle:
            rng = RandomState(729387987) #TODO: check it rly leads to same split when being called twice
            folds = list(KFold(n_trials, n_folds=self.n_folds,
                shuffle=self.shuffle, random_state=rng))
        else:
            folds = list(KFold(n_trials, n_folds=self.n_folds,
                shuffle=False))
        # [1] needed as it is always a split of whole dataset into train/test
        # indices
        test_fold = folds[self.i_test_fold][1]
        valid_fold = folds[i_valid_fold][1]
        full_fold = range(n_trials)
        train_fold = np.setdiff1d(full_fold,
            np.concatenate((valid_fold, test_fold)))

        datasets = split_set_by_indices(dataset, train_fold, valid_fold,
            test_fold)
        return datasets

def split_set_by_indices(dataset, train_fold, valid_fold, test_fold):
        n_trials = dataset.get_topological_view().shape[0]
        # Make sure there are no overlaps and we have all possible trials
        # assigned
        assert np.intersect1d(valid_fold, test_fold).size == 0
        assert np.intersect1d(train_fold, test_fold).size == 0
        assert np.intersect1d(train_fold, valid_fold).size == 0
        assert (set(np.concatenate((train_fold, valid_fold, test_fold))) == 
            set(range(n_trials)))
        
        train_set = DenseDesignMatrixWrapper(
            topo_view=dataset.get_topological_view()[train_fold], 
            y=dataset.y[train_fold], 
            axes=dataset.view_converter.axes)
        valid_set = DenseDesignMatrixWrapper(
            topo_view=dataset.get_topological_view()[valid_fold], 
            y=dataset.y[valid_fold], 
            axes=dataset.view_converter.axes)
        test_set = DenseDesignMatrixWrapper(
            topo_view=dataset.get_topological_view()[test_fold], 
            y=dataset.y[test_fold], 
            axes=dataset.view_converter.axes)
        # make ordered dict to make it easier to iterate, i.e. for logging
        datasets = OrderedDict([('train', train_set), ('valid', valid_set), 
                ('test', test_set)])
        return datasets

def concatenate_sets(first_set, second_set):
    """Concatenates topo views and y(targets)
    """
    assert first_set.view_converter.axes == second_set.view_converter.axes,\
        "first set and second set should have same axes ordering"
    assert first_set.view_converter.axes[0] == 'b', ("Expect batch axis "
        "as first axis")
    merged_topo_view = np.concatenate((first_set.get_topological_view(),
        second_set.get_topological_view()))
    merged_y = np.concatenate((first_set.y, second_set.y)) 
    merged_set = DenseDesignMatrixWrapper(
        topo_view=merged_topo_view,
        y=merged_y,
        axes=first_set.view_converter.axes)
    return merged_set

class PreprocessedSplitter(object):
    def __init__(self, dataset_splitter, preprocessor):
        self.dataset_splitter = dataset_splitter
        self.preprocessor = preprocessor

    def get_train_valid_test(self, dataset):
        datasets = self.dataset_splitter.split_into_train_valid_test(dataset)
        if dataset.reloadable:
            dataset.free_memory()
        if self.preprocessor is not None:
            log.info("Preprocessing...")
            # lets make copy so that changes by preprocesosr are not permanent
            # i.e. still there after early stop
            datasets = deepcopy(datasets)
            self.preprocessor.apply(datasets['train'], can_fit=True)
            self.preprocessor.apply(datasets['valid'], can_fit=False)
            self.preprocessor.apply(datasets['test'], can_fit=False)
            log.info("Done.")
        return datasets

    def get_train_merged_valid_test(self, dataset):
        dataset.ensure_is_loaded()
        this_datasets = self.dataset_splitter.split_into_train_valid_test(dataset)
        if dataset.reloadable:
            dataset.free_memory()
        train_valid_set = concatenate_sets(this_datasets['train'],
            this_datasets['valid'])
        test_set = this_datasets['test']
        #preprocessing of train only so that later split is correct...
        if self.preprocessor is not None:
            log.info("Preprocessing original train to make sure split is correct")
            self.preprocessor.apply(this_datasets['train'], can_fit=False)
        n_train_set_trials = len(this_datasets['train'].y)
        del this_datasets['train']
        if self.preprocessor is not None:
            log.info("Preprocessing...")
            # lets make copy of test set just to be sure
            # probably unnecessary...
            # train_valid set should already be a new object
            # after concatenation call above...
            test_set = deepcopy(test_set)
            self.preprocessor.apply(train_valid_set, can_fit=True)
            self.preprocessor.apply(test_set, can_fit=False)
            #preprocessing of valid only so that later split is correct...
            # todelay: think if this is smart or maybe better just use valid set
            # below without the splitting
            self.preprocessor.apply(this_datasets['valid'], can_fit=False)
            log.info("Done.")
        _, valid_set = self.split_sets(train_valid_set, 
            n_train_set_trials, len(this_datasets['valid'].y))
        
        # In case you stored removed_ys, restore them for recovered valid set
        if hasattr(this_datasets['valid'], 'removed_ys'):
            valid_set.removed_ys = this_datasets['valid'].removed_ys
        # train valid is the new train set!!
        return {'train': train_valid_set, 'valid': valid_set, 
            'test': test_set}

    
    def split_sets(self, full_set, split_index, split_to_end_num):
        """Assumes that full set may be doubled or tripled in size
        and split index refers to original size. So
        if we originally had 100 trials (set1) + 20 trials (set2)
        merged to 120 trials, we get a split index of 100.
        If we later have 360 trials we assume that the 360 trials
        consist of:
        100 trials set1 + 20 trials set2 + 100 trials set1 + 20 trials set2
        + 100 trials set1 + 20 trials set2
        (and not 300 trials set1 + 60 trials set2)
        """
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
        first_set = DenseDesignMatrixWrapper(
            topo_view=topo_first,
            y=y_first,
            axes=full_set.view_converter.axes)
        second_set = DenseDesignMatrixWrapper(
            topo_view=topo_second,
            y=y_second,
            axes=full_set.view_converter.axes)
        return first_set, second_set

class KaggleTrainValidTestSplitter(TrainValidTestSplitter):
    def __init__(self, use_test_as_valid=False):
        self.use_test_as_valid = use_test_as_valid

    def split_into_train_valid_test(self, dataset):
        # make into 4d tensors
        i_valid_series = 6
        if self.use_test_as_valid:
            i_valid_series = 7
        X_train = np.concatenate(dataset.train_X_series[:i_valid_series])[:,:,np.newaxis,np.newaxis]
        X_valid = dataset.train_X_series[i_valid_series][:,:,np.newaxis,np.newaxis]
        X_test = dataset.train_X_series[7][:,:,np.newaxis,np.newaxis]
        
        y_train = np.concatenate(dataset.train_y_series[:i_valid_series])
        y_valid = dataset.train_y_series[i_valid_series]
        y_test = dataset.train_y_series[7]
    
        # create dense design matrix sets
        train_set = DenseDesignMatrixWrapper(
            topo_view=X_train, 
            y=y_train, axes=('b','c',0,1))
        valid_set = DenseDesignMatrixWrapper(
            topo_view=X_valid,
            y=y_valid, axes=('b','c',0,1))
        test_set = DenseDesignMatrixWrapper(
             topo_view=X_test,
             y=y_test, axes=('b','c',0,1))
        return OrderedDict([('train', train_set),
            ('valid', valid_set), 
            ('test',  test_set)])
        

class AllSubjectsKaggleTrainValidTestSplitter(TrainValidTestSplitter):
    def __init__(self, use_test_as_valid=False):
        self.use_test_as_valid = use_test_as_valid
        
    def split_into_train_valid_test(self, dataset):
        # First split all sets individually, then merge them
        one_set_splitter = KaggleTrainValidTestSplitter(
            use_test_as_valid=self.use_test_as_valid)
        splitted_sets = [one_set_splitter.split_into_train_valid_test(s) 
            for s in dataset.kaggle_sets]
        train_topos = [s['train'].get_topological_view() for s in splitted_sets]
        train_ys = [s['train'].y for s in splitted_sets]
        # create dense design matrix sets
        train_set = DenseDesignMatrixWrapper(
            topo_view=np.concatenate(train_topos, axis=0), 
            y=np.concatenate(train_ys, axis=0), axes=('b','c',0,1))
        
        valid_topos = [s['valid'].get_topological_view() for s in splitted_sets]
        valid_ys = [s['valid'].y for s in splitted_sets]
        valid_set = DenseDesignMatrixWrapper(
            topo_view=np.concatenate(valid_topos, axis=0), 
            y=np.concatenate(valid_ys, axis=0), axes=('b','c',0,1))
        
        test_topos = [s['test'].get_topological_view() for s in splitted_sets]
        test_ys = [s['test'].y for s in splitted_sets]
        test_set = DenseDesignMatrixWrapper(
            topo_view=np.concatenate(test_topos, axis=0), 
            y=np.concatenate(test_ys, axis=0), axes=('b','c',0,1))
        
        return OrderedDict([('train', train_set),
            ('valid', valid_set), 
            ('test',  test_set)])
    