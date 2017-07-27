from collections import OrderedDict
from braindecode2.datasets.signal_target import SignalAndTarget


class TrainValidTestSplitter(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def split_into_train_valid_test(self, dataset):
        raise NotImplementedError("Subclass needs to implement this")

class SingleFoldSplitter(TrainValidTestSplitter):
    def __init__(self, n_folds=10, i_test_fold=-1,
                 shuffle=False):
        self.n_folds = n_folds
        self.i_test_fold = i_test_fold
        self.shuffle = shuffle

    def split_into_train_valid_test(self, dataset):
        """Split into train valid test by splitting
        dataset into num folds, test fold nr should be given,
        valid fold will be the one immediately before the test fold,
        train folds the remaining 8 folds
        """
        n_trials = dataset.X.shape[0]
        # also works in case test fold nr is 0 as it will just take -1
        # which is fine last fold)
        i_valid_fold = self.i_test_fold - 1

        if self.shuffle:
            rng = RandomState(
                729387987)  # TODO: check it rly leads to same split when being called twice
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
    n_trials = dataset.X.shape[0]
    # Make sure there are no overlaps and we have all possible trials
    # assigned
    assert np.intersect1d(valid_fold, test_fold).size == 0
    assert np.intersect1d(train_fold, test_fold).size == 0
    assert np.intersect1d(train_fold, valid_fold).size == 0
    assert (set(np.concatenate((train_fold, valid_fold, test_fold))) ==
            set(range(n_trials)))

    train_set = SignalAndTarget(
        X=dataset.X[train_fold],
        y=dataset.y[train_fold])
    valid_set = SignalAndTarget(
        X=dataset.X[valid_fold],
        y=dataset.y[valid_fold])
    test_set = SignalAndTarget(
        X=dataset.X[test_fold],
        y=dataset.y[test_fold])
    # make ordered dict to make it easier to iterate, i.e. for logging
    datasets = OrderedDict([('train', train_set), ('valid', valid_set),
                            ('test', test_set)])
    return datasets