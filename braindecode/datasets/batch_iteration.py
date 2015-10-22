from sklearn.cross_validation import KFold
from numpy.random import RandomState

class BalancedBatchIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.rng = RandomState(328774)
    
    def get_train_batches(self, train_set):
        num_trials = train_set.get_topological_view().shape[0]
        folds = KFold(num_trials, n_folds=num_trials // self.batch_size, 
                      random_state=self.rng, shuffle=True)
        # We will use the test folds as our mini-batches,
        # training fold indices are completely ignored here
        # test folds should all be distinct and together be the complete set
        for f in folds:
            trial_inds = f[1] # f[1] is the "test fold"
            yield (train_set.get_topological_view()[trial_inds],
                train_set.y[trial_inds])
            
    def reset_rng(self):
        self.rng = RandomState(328774)
        
    


def get_balanced_batches(num_trials, batch_size, rng, shuffle=True):
    folds = KFold(num_trials, n_folds=num_trials // batch_size, 
                  shuffle=shuffle, random_state=rng)
    test_folds = [f[1] for f in folds] 
    return test_folds
