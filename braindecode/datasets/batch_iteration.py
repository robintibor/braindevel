from sklearn.cross_validation import KFold

def get_balanced_batches(num_trials, batch_size, rng, shuffle=True):
    # We will use the test folds as our mini-batches,
    # training fold indices are completely ignored here
    folds = KFold(num_trials, n_folds=num_trials // batch_size, 
                  shuffle=shuffle, random_state=rng)
    test_folds = [f[1] for f in folds] 
    return test_folds
