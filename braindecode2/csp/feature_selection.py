import numpy as np

def select_features(labels, trials, num_features, k=0.5):
    # TODO: delete this?
    """Trials in #trialsx#features format."""
    assert False, "rewrite or delete"
    assert num_features <= trials.shape[1], ("Cannot select more features "
        "than existing features")
    selected_feature_mask = np.array([False] * trials.shape[1])
    while np.sum(selected_feature_mask) < num_features:
        best_feature_mask = None
        lowest_entropy = float('inf')
        unselected_feature_inds = np.flatnonzero(
            np.logical_not(selected_feature_mask))
        invcov_trials = np.linalg.inv(np.atleast_2d(np.cov(trials.T)))
        for feature_i in unselected_feature_inds:
            next_feature_mask = selected_feature_mask.copy()
            next_feature_mask[feature_i] = True
            next_feature_inds = np.flatnonzero(next_feature_mask)
            next_trials = trials[:, next_feature_mask]
            try:
                entropy = cond_entropy_class_given_features(labels,
                    trials[:,next_feature_mask], k=k, 
                    invcov_trials=invcov_trials[next_feature_inds][:,next_feature_inds])
            except np.linalg.LinAlgError:
                entropy = float('inf')
            if (entropy < lowest_entropy):
                lowest_entropy = entropy
                best_feature_mask = next_feature_mask
            
        if np.isfinite(lowest_entropy):
            selected_feature_mask = best_feature_mask
        else:
            # TODO: handle this properly?
            selected_feature_mask[unselected_feature_inds[0]] = True
    return np.flatnonzero(selected_feature_mask)

def cond_entropy_class_given_features(labels, trials, k=0.5, 
    exclude_trial_itself=False, invcov_trials=None):
    """Entropy of estimated probability distribution for class given the features.
    exclude_trial_itself: When computing conditional probabilites for each trial,
    exclude trial itself to estimate the conditional probability distribution. 
    Might make it more robust(?).
    Trials in #trialsx#features format."""
    unique_labels = np.unique(labels)
    assert len(unique_labels) == 2, "only implemented for two class case"
    # TODELAY: replace with ledoit wolf covariance estimator?
    if invcov_trials is None:
        invcov_trials = np.linalg.inv(np.atleast_2d(np.cov(trials.T)))
    #invcov_trials = 1 / (np.atleast_2d(np.cov(trials.T)))
    if (exclude_trial_itself):
        cond_probs = [cond_prob_label_given_trial(unique_labels[0], t, 
                                                  np.delete(labels, i), 
                                                  np.delete(trials, i, axis=0), 
                                                  k=k, invcov=invcov_trials)
                      for i,t in enumerate(trials)]
    else:
        cond_probs = [cond_prob_label_given_trial(unique_labels[0], t, labels, trials, 
                                                  k=k, invcov=invcov_trials) 
                      for i,t in enumerate(trials)]
        
    neg_probs = 1 - np.array(cond_probs)
    entropy_sum_parts = (neg_probs * np.log(neg_probs)) + (cond_probs * np.log(cond_probs))
    entropy = -np.sum(entropy_sum_parts) / len(entropy_sum_parts)
    return entropy

def cond_prob_label_given_trial(label, trial, labels, trials, k=0.5, invcov=None):
    """ Calculates conditional probability of a label given a trial.
    You can supply inverse covariance matrix to prevent multiple computations.
    Otherwise it will be computed from trials inside the function.
    Trials in #trialsx#features format."""
    #http://www.researchgate.net/profile/Nojun_Kwak/publication/3193472_Input_feature_selection_by_mutual_information_based_on_Parzen_window/links/02e7e51789978d9299000000.pdf
    if trials.ndim == 1:
        trials = trials[:, np.newaxis]
    if invcov is None:
        cov_trials = np.cov(trials.T)
        # TODELAY: replace with ledoit wolf covariance estimator?
        invcov = np.linalg.inv(np.atleast_2d(cov_trials))
    #invcov = 1 /np.atleast_2d(cov_trials)
    # compute differene of given trial to remaining trials for all features 
    diffs = trial - np.atleast_2d(trials)
    # variable names may be misleading :) I don't fully understand the logic :)
    #squared_covared_diffs = [np.dot(np.dot(diff, invcov), diff) for diff in diffs]
    squared_covared_diffs = np.sum(np.dot(diffs,invcov) * diffs, axis=1)
    h = k * np.log(len(labels))
    
    gaussian_diffs = np.exp(-np.array(squared_covared_diffs) / (2 * (h*h)))
    wanted_inds = np.array(labels) == label
    return np.sum(gaussian_diffs[wanted_inds]) / np.sum(gaussian_diffs)