import numpy as np
import scipy.stats

def stop_by_same_pred_repeated(pred_samples, preds,
        n_same_preds):
    this_pred_label_samples = []
    this_pred_labels = []
    i_pred = 0
    pred_buffer = []
    while i_pred < len(pred_samples):
        # +1 to make into 1-based indexing
        this_pred_label = np.argmax(preds[i_pred]) + 1
        pred_buffer.append(this_pred_label)
        unique_preds = np.unique(pred_buffer[-n_same_preds:])
        if len(unique_preds) == 1 and (len(pred_buffer) >= n_same_preds):
            this_pred_labels.append(unique_preds[0])
            this_pred_label_samples.append(pred_samples[i_pred])
            pred_buffer = []
        i_pred +=1
    return this_pred_label_samples, this_pred_labels

def stop_by_ttest_vs_all(pred_samples, preds, p_threshold=0.1, n_min_preds=5,
        max_buffer_length=30):
    
    this_pred_label_samples = []
    this_pred_labels = []
    i_pred = 0
    pred_buffer = np.zeros((0, preds.shape[1]))
    while i_pred < len(pred_samples):
        pred_buffer = np.append(pred_buffer, preds[i_pred][np.newaxis], axis=0)
        pred_buffer = pred_buffer[-max_buffer_length:]
        if len(pred_buffer) > n_min_preds:
            i_highest_class = np.argmax(np.mean(pred_buffer, axis=0))
            # do the ttest
            highest_class_preds = pred_buffer[:,i_highest_class]
            i_other_classes = np.setdiff1d(range(preds.shape[1]),
                [i_highest_class])
            other_preds = pred_buffer[:,i_other_classes].flatten()
            _, p_val = scipy.stats.ttest_ind(highest_class_preds,other_preds,
                equal_var=False)
            assert not np.isnan(p_val)
            if p_val < p_threshold:
                # +1 for 1-based indexing
                this_pred_labels.append(i_highest_class + 1)
                this_pred_label_samples.append(pred_samples[i_pred])
                pred_buffer = np.zeros((0, preds.shape[1]))
        i_pred += 1
    return this_pred_label_samples, this_pred_labels

def stop_by_ttest_pairwise(pred_samples, preds, p_threshold=0.1, n_min_preds=5,
        max_buffer_length=30):
    this_pred_label_samples = []
    this_pred_labels = []
    i_pred = 0
    pred_buffer = np.zeros((0, preds.shape[1]))
    while i_pred < len(pred_samples):
        pred_buffer = np.append(pred_buffer, preds[i_pred][np.newaxis], axis=0)
        pred_buffer = pred_buffer[-max_buffer_length:]
        if len(pred_buffer) > n_min_preds:
            i_highest_class = np.argmax(np.mean(pred_buffer, axis=0))
            max_p_val = 0
            # do the ttests
            for i_class in np.setdiff1d(range(5), [i_highest_class]):
                _, p_val = scipy.stats.ttest_rel(pred_buffer[:,i_class],
                                             pred_buffer[:,i_highest_class])
                max_p_val = max(p_val, max_p_val)
            assert not np.isnan(max_p_val)
            if max_p_val < p_threshold:
                # +1 for 1-based indexing
                this_pred_labels.append(i_highest_class + 1)
                this_pred_label_samples.append(pred_samples[i_pred])
                pred_buffer = np.zeros((0, preds.shape[1]))
        i_pred += 1
    return this_pred_label_samples, this_pred_labels

def stop_by_ttest_threshold(pred_samples, preds, p_threshold=0.1,
    pred_threshold=0.9,
    n_min_preds=5,
        max_buffer_length=30):
    this_pred_label_samples = []
    this_pred_labels = []
    i_pred = 0
    pred_buffer = np.zeros((0, preds.shape[1]))
    while i_pred < len(pred_samples):
        pred_buffer = np.append(pred_buffer, preds[i_pred][np.newaxis], axis=0)
        pred_buffer = pred_buffer[-max_buffer_length:]
        if len(pred_buffer) > n_min_preds:
            i_highest_class = np.argmax(np.mean(pred_buffer, axis=0))
            highest_class_preds = pred_buffer[:,i_highest_class]
            if np.mean(highest_class_preds) > pred_threshold:
                _, p_val = scipy.stats.ttest_1samp(highest_class_preds, pred_threshold)
                if p_val < p_threshold:
                    # +1 for 1-based indexing
                    this_pred_labels.append(i_highest_class + 1)
                    this_pred_label_samples.append(pred_samples[i_pred])
                    pred_buffer = np.zeros((0, preds.shape[1]))
        i_pred += 1
    return this_pred_label_samples, this_pred_labels

def evaluate_online_preds(pred_samples, pred_labels, labels,
        pred_offset):
    """
    Assumes labels and pred_labels are 1-based
    Assumes label 0 is break and label 5 is rest class and treats both as break.
        
    """
    # Set break to rest
    pred_labels = np.array(pred_labels)
    pred_samples = np.array(pred_samples)
    labels = np.array(labels)
    labels[labels == 0] = 5
    
    bounds = np.flatnonzero(np.diff(labels)) + 1
    # add a bound at end of dataset
    bounds = np.append(bounds, len(labels))
    missed_predictions = 0
    correct_predictions = 0
    false_trial_predictions = 0
    false_break_predictions = 0
    #print("bounds", bounds)

    for i_bound in xrange(len(bounds[:-1])):
        start = bounds[i_bound]
        stop = bounds[i_bound+1]
        this_trial_label = np.int32(labels[start])
        #print("this trial label", this_trial_label)
        # change break to rest (and ignore rest afterwards)
        assert (this_trial_label >= 1)  and (this_trial_label <= 5)
        if this_trial_label == 5:
            # for breaks, allow first ones until offset to be incorrect
            # but stop right at end of break
            i_this_pred_start = np.searchsorted(pred_samples,
                start + pred_offset, 'left')
            i_this_pred_stop = np.searchsorted(pred_samples, stop, 'left')
        else:
            # for trials, look until end of trial + offset
            i_this_pred_start = np.searchsorted(pred_samples, start, 'left')
            i_this_pred_stop = np.searchsorted(pred_samples,
                                           stop + pred_offset, 'left')
        #print "this pred start", i_this_pred_start
        #print "this pred stop", i_this_pred_stop
        this_trial_pred_labels = pred_labels[i_this_pred_start:i_this_pred_stop]
        this_trial_pred_labels = np.unique(this_trial_pred_labels)
        #print "this trial pred labels", this_trial_pred_labels
        # ignore rest predictions
        this_trial_pred_labels = np.setdiff1d(this_trial_pred_labels , [5])
        if (this_trial_label != 5):
            if (this_trial_label in this_trial_pred_labels):
                correct_predictions += 1
            else:
                missed_predictions += 1

        # now remove correct prediction to check for false (positive) predictions 
        this_trial_pred_labels = np.setdiff1d(this_trial_pred_labels ,
            [this_trial_label])
        if len(this_trial_pred_labels) > 0:
            if this_trial_label != 5:
                false_trial_predictions += 1
            else:
                false_break_predictions += 1
    
    return correct_predictions, missed_predictions, false_trial_predictions, false_break_predictions

