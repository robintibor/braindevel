import numpy as np
import seaborn
import matplotlib.pyplot as plt
from braindecode.analysis.plot_util import plot_mean_and_std

def compute_center_events(y):
    center_events_per_class = [compute_center_events_for_class(y,i_class)
                              for i_class in range(6)]
    return np.array(center_events_per_class)
def compute_center_events_for_class(y,i_class):
        diffs = np.diff(y[:,i_class])
        boundary_inds = np.flatnonzero(diffs)
        center_events = [int(np.mean([a,b])) for a,b in boundary_inds.reshape(-1,2)]
        return center_events
    
def show_mean_preds_around_events(y, all_preds):
    plt.figure(figsize=(12,3))

    class_names = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase',
        'LiftOff', 'Replace', 'BothReleased']
    all_preds_all_classes = []
    for i_class in range(6):
        center_events = compute_center_events_for_class(y,i_class)
        all_class_preds = [all_preds[i-500:i+500, i_class] for i in center_events]
        plot_mean_and_std(all_class_preds, axis=0, color=seaborn.color_palette()[i_class])
        all_preds_all_classes.append(all_class_preds)
        
    plt.axvspan(500-75,500+75, alpha=0.4, color='grey')
    plt.legend(class_names)

def transform_to_time_activations(activations, all_batch_sizes):
    """ Expects format: batch x (activation=> trialrow x channels x 0 x 1 (bc01)).
    Transforms so that in 0 axis now there is time again, not strided time.
    Also removes final 1 axis if it is empty..."""
    transformed_activations = []
    for batch_act, batch_size in zip(activations, all_batch_sizes):
        n_rows = batch_act.shape[0]
        n_chans = batch_act.shape[1]
        assert n_rows % batch_size == 0
        n_stride = n_rows // batch_size
        reshaped_act = batch_act.swapaxes(0,1).reshape(n_chans, n_stride, -1)
        reshaped_act = reshaped_act.swapaxes(1,2).reshape(n_chans, batch_size, -1).swapaxes(0,1)
        transformed_activations.append(reshaped_act)
    return np.array(transformed_activations)
            
def transform_to_cnt_activations(time_activations, n_sample_preds, n_samples):
    """Time series of activations over whole dataset."""
    # concatenate batches
    time_activations = np.concatenate(time_activations)
    valid_mask = np.logical_not(np.isnan(time_activations[0,0]))
    # check that all have same valid mask as it should be
    for window_act in time_activations:
        for filter_act in window_act:
            assert np.array_equal(valid_mask, np.logical_not(np.isnan(filter_act)))


    relevant_acts = time_activations[:,:,valid_mask]

    # earlier layers might contain some predictions that are before 
    # the sample preditions, remove those...
    relevant_acts = relevant_acts[:,:,-n_sample_preds:]
    # now remove the possible overlap between last batch and batch
    # before last batch
    legitimate_last_preds = n_samples % n_sample_preds
    if legitimate_last_preds != 0:
        before_final_overlap = np.concatenate(relevant_acts[:-1], axis=1)
        act_time_series = np.concatenate((before_final_overlap, 
              relevant_acts[-1][:,-legitimate_last_preds:]), axis=1)
    else:
        # if equal to zero actually all last preds are ok!
        act_time_series = np.concatenate(relevant_acts, axis=1)
    assert act_time_series.shape[1] == n_samples
    
    return act_time_series

def transform_to_trial_acts(activations, all_batch_sizes, n_sample_preds,
    n_trial_len):
    time_acts = transform_to_time_activations(activations, all_batch_sizes)
    trial_acts = [transform_to_cnt_activations(t[np.newaxis], n_sample_preds, 
                                               n_samples = n_trial_len)
                  for t in time_acts]
    trial_acts = np.array(trial_acts)
    return trial_acts