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
