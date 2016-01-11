import numpy as np

def compute_matthews_correlation_coeff(preds, y, threshold):
    n_samples = len(y)
    assert len(y) == len(preds)
    predicted_label = preds > threshold
    correct_samples = (predicted_label == y)
    true_positive = np.logical_and(y == 1, correct_samples)
    true_negative = np.logical_and(y == 0, correct_samples)
    
    false_positive = np.logical_and(predicted_label, np.logical_not(correct_samples))
    false_negative = np.logical_and(np.logical_not(predicted_label),
                                   np.logical_not(correct_samples))
    tp, tn, fp, fn = (np.sum(true_positive), np.sum(true_negative),
        np.sum(false_positive), np.sum(false_negative))
    matthews_correlation_coeff = ((tp * tn + fp * fn) / 
                                  np.sqrt((tp + fp) * (tp + fn) * 
                                         (tn + fp) * (tn + fn)))
    if np.isnan(matthews_correlation_coeff):
        matthews_correlation_coeff = 0
    return matthews_correlation_coeff
    
    
    
def find_best_threshold(preds,y):
    """Uses Matthews Correlation Coefficient to find best threshold.
    Returns best threshold and corresponding MCC value."""
    
    threshold_and_mccs = []
    for threshold in np.linspace(0,1,100):
        matthews_correlation_coeff = compute_matthews_correlation_coeff(preds, 
            y, threshold)
        threshold_and_mccs.append((threshold, matthews_correlation_coeff))
    threshold_and_mccs = np.array(threshold_and_mccs)
    i_best_threshold = np.argmax(threshold_and_mccs[:,1])
    return threshold_and_mccs[i_best_threshold]
