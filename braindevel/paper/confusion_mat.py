import os.path
import logging
import numpy as np
import pickle
from braindevel.veganlasagne.layers import create_pred_fn
from braindevel.experiments.load import load_exp_and_model
log = logging.getLogger(__name__)

def compute_pred_target_labels_for_cnt_exp(exp, model):
    exp.dataset.ensure_is_loaded()
    test_set = exp.dataset_provider.get_train_merged_valid_test(exp.dataset)['test']
    all_batches = list(exp.iterator.get_batches(test_set, shuffle=False))
    log.info("Compile prediction function...")
    pred_fn = create_pred_fn(model)
    log.info("Done.")
    log.info("Computing predictions")
    all_preds = []
    for batch in all_batches:
        all_preds.append(pred_fn(batch[0]))
    log.info("Done")
    # second monitor should be correct one, else you would have to loop to find it
    monitor = exp.monitors[2]
    assert monitor.__class__.__name__ == 'CntTrialMisclassMonitor'
    all_batch_sizes = [len(b[0])  for b in all_batches]
    pred_labels, target_labels = monitor.compute_pred_and_target_labels(test_set, all_preds, all_batch_sizes)
    return pred_labels, target_labels
    
def add_labels_to_cnt_exps_from_dataframe(df):
    folder = df.attrs['folder']
    for exp_id in df.index:
        exp_base_name = os.path.join(folder, str(exp_id))
        add_labels_to_cnt_exp_result(exp_base_name)

def add_labels_to_cnt_exp_result(basename):
    result_file_name = basename + '.result.pkl'
    result = np.load(result_file_name)
    # check if only dummy targets there, in that case add targets and
    # predictions
    if np.array_equal(result.targets, [3,4,1,2,3,4]):
        exp, model = load_exp_and_model(basename)
        pred_labels, target_labels = compute_pred_target_labels_for_cnt_exp(exp, model)
        # add targets
        result.predictions = pred_labels
        result.targets = target_labels
        pickle.dump(result, open(result_file_name, 'w'))
    else:
        log.warn("Targets/predictions already there for {:s}".format(basename))
