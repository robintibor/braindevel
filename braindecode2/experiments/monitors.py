import numpy as np
import time

from braindecode2.trial_segment import compute_trial_start_end_samples


class MisclassMonitor(object):
    def __init__(self, exponentiate_preds=False, col_suffix='misclass'):
        self.col_suffix = col_suffix
        self.exponentiate_preds = exponentiate_preds

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        all_pred_labels = []
        all_target_labels = []
        for i_batch in range(len(all_batch_sizes)):
            preds = all_preds[i_batch]
            if True:
                preds = np.exp(preds)
            pred_labels = np.argmax(preds, axis=1)
            all_pred_labels.extend(pred_labels)
            targets = all_targets[i_batch]
            # targets may be one-hot-encoded or not
            if targets.ndim >= pred_labels.ndim:
                targets = np.argmax(targets, axis=1)
            assert targets.shape == pred_labels.shape
            all_target_labels.extend(targets)
        all_pred_labels = np.array(all_pred_labels)
        all_target_labels = np.array(all_target_labels)
        assert all_pred_labels.shape == all_target_labels.shape

        misclass = 1 - np.mean(all_target_labels == all_pred_labels)
        column_name = "{:s}_{:s}".format(setname, self.col_suffix)
        return {column_name: float(misclass)}


class LossMonitor(object):
    def monitor_epoch(self,):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        batch_weights = np.array(all_batch_sizes)/ np.sum(all_batch_sizes)
        loss_per_batch = [np.mean(loss) for loss in all_losses]
        mean_loss = np.sum(batch_weights * loss_per_batch)
        column_name = "{:s}_loss".format(setname)
        return {column_name: mean_loss}



class CntTrialMisclassMonitor(object):
    def __init__(self, input_time_length=None):
        self.input_time_length = input_time_length

    def monitor_epoch(self,):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        assert self.input_time_length is not None, "Need to know input time length..."
        all_pred_labels, all_target_labels = self.compute_pred_and_target_labels(
            dataset, all_preds)
        misclass = 1 - np.mean(all_pred_labels == all_target_labels)
        column_name = "{:s}_misclass".format(setname)
        return {column_name: float(misclass)}

    def compute_pred_and_target_labels(self, dataset, all_preds,):
        all_target_labels = []
        preds_per_trial = compute_preds_per_trial(dataset.y,
                                                  all_preds,
                                                  self.input_time_length)
        all_pred_labels = [np.argmax(np.mean(p, axis=1))
                           for p in preds_per_trial]
        i_trial_starts, i_trial_ends = compute_trial_start_end_samples(
            dataset.y, check_trial_lengths_equal=False,
            input_time_length=self.input_time_length)
        for i_trial, (start, end) in enumerate(
                zip(i_trial_starts, i_trial_ends)):
            targets = dataset.y[start:end + 1]  # end is not inclusive
            assert len(targets) == preds_per_trial[i_trial].shape[1]
            # max would have several 1s for different classes
            # if there are any two different classes with 1s
            # in all samples
            assert np.sum(np.max(targets, axis=0)) == 1, ("Trial should only "
                                                          "have one class")
            assert np.sum(targets) == len(targets), ("Every sample should have "
                                                     "one positive marker")
            target_label = np.argmax(np.max(targets, axis=0))
            all_target_labels.append(target_label)

        all_pred_labels = np.array(all_pred_labels)
        all_target_labels = np.array(all_target_labels)
        assert all_pred_labels.shape == all_target_labels.shape
        return all_pred_labels, all_target_labels


class CroppedTrialMisclassMonitor(object):
    def __init__(self, input_time_length=None):
        self.input_time_length = input_time_length

    def monitor_epoch(self,):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        assert self.input_time_length is not None, "Need to know input time length..."
        all_pred_labels = self.compute_pred_labels(dataset, all_preds)
        assert all_pred_labels.shape == dataset.y.shape
        misclass = 1 - np.mean(all_pred_labels == dataset.y)
        column_name = "{:s}_misclass".format(setname)
        return {column_name: float(misclass)}

    def compute_pred_labels(self, dataset, all_preds,):
        n_preds_per_input = all_preds[0].shape[2]
        n_receptive_field = self.input_time_length - n_preds_per_input + 1

        i_trial_starts = [0] * len(dataset.y)
        i_trial_ends = [trial.shape[1] - n_receptive_field
                        for trial in dataset.X]
        preds_per_trial = compute_preds_per_trial_from_start_end(
            all_preds, i_trial_starts, i_trial_ends)
        all_pred_labels = [np.argmax(np.mean(p, axis=1))
                           for p in preds_per_trial]

        all_pred_labels = np.array(all_pred_labels)
        assert all_pred_labels.shape == dataset.y.shape
        return all_pred_labels


def compute_preds_per_trial(y, all_preds, input_time_length):
    """
    Parameters
    ----------
    y
    all_preds
    all_batch_sizes
    input_time_length

    Returns
    -------
    preds_per_trial: list of 2darray
        Trials x classes x time
    """
    i_trial_starts, i_trial_ends = compute_trial_start_end_samples(
        y, check_trial_lengths_equal=False,
        input_time_length=input_time_length)
    return compute_preds_per_trial_from_start_end(
        all_preds, i_trial_starts, i_trial_ends)


def compute_preds_per_trial_from_start_end(
        all_preds, i_trial_starts, i_trial_ends):
    # TODO: change to just accept trial lengths. or call n_preds_per_trial
    # and dont foget to remove  +1 from needed_samples line if you really use len
    i_pred_block = 0
    all_preds_arr = np.concatenate(all_preds, axis=0)
    #all_preds_arr has shape forward_passes x classes x time
    preds_per_trial = []
    for i_trial in range(len(i_trial_starts)):
        # + 1 since end is inclusive
        # so if trial end is 1 and trial start is 0
        # need two samples (0 and 1)
        needed_samples = (i_trial_ends[i_trial] - i_trial_starts[i_trial]) + 1
        preds_this_trial = []
        while needed_samples > 0:
            # - needed_samples: only has an effect
            # in case there are more samples thatn we actually still need
            # in the block
            # That can happen since final block of a trial can overlap
            # with block before so we can have some redundant preds
            pred_samples = all_preds_arr[i_pred_block,:,
                           -needed_samples:]
            preds_this_trial.append(pred_samples)
            needed_samples -= pred_samples.shape[1]
            i_pred_block += 1

        preds_this_trial = np.concatenate(preds_this_trial, axis=1)
        preds_per_trial.append(preds_this_trial)
    assert i_pred_block == len(all_preds_arr), (
        "Expect that all prediction forward passes are needed, "
        "used {:d}, existing {:d}".format(
        i_pred_block, len(all_preds_arr)))
    return preds_per_trial


class RuntimeMonitor(object):
    def __init__(self):
        self.last_call_time = None

    def monitor_epoch(self,):
        cur_time = time.time()
        if self.last_call_time is None:
            # just in case of first call
            self.last_call_time = cur_time
        epoch_runtime = cur_time - self.last_call_time
        self.last_call_time = cur_time
        return {'runtime': epoch_runtime}

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        return {}