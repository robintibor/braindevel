import numpy as np
import time

class MisclassMonitor(object):
    def __init__(self, exponentiate_preds=False, col_suffix='misclass'):
        self.col_suffix = col_suffix
        self.exponentiate_preds = exponentiate_preds

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, losses,
                    all_batch_sizes, targets, dataset):
        all_pred_labels = []
        all_target_labels = []
        for i_batch in range(len(all_batch_sizes)):
            preds = all_preds[i_batch]
            if self.exponentiate_preds:
                preds = np.exp(preds)
            pred_labels = np.argmax(preds, axis=1)
            all_pred_labels.extend(pred_labels)
            all_target_labels.extend(targets[i_batch])
        all_pred_labels = np.array(all_pred_labels)
        all_target_labels = np.array(all_target_labels)

        # in case of one hot encoding convert back to scalar class numbers
        if all_target_labels.ndim == 2:
            all_target_labels = np.argmax(all_target_labels, axis=1)
        misclass = 1 - (np.sum(all_pred_labels == all_target_labels) /
                        float(len(all_target_labels)))
        column_name = "{:s}_{:s}".format(setname, self.col_suffix)
        return {column_name: float(misclass)}


class LossMonitor(object):
    def monitor_epoch(self,):
        return

    def monitor_set(self, setname, all_preds, losses,
                    all_batch_sizes, targets, dataset):
        batch_weights = np.array(all_batch_sizes)/ np.sum(all_batch_sizes)
        loss_per_batch = [np.mean(loss) for loss in losses]
        mean_loss = np.sum(batch_weights * loss_per_batch)
        column_name = "{:s}_loss".format(setname)
        return {column_name: mean_loss}


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

    def monitor_set(self, setname, all_preds, losses,
                    all_batch_sizes, targets, dataset):
        return {}
