from abc import ABCMeta, abstractmethod
import time
from copy import deepcopy
import numpy as np
import theano
from sklearn.metrics import roc_auc_score
from braindecode.datahandling.batch_iteration import compute_trial_start_end_samples
from sklearn.metrics import cohen_kappa_score

class Monitor(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def setup(self, monitor_chans, datasets):
        raise NotImplementedError("Subclass needs to implement this")

    @abstractmethod
    def monitor_epoch(self, monitor_chans):
        raise NotImplementedError("Subclass needs to implement this")
    @abstractmethod
    def monitor_set(self, monitor_chans, setname, preds, losses, 
            all_batch_sizes, targets, dataset):
        raise NotImplementedError("Subclass needs to implement this")

class MonitorManager(object):
    def __init__(self, monitors):
        self.monitors = monitors
        
    def create_theano_functions(self, input_var, target_var, predictions_var, 
            loss_var):
        self.pred_loss_func = theano.function([input_var, target_var], 
            [predictions_var, loss_var])
        
    def setup(self, monitor_chans, datasets):
        for monitor in self.monitors:
            monitor.setup(monitor_chans, datasets)
        
    def monitor_epoch(self, monitor_chans, datasets, iterator):
        # first call monitor epoch of all monitors, 
        # then monitor set with losses and preds
        # maybe change this method entirely if it turns out to be too
        # inflexible
        for m in self.monitors:
            m.monitor_epoch(monitor_chans)
        
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            all_preds = []
            all_losses = []
            batch_sizes = []
            targets = []
            for batch in iterator.get_batches(dataset, shuffle=False):
                preds, loss = self.pred_loss_func(batch[0], batch[1])
                all_preds.append(preds)
                all_losses.append(loss)
                batch_sizes.append(len(batch[0]))
                targets.append(batch[1])

            for m in self.monitors:
                m.monitor_set(monitor_chans, setname, all_preds, all_losses,
                    batch_sizes, targets, dataset)

class LossMonitor(Monitor):
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_loss".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, preds, losses, 
            all_batch_sizes, targets, dataset):
        total_loss = 0.0
        num_trials = 0
        for i_batch in range(len(all_batch_sizes)):
            batch_size = all_batch_sizes[i_batch]
            batch_loss = losses[i_batch]
            # at the end we want the mean over whole dataset
            # so weigh this mean (loss func arleady computes mean for batch)
            # by the size of the batch... this works also if batches
            # not all the same size
            num_trials += batch_size
            total_loss += (batch_loss * batch_size)
            
        mean_loss = total_loss / float(num_trials)
        monitor_key = "{:s}_loss".format(setname)
        monitor_chans[monitor_key].append(mean_loss)
        
class MisclassMonitor(Monitor):
    def __init__(self, chan_name='misclass'):
        self.chan_name = chan_name

    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_preds, losses, 
            all_batch_sizes, targets, dataset):
        all_pred_labels = []
        all_target_labels = []
        for i_batch in range(len(all_batch_sizes)):
            preds = all_preds[i_batch]
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
        monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
        monitor_chans[monitor_key].append(float(misclass))

class WindowMisclassMonitor(Monitor):
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_batch_preds, losses, 
            all_batch_sizes, targets, dataset):
        all_preds = []
        for i_batch in range(len(all_batch_sizes)):
            batch_preds = all_batch_preds[i_batch]
            all_preds.extend(batch_preds)
        
        n_trials = len(dataset.y)
        preds_by_trial = np.reshape(all_preds, (n_trials, -1 , len(all_preds[0])))
        preds_by_trial = np.sum(preds_by_trial, axis=1)
        pred_labels = np.argmax(preds_by_trial, axis=1)
        accuracy = np.sum(pred_labels == dataset.y) / float(len(dataset.y))
        misclass = 1 - accuracy
        monitor_key = "{:s}_misclass".format(setname)
        monitor_chans[monitor_key].append(float(misclass))

class RuntimeMonitor(Monitor):
    def setup(self, monitor_chans, datasets):
        self.last_call_time = None
        monitor_chans['runtime'] = []

    def monitor_epoch(self, monitor_chans):
        cur_time = time.time()
        if self.last_call_time is None:
            # just in case of first call
            self.last_call_time = cur_time
        monitor_chans['runtime'].append(cur_time - self.last_call_time)
        self.last_call_time = cur_time

    def monitor_set(self, monitor_chans, setname, all_preds, losses, 
            all_batch_sizes, targets, dataset):
        return

class DummyMisclassMonitor(Monitor):
    """ For Profiling tests...."""
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_set(self, monitor_chans, setname, all_preds, losses, 
            all_batch_sizes, targets, dataset):
        misclass = 0.5
        monitor_key = "{:s}_misclass".format(setname)
        monitor_chans[monitor_key].append(float(misclass))

    def monitor_epoch(self, monitor_chans):
        return

def safe_roc_auc_score(y_true, y_score):
    """Returns nan if targets only contain one class"""
    if len(np.unique(y_true)) == 1:
        return np.nan
    else:
        return roc_auc_score(y_true, y_score)
    
def auc_classes_mean(y, preds):
    # nanmean to ignore classes that are not present
    return np.nanmean([safe_roc_auc_score(
            np.int32(y[:,i] == 1), preds[:,i]) 
            for i in range(y.shape[1])])

class AUCMeanMisclassMonitor(Monitor):
    def __init__(self, input_time_length=None, n_sample_preds=None):
        self.input_time_length = input_time_length
        self.n_sample_preds = n_sample_preds
    
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_preds, losses, 
            all_batch_sizes, targets, dataset):
        # remove last preds that were duplicates due to overlap of final windows
        n_samples = len(dataset.y)
        if self.input_time_length is not None:
            all_preds_arr = get_reshaped_cnt_preds(all_preds, n_samples, 
                self.input_time_length, self.n_sample_preds)
        else:
            all_preds_arr = np.concatenate(all_preds)
        
        auc_mean = auc_classes_mean(dataset.y, all_preds_arr)
        misclass = 1 - auc_mean
        monitor_key = "{:s}_misclass".format(setname)
        monitor_chans[monitor_key].append(float(misclass))
        
def get_reshaped_cnt_preds(all_preds, n_samples, input_time_length,
        n_sample_preds):
    """Taking predictions from a multiple prediction/parallel net
    and removing the last predictions (from last batch) which are duplicated.
    """
    all_preds = deepcopy(all_preds)
    # fix the last predictions, they are partly duplications since the last window
    # is made to fit into the timesignal 
    # sample preds
    # might not exactly fit into number of samples)
    legitimate_last_preds = n_samples % n_sample_preds
    if legitimate_last_preds != 0: # in case = 0 there was no overlap, no need to do anything!
        fixed_last_preds = all_preds[-1][-legitimate_last_preds:]
        final_batch_size = all_preds[-1].shape[0] / n_sample_preds
        if final_batch_size > 1:
            # need to take valid sample preds from batches before
            samples_from_legit_batches = n_sample_preds * (final_batch_size - 1)
            fixed_last_preds = np.append(all_preds[-1][:samples_from_legit_batches], 
                 fixed_last_preds, axis=0)
        all_preds[-1] = fixed_last_preds
    
    all_preds_arr = np.concatenate(all_preds)
    
    return all_preds_arr

class CntTrialMisclassMonitor(Monitor):
    def __init__(self, input_time_length=None):
        self.input_time_length = input_time_length

    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_preds, losses, 
            all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        assert self.input_time_length is not None, "Need to know input time length..."
        all_pred_labels, all_target_labels = self.compute_pred_and_target_labels(
            dataset, all_preds, all_batch_sizes) 
        
        misclass = 1 - (np.sum(all_pred_labels == all_target_labels) / 
            float(len(all_target_labels)))
        monitor_key = "{:s}_misclass".format(setname)
        monitor_chans[monitor_key].append(float(misclass))
        return
    
    def compute_pred_and_target_labels(self, dataset, all_preds, all_batch_sizes):
        all_target_labels = []
        preds_per_trial = compute_preds_per_trial(dataset.y, 
            all_preds, all_batch_sizes, self.input_time_length)
        all_pred_labels = [np.argmax(np.mean(p, axis=0)) 
            for p in preds_per_trial]
        i_trial_starts, i_trial_ends = compute_trial_start_end_samples(
            dataset.y, check_trial_lengths_equal=False,
            input_time_length=self.input_time_length)
        for i_trial, (start, end) in enumerate(zip(i_trial_starts, i_trial_ends)):
            targets = dataset.y[start:end+1] # end is not inclusive
            assert len(targets) == len(preds_per_trial[i_trial])
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
        return all_pred_labels, all_target_labels


def compute_preds_per_trial(y, all_preds, all_batch_sizes, input_time_length):
    i_trial_starts, i_trial_ends = compute_trial_start_end_samples(
        y, check_trial_lengths_equal=False,
        input_time_length=input_time_length)
    return compute_preds_per_trial_from_start_end(
        all_preds, all_batch_sizes, i_trial_starts, i_trial_ends)


def compute_preds_per_trial_from_start_end(
        all_preds, all_batch_sizes, i_trial_starts, i_trial_ends):
    i_pred_block = 0
    n_sample_preds = all_preds[0].shape[0] / all_batch_sizes[0]
    all_preds_arr = np.concatenate(all_preds, axis=0)
    preds_per_forward_pass = np.reshape(all_preds_arr, (-1, n_sample_preds,
                                                        all_preds_arr.shape[1]))
    preds_per_trial = []
    for i_trial in xrange(len(i_trial_starts)):
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
            pred_samples = preds_per_forward_pass[i_pred_block,
                           -needed_samples:]
            preds_this_trial.append(pred_samples)
            needed_samples -= len(pred_samples)
            i_pred_block += 1

        preds_this_trial = np.concatenate(preds_this_trial, axis=0)
        preds_per_trial.append(preds_this_trial)
    assert i_pred_block == len(preds_per_forward_pass), ("Expect that all "
                                                         "prediction forward passes are needed, used {:d}, existing {:d}".format(
        i_pred_block, len(preds_per_forward_pass)))
    return preds_per_trial


class KappaMonitor(Monitor):
    def __init__(self, input_time_length, chan_name='kappa', mode='mean'):
        self.chan_name = chan_name
        self.input_time_length = input_time_length
        self.mode = mode

    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_preds, losses, 
            all_batch_sizes, targets, dataset):
        
        preds_per_trial = compute_preds_per_trial(dataset.y, 
            all_preds, all_batch_sizes, self.input_time_length)

        targets_per_trial = compute_preds_per_trial(dataset.y, 
            targets, all_batch_sizes, self.input_time_length)
    
        preds_per_trial = np.array(preds_per_trial)
        targets_per_trial = np.array(targets_per_trial)
        #assert np.allclose(np.sum(preds_per_trial, axis=2), 1,rtol=0.01,
        #    atol=0.01)
        assert np.all(np.sum(targets_per_trial, axis=2) == 1) 
        pred_labels_per_timepoint_per_trial = np.argmax(preds_per_trial, axis=2)
        labels_per_timepoint_per_trial = np.argmax(targets_per_trial, axis=2)
        kappa_timecourse = np.array([cohen_kappa_score(p,t) for p,t in zip(pred_labels_per_timepoint_per_trial.T,
                                      labels_per_timepoint_per_trial.T)])
        if self.mode == 'mean':
            kappa = np.mean(kappa_timecourse)
        else:
            assert self.mode == 'max'
            kappa = np.max(kappa_timecourse)
        
        monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
        monitor_chans[monitor_key].append(-float(kappa))


class MeanSquaredErrorMonitor(Monitor):
    def __init__(self,chan_name='mse',
            out_factor=1, demean_preds=False):
        self.chan_name = chan_name
        self.out_factor = out_factor
        self.demean_preds = demean_preds

    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_preds, losses, 
            all_batch_sizes, targets, dataset):
        
        all_preds_arr = np.concatenate(all_preds)
        if self.demean_preds:
            if setname == 'train':
                self._pred_mean = np.mean(all_preds_arr)
            all_preds_arr = all_preds_arr - self._pred_mean
        all_preds_arr = all_preds_arr * self.out_factor
        all_preds_arr = np.clip(all_preds_arr, -self.out_factor, self.out_factor)
        all_targets_arr = np.concatenate(targets)
        
        mse = np.mean(np.square(all_preds_arr - all_targets_arr))
        monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
        monitor_chans[monitor_key].append(mse)

class CorrelationMonitor(Monitor):
    def __init__(self,chan_name='corr'):
        self.chan_name = chan_name

    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_preds, losses, 
            all_batch_sizes, targets, dataset):
        all_preds_arr = np.concatenate(all_preds).squeeze()
        all_targets_arr = np.concatenate(targets).squeeze()
        corr = np.corrcoef(all_preds_arr, all_targets_arr)[0,1]
        monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
        monitor_chans[monitor_key].append(-corr)


class MeanSquaredErrorClassMonitor(Monitor):
    def __init__(self, out_factor=1, demean_preds=False, chan_name='mse'):
        self.out_factor = out_factor
        self.chan_name = chan_name
        self.demean_preds = demean_preds

    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_preds, losses, 
            all_batch_sizes, targets, dataset):
        all_preds_arr = np.concatenate(all_preds)
        if self.demean_preds:
            if setname == 'train':
                self._pred_mean = np.mean(all_preds_arr)
            all_preds_arr = all_preds_arr - self._pred_mean

        all_targets_arr = np.concatenate(targets)
        
        target_label = np.zeros(len(all_targets_arr))
        target_label[all_targets_arr[:,0] == 1] = -1
        target_label[all_targets_arr[:,1] == 1] = 1
        
        single_pred = (all_preds_arr[:,1] - all_preds_arr[:,0]) * (
            1 - all_preds_arr[:,2]) * self.out_factor
        
        mse = np.mean(np.square(target_label - single_pred))
        
        monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
        monitor_chans[monitor_key].append(mse)

class CorrelationClassMonitor(Monitor):
    def __init__(self,chan_name='corr'):
        self.chan_name = chan_name

    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_preds, losses, 
            all_batch_sizes, targets, dataset):
        all_preds_arr = np.concatenate(all_preds)
        all_targets_arr = np.concatenate(targets)
        target_label = np.zeros(len(all_targets_arr))
        target_label[all_targets_arr[:,0] == 1] = -1
        target_label[all_targets_arr[:,1] == 1] = 1
        single_pred = (all_preds_arr[:,1] - all_preds_arr[:,0]) * (
            1 - all_preds_arr[:,2])
        corr = np.corrcoef(target_label, single_pred)[0,1]
        monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
        monitor_chans[monitor_key].append(-corr)

""" OLD DELETE:
class MeanSquaredErrorMonitor(Monitor):
    def __init__(self,chan_name='mse'):
        self.chan_name = chan_name

    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_preds, losses, 
            all_batch_sizes, targets, dataset):
        
        all_preds_arr = np.concatenate(all_preds)
        n_missing_preds = dataset.y.shape[0] - all_preds_arr.shape[0]
        padded_preds = np.concatenate((np.zeros((n_missing_preds, all_preds_arr.shape[1])), all_preds_arr), axis=0)
        assert dataset.y.shape == padded_preds.shape
        
        target_label = np.zeros(len(dataset.y))
        target_label[dataset.y[:,0] == 1] = -1
        target_label[dataset.y[:,1] == 1] = 1
        
        single_pred = padded_preds[:,1] - padded_preds[:,0]
        
        mse = np.mean(np.square(target_label - single_pred))
        
        monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
        monitor_chans[monitor_key].append(mse)

class CorrelationMonitor(Monitor):
    def __init__(self,chan_name='corr'):
        self.chan_name = chan_name

    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_preds, losses, 
            all_batch_sizes, targets, dataset):
        
        all_preds_arr = np.concatenate(all_preds)
        n_missing_preds = dataset.y.shape[0] - all_preds_arr.shape[0]
        padded_preds = np.concatenate((np.zeros((n_missing_preds, all_preds_arr.shape[1])), all_preds_arr), axis=0)
        assert dataset.y.shape == padded_preds.shape
        
        target_label = np.zeros(len(dataset.y))
        target_label[dataset.y[:,0] == 1] = -1
        target_label[dataset.y[:,1] == 1] = 1
        
        single_pred = padded_preds[:,1] - padded_preds[:,0]
        
        corr = np.corrcoef(target_label, single_pred)[0,1]
        
        monitor_key = "{:s}_{:s}".format(setname, self.chan_name)
        monitor_chans[monitor_key].append(-corr)
"""