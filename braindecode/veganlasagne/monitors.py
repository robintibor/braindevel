from abc import ABCMeta, abstractmethod
import numpy as np
import time
from braindecode.datahandling.batch_iteration import WindowsIterator

class Monitor(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def setup(self, monitor_chans, datasets):
        raise NotImplementedError("Subclass needs to implement this")

    @abstractmethod
    def monitor_epoch(self, pred_func, loss_func, datasets, iterator):
        raise NotImplementedError("Subclass needs to implement this")

class LossMonitor(Monitor):
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_loss".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans, pred_func, loss_func, datasets,
            iterator):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            # compute losses batchwise so that they fit on graphics card
            #batch_size = 50
            total_loss = 0.0
            num_trials = 0
            for batch in iterator.get_batches(dataset, shuffle=False):
                batch_size = batch[0].shape[0]
                batch_loss = loss_func(batch[0], batch[1])
                # at the end we want the mean over whole dataset
                # so weigh this mean (loss func arleady computes mean for batch)
                # by the size of the batch... this works also if batches
                # not all the same size
                num_trials += batch_size
                total_loss += (batch_loss * batch_size)
            
            mean_loss = total_loss / num_trials
            monitor_key = "{:s}_loss".format(setname)
            monitor_chans[monitor_key].append(float(mean_loss))
            
class MisclassMonitor(Monitor):
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans,
            pred_func, loss_func, datasets, iterator):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            all_target_labels = []
            all_pred_labels = []
            for batch in iterator.get_batches(dataset, shuffle=False):
                preds = pred_func(batch[0])
                pred_labels = np.argmax(preds, axis=1)
                all_pred_labels.extend(pred_labels)
                all_target_labels.extend(batch[1])
            all_pred_labels = np.array(all_pred_labels)
            all_target_labels = np.array(all_target_labels)
            misclass = 1 - (np.sum(all_pred_labels == all_target_labels) / 
                float(len(all_target_labels)))
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key].append(float(misclass))
    
class WindowMisclassMonitor(Monitor):
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans,
            pred_func, loss_func, datasets, iterator):
        assert(isinstance(iterator, WindowsIterator))
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            all_preds = []
            for batch in iterator.get_batches(dataset, shuffle=False):
                batch_preds = pred_func(batch[0])
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

    def monitor_epoch(self, monitor_chans,
            pred_func, loss_func, datasets, iterator):
        cur_time = time.time()
        if self.last_call_time is None:
            # just in case of first call
            self.last_call_time = cur_time
        monitor_chans['runtime'].append(cur_time - self.last_call_time)
        self.last_call_time = cur_time

class DummyMisclassMonitor(Monitor):
    """ For Profiling tests...."""
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans,
            pred_func, loss_func, datasets, iterator):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            misclass = 0.5
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key].append(float(misclass))