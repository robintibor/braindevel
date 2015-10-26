from abc import ABCMeta, abstractmethod
import numpy as np
import time
from braindecode.datasets.batch_iteration import SampleWindowsIterator
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

    def monitor_epoch(self, monitor_chans, pred_func, loss_func,
            datasets, iterator):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            # compute losses batchwise so that they fit on graphics card
            #batch_size = 50
            total_loss = 0.0
            num_trials = 0
            for batch in iterator.get_batches(dataset,deterministic=True):
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
            batch_size = 50
            # compute batchwise so that they fit on graphics card
            # this has been performance tested on a case where
            # whole set had size 400 and 
            # still fits on gpu and it was only about
            # factor 2 slower...
            preds = ([pred_func(dataset.get_topological_view()[i:i+batch_size]) 
                      for i in xrange(0, len(dataset.y), batch_size)])
            preds = np.concatenate(preds)
            pred_classes = np.argmax(preds, axis=1)
            misclass = 1 - (np.sum(pred_classes == dataset.y) / 
                float(len(dataset.y)))
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key].append(float(misclass))
    
class SampleWindowMisclassMonitor(Monitor):
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans,
            pred_func, loss_func, datasets, iterator):
        #TODO:reenable assert(isinstance(iterator,SampleWindowsIterator))
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            all_pred_labels = []
            all_target_labels = []
            for batch in iterator.get_batches(dataset, deterministic=True,
                    merge_trial_window_dims=False):
                batch_features = batch[0]
                batch_y = batch[1]
                flat_batch_features = np.concatenate(batch_features)
                preds = pred_func(flat_batch_features)
                batch_size = batch_features.shape[0]
                
                # transform to #trials x #windows again
                preds = np.reshape(preds,(batch_size,-1,preds.shape[1]))
                # pred of trial is mean over windows (windows in dim 1)
                preds = np.mean(preds, axis=1)
                pred_labels = np.argmax(preds,axis=1)
                all_pred_labels.extend(pred_labels)
                all_target_labels.extend(batch_y)
            all_pred_labels = np.array(all_pred_labels)
            all_target_labels = np.array(all_target_labels)
            assert len(all_target_labels) == len(dataset.y)
            misclass = 1 - (np.sum(all_pred_labels == all_target_labels) / 
                float(len(all_pred_labels)))
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
