from abc import ABCMeta, abstractmethod
import numpy as np
import time
class Monitor(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def setup(self, monitor_chans, datasets):
        raise NotImplementedError("Subclass needs to implement this")

    @abstractmethod
    def monitor_epoch(self, pred_func, loss_func, datasets):
        raise NotImplementedError("Subclass needs to implement this")

class LossMonitor(Monitor):
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_loss".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans, pred_func, loss_func,
            datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            # compute losses batchwise so that they fit on graphics card
            batch_size = 50
            total_loss = 0.0
            start = 0
            while start < len(dataset.y):
                actual_batch_size = min(start + batch_size, 
                    len(dataset.y)) - start
                total_loss += loss_func(
                    dataset.get_topological_view()[start:start+batch_size],
                    dataset.y[start:start+batch_size])
                # at the end we want the mean over whole dataset
                # so weigh this mean (loss func arleady computes mean for batch)
                # by the size of the batch... this works also if batches
                # not all the same size
                total_loss *= (float(actual_batch_size)/len(dataset.y))
                start += batch_size
            monitor_key = "{:s}_loss".format(setname)
            monitor_chans[monitor_key].append(float(total_loss))
            
class MisclassMonitor(Monitor):
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans,
            pred_func, loss_func, datasets):
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
    
class RuntimeMonitor(Monitor):
    def setup(self, monitor_chans, datasets):
        self.last_call_time = None
        monitor_chans['runtime'] = []

    def monitor_epoch(self, monitor_chans,
            pred_func, loss_func, datasets):
        cur_time = time.time()
        if self.last_call_time is None:
            # just in case of first call
            self.last_call_time = cur_time
        monitor_chans['runtime'].append(cur_time - self.last_call_time)
        self.last_call_time = cur_time
        
    
    