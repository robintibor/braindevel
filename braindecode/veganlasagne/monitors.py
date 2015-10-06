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
            monitor_key = "{:s}_y_loss".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans, pred_func, loss_func,
            datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            loss = loss_func(dataset.get_topological_view(),
                    dataset.y) 
            monitor_key = "{:s}_y_loss".format(setname)
            monitor_chans[monitor_key].append(float(loss))
            
class MisclassMonitor(Monitor):
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_y_misclass".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans,
            pred_func, loss_func, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            monitor_key = "{:s}_y_misclass".format(setname)
            preds = pred_func(dataset.get_topological_view())
            pred_classes = np.argmax(preds, axis=1)
            misclass = 1 - (np.sum(pred_classes == dataset.y) / 
                float(len(dataset.y)))
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
        
    
    