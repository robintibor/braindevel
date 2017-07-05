import numpy as np

class NoDecrease(object):
    """ Stops if there is no decrease on a given monitor channel
    for given number of epochs."""
    def  __init__(self, chan_name, num_epochs, min_decrease=1e-6):
        self.chan_name = chan_name
        self.num_epochs = num_epochs
        self.min_decrease = min_decrease
        self.best_epoch = 0
        self.lowest_val = float('inf')
        
    def should_stop(self, monitor_chans):
        # -1 due to doing one monitor at start of training
        i_epoch = len(monitor_chans.values()[0]) - 1
        current_val = monitor_chans[self.chan_name][-1]
        if current_val < ((1 - self.min_decrease) * self.lowest_val):
            self.best_epoch = i_epoch
            self.lowest_val = current_val
        
        return (i_epoch - self.best_epoch) >= self.num_epochs
        
class MaxEpochs(object):
    def  __init__(self, num_epochs):
        self.num_epochs = num_epochs
        
    def should_stop(self, monitor_chans):
        # -1 due to doing one monitor at start of training
        i_epoch = len(monitor_chans.values()[0]) - 1
        return i_epoch >= self.num_epochs


class Or(object):
    def  __init__(self, stop_criteria):
        self.stop_criteria = stop_criteria
        
    def should_stop(self, monitor_chans):
        return np.any([s.should_stop(monitor_chans) 
            for s in self.stop_criteria])
        
class ChanBelow():
    """ Stops if the given monitor channel is below the given value."""
    def  __init__(self, chan_name, target_value):
        self.chan_name = chan_name
        self.target_value = target_value
        
    def should_stop(self, monitor_chans):
        # -1 due to doing one monitor at start of training
        current_val = monitor_chans[self.chan_name][-1]
        return current_val < self.target_value