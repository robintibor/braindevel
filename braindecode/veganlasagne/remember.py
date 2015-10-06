"""Classes that remember parameters during training, e.g.,
 remember best model so far"""

class RememberBest():
    def  __init__(self, chan_name):
        self.chan_name = chan_name
        self.best_epoch = 0
        self.lowest_val = float('inf')
        
    def remember_epoch(self, monitor_chans, all_params):
        # -1 due to doing one monitor at start of training
        i_epoch = len(monitor_chans.values()[0]) - 1
        current_val = monitor_chans[self.chan_name][-1]
        if current_val <= self.lowest_val:
            self.best_epoch = i_epoch
            self.lowest_val = current_val
            self.best_params = dict([(p, p.get_value()) for p in all_params])

    def reset_to_best_model(self, monitor_chans, all_params):
        for key in monitor_chans:
            monitor_chans[key] = monitor_chans[key][:self.best_epoch+1]
        for p in all_params:
            p.set_value(self.best_params[p])