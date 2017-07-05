from braindecode.mywyrm.processing import segment_dat_fast

class SignalProcessor(object):
    """ Class to process loaded wyrm set, segment to trials etc."""
    def __init__(self, set_loader, sensor_names=None,
            cnt_preprocessors=[], epo_preprocessors=[],
            segment_ival=(0,4000), 
            marker_def={'1': [1], '2': [2], '3': [3], '4': [4]}):
        """ Constructor will not call superclass constructor yet"""
        self.__dict__.update(locals())
        del self.self

    def load(self):
        # TODELAY: Later switch to a wrapper dataset for all files
        self.load_signal_and_markers()
        self.preprocess_continuous_signal()
        self.segment_into_trials()
        self.remove_continuous_signal() # not needed anymore
        self.preprocess_trials()

    def load_signal_and_markers(self):
        self.cnt = self.set_loader.load()

    def preprocess_continuous_signal(self):
        for func, kwargs in self.cnt_preprocessors:
            self.cnt = func(self.cnt, **kwargs)

    def segment_into_trials(self):
        assert self.segment_ival is not None
        # adding the numbers at start to force later sort in segment_dat
        # to sort them in given order
        self.epo = segment_dat_fast(self.cnt, 
            ival=self.segment_ival,
            marker_def=self.marker_def)

    def remove_continuous_signal(self):
        del self.cnt

    def preprocess_trials(self):
        for func, kwargs in self.epo_preprocessors:
            self.epo = func(self.epo, **kwargs)
