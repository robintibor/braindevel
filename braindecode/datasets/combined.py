import numpy as np

class CombinedSet(object):
    reloadable=False
    def __init__(self, sets):
        self.sets = sets

    def ensure_is_loaded(self):
        for dataset in self.sets:
            dataset.ensure_is_loaded()
    def load(self):
        for dataset in self.sets:
            dataset.load()
        # hack to have correct y dimensions
        self.y = self.sets[-1].y[0:1]
