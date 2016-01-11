import numpy as np

class CombinedSet(object):
    reloadable=False
    def __init__(self, sets):
        self.sets = sets

    def ensure_is_loaded(self):
        for dataset in self.sets:
            dataset.ensure_is_loaded()
    def load(self):
        # TODO: change this please...
        self.y = np.zeros((1))
        for dataset in self.sets:
            dataset.load()
            