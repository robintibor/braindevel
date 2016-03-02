import numpy as np
from pylearn2.utils import serial
from copy import deepcopy

class BinaryResult(object):
    """ dummy class for binary results """
    pass

class CSPResult(object):
    """ For storing a result"""
    def __init__(self, csp_trainer, parameters, training_time):
        self.multi_class = csp_trainer.multi_class
        self.templates = {}
        self.training_time = training_time
        self.parameters = deepcopy(parameters)
        # Copy cleaning results
        self.rejected_chan_names = csp_trainer.rejected_chan_names
        self.rejected_trials = csp_trainer.rejected_trials
        self.clean_trials = csp_trainer.clean_trials
        # Copy some binary results
        self.binary = BinaryResult()
        self.binary.train_accuracy = csp_trainer.binary_csp.train_accuracy
        self.binary.test_accuracy = csp_trainer.binary_csp.test_accuracy
        self.binary.filterbands = csp_trainer.binary_csp.filterbands
    
    def get_misclasses(self):
        return {
        'train': 
                np.array([ 1 -acc for acc in self.multi_class.train_accuracy]),
        'test': 
            np.array([ 1 -acc for acc in self.multi_class.test_accuracy])
        }

    def save(self, filename):
        serial.save(filename, self)
 
TrainCSPResult = CSPResult # backwards compatibility, printing earlier results   

class CSPModel(object):
    # TODO: check if it even works/is needed?
    """ For storing a model. Warning can be quite big"""
    def __init__(self, csp_trainer):
        self.csp_trainer = csp_trainer
    
    def save(self, filename):
        if (hasattr(self.csp_trainer.binary_csp, 'cnt')):
            del self.csp_trainer.binary_csp.cnt
        del self.csp_trainer.bbci_set
        del self.csp_trainer.cnt
        serial.save(filename, self.csp_trainer)
