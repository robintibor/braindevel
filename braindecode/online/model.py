import lasagne
import theano
from braindecode.veganlasagne.layers import get_input_time_length
import numpy as np
class OnlineModel(object):
    def __init__(self, model):
        self.model = model
    
    def initialize(self):
        output = lasagne.layers.get_output(self.model, deterministic=True)
        inputs = lasagne.layers.get_all_layers(self.model)[0].input_var
        pred_fn = theano.function([inputs], output)
        self.pred_fn = pred_fn
        
    def get_n_samples_pred_window(self):
        return get_input_time_length(self.model)
    
    def predict(self, topo):
        """ Accepts topo as #samples x #chans """
        topo = topo.T[np.newaxis,:,:,np.newaxis]
        return self.pred_fn(topo)

    