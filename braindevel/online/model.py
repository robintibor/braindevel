import lasagne
import theano
from braindevel.veganlasagne.layers import get_input_time_length
import numpy as np
import theano.tensor as T
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import sgd, adam


class OnlineModel(object):
    def __init__(self, model):
        self.model = model

    def initialize(self):
        output = lasagne.layers.get_output(self.model, deterministic=True)
        inputs = lasagne.layers.get_all_layers(self.model)[0].input_var
        pred_fn = theano.function([inputs], output)
        self.pred_fn = pred_fn
        """
        # for now all hardcoded
        targets = T.ivector()
        loss = categorical_crossentropy(output, targets)
        params = lasagne.layers.get_all_params(self.model)
        updates= adam(loss.mean(), params, learning_rate=0.0001)

        # TODO:maxnormconstraint
        self.train_fn = theano.function([inputs, targets], 
            updates=updates)
        """

    def get_n_samples_pred_window(self):
        return get_input_time_length(self.model)

    def predict(self, topo):
        """ Accepts topo as #samples x #chans """
        topo = topo.T[np.newaxis,:,:,np.newaxis]
        params = lasagne.layers.get_all_params(self.model)

        return self.pred_fn(topo)

    def train(self, topo, y):
        topo = topo.T[np.newaxis,:,:,np.newaxis]
        self.train_fn(topo, y)
        return


class OnlineCntModel(object):
    def __init__(self, model):
        self.model = model

    def initialize(self):
        output = lasagne.layers.get_output(self.model, deterministic=True)
        inputs = lasagne.layers.get_all_layers(self.model)[0].input_var
        pred_fn = theano.function([inputs], output)
        self.pred_fn = pred_fn
        """
        # for now all hardcoded
        targets = T.ivector()
        loss = categorical_crossentropy(output, targets)
        params = lasagne.layers.get_all_params(self.model)
        updates= adam(loss.mean(), params, learning_rate=0.0001)

        # TODO:maxnormconstraint
        self.train_fn = theano.function([inputs, targets], 
            updates=updates)
        """

    def get_n_samples_pred_window(self):
        return get_input_time_length(self.model)

    def predict(self, topo):
        """ Accepts topo as #samples x #chans """
        topo = topo.T[np.newaxis,:,:,np.newaxis]
        preds = self.pred_fn(topo)
        return np.mean(preds, axis=0, keepdims=True)

    def train(self, topo, y):
        topo = topo.T[np.newaxis,:,:,np.newaxis]
        self.train_fn(topo, y)
        return