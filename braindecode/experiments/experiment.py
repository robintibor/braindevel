import lasagne 
from numpy.random import RandomState  
import theano  
import theano.tensor as T  

class Experiment(object):
    def setup(self, final_layer, X, y, loss_var_func, updates_var_func, target_var=None):
        lasagne.random.set_rng(RandomState(9859295))
        self.final_layer = final_layer
        self.X = X
        self.y = y
        self.loss_var_func = loss_var_func
        self.updates_var_func = updates_var_func
        if target_var is None:
            target_var = T.ivector('targets')
        prediction = lasagne.layers.get_output(final_layer)
        loss = loss_var_func(prediction, target_var).mean()

        # create parameter update expressions
        params = lasagne.layers.get_all_params(final_layer, trainable=True)
        updates = updates_var_func(loss, params)
        input_var = lasagne.layers.get_all_layers(final_layer)[0].input_var
        self.loss_func = theano.function([input_var, target_var], loss)
        self.train_func = theano.function([input_var, target_var], updates=updates)

    def run(self):
        for epoch in range(70):
            self.train_func(self.X, self.y)
            if epoch % 5 == 0:
                epoch_loss = self.loss_func(self.X, self.y)
                print("Epoch {:d} Loss {:g}".format(epoch, epoch_loss / len(self.X)))