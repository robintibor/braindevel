import lasagn
from numpy.random import RandomState
import theano
import theano.tensor as T

class Experiment(object):
    def setup(self, final_layer, dataset_splitter, loss_var_func,
            updates_var_func, target_var=None):
        lasagne.random.set_rng(RandomState(9859295))
        self.final_layer = final_layer
        self.dataset_splitter = dataset_splitter
        
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
        datasets = self.dataset_splitter.split_into_train_valid_test()
        train_set = datasets['train']
        
        self.monitor_epoch(datasets, 0)
        for epoch in range(70):
            self.train_func(train_set.get_topological_view(), train_set.y)
            self.monitor_epoch(datasets, epoch + 1)
            
    def monitor_epoch(self, all_datasets, epoch):
        print("Epoch {:d}".format(epoch))
        for key in all_datasets:
            dataset = all_datasets[key]
            loss = self.loss_func(dataset.get_topological_view(),
                    dataset.y) 
            print("{:6s} Loss {:.5f}".format(key,
                loss / len(dataset.get_topological_view())))
        