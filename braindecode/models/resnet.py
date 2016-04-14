from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer, NonlinearityLayer
from braindecode.veganlasagne.residual_net import residual_block
from lasagne.nonlinearities import softmax, identity
from braindecode.veganlasagne.batch_norm import batch_norm
import lasagne
from braindecode.veganlasagne.layers import FinalReshapeLayer

class ResNet(object):
    def __init__(self, in_chans, input_time_length,
            projection,  n_layers_per_block,
            n_first_filters, first_filter_length, final_pool_length,
            nonlinearity,
            batch_norm_alpha,
            batch_norm_epsilon,):
        self.__dict__.update(locals())
        del self.self

    def get_layers(self):
        def resnet_residual_block(model, 
            increase_units_factor=None, half_time=False):
            """ With correct batch norm alpha and epsilon. """
            return residual_block(model, 
                batch_norm_epsilon=self.batch_norm_epsilon,
                batch_norm_alpha=self.batch_norm_alpha,
                increase_units_factor=increase_units_factor, 
                half_time=half_time,
                nonlinearity=self.nonlinearity,
                projection=self.projection)

        model = InputLayer([None, self.in_chans, self.input_time_length, 1])
        model = batch_norm(Conv2DLayer(model,
            num_filters=self.n_first_filters, 
            filter_size=(self.first_filter_length,1),
            stride=(1,1), nonlinearity=self.nonlinearity, 
             pad='same', W=lasagne.init.HeNormal(gain='relu')),
             epsilon=self.batch_norm_epsilon,
             alpha=self.batch_norm_alpha)
        for _ in range(self.n_layers_per_block):
            model = resnet_residual_block(model)

        model = resnet_residual_block(model, 
            increase_units_factor=2, half_time=True)
        for _ in range(1,self.n_layers_per_block):
            model = resnet_residual_block(model)

        model = resnet_residual_block(model,
            increase_units_factor=1.5, half_time=True)
        for _ in range(1,self.n_layers_per_block):
            model = resnet_residual_block(model)
            
        model = resnet_residual_block(model, half_time=True)
        for _ in range(1,self.n_layers_per_block):
            model = resnet_residual_block(model)

        model = resnet_residual_block(model, half_time=True)
        for _ in range(1,self.n_layers_per_block):
            model = resnet_residual_block(model)
            
        model = resnet_residual_block(model, half_time=True)
        for _ in range(1,self.n_layers_per_block):
            model = resnet_residual_block(model)
            
        model = resnet_residual_block(model, half_time=True)
        
        # Replacement for global mean pooling
        model = Pool2DLayer(model, pool_size=(self.final_pool_length,1),
            stride=(1,1), 
            mode='average_exc_pad')
        model = Conv2DLayer(model, filter_size=(1,1), num_filters=4,
            W=lasagne.init.HeNormal(), nonlinearity=identity)
        model = FinalReshapeLayer(model)
        model = NonlinearityLayer(model, nonlinearity=softmax)
        return lasagne.layers.get_all_layers(model)

