from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer, NonlinearityLayer
from braindecode.veganlasagne.residual_net import residual_block
from lasagne.nonlinearities import rectify, softmax
from braindecode.veganlasagne.batch_norm import batch_norm
import lasagne
from braindecode.veganlasagne.layers import FinalReshapeLayer

class ResNet(object):
    def __init__(self, in_chans, input_time_length, n_layers_per_block):
        self.in_chans = in_chans
        self.input_time_length = input_time_length
        self.n_layers_per_block = n_layers_per_block
        
    def get_layers(self):
        model = InputLayer([None, self.in_chans, self.input_time_length, 1])
        model = batch_norm(Conv2DLayer(model, num_filters=24, filter_size=(3,1), 
             stride=(1,1), nonlinearity=rectify, 
             pad='same', W=lasagne.init.HeNormal(gain='relu')))
        for _ in range(self.n_layers_per_block):
            model = residual_block(model)
            
        
        model = residual_block(model, increase_units_factor=2, half_time=True)
        for _ in range(1,self.n_layers_per_block):
            model = residual_block(model)

        model = residual_block(model, increase_units_factor=1.5, half_time=True)
        for _ in range(1,self.n_layers_per_block):
            model = residual_block(model)
            
        model = residual_block(model, half_time=True)
        
        for _ in range(1,self.n_layers_per_block):
            model = residual_block(model)
            
        model = residual_block(model, half_time=True)
        for _ in range(1,self.n_layers_per_block):
            model = residual_block(model)
            
        model = residual_block(model, half_time=True)
        
        
        for _ in range(1,self.n_layers_per_block):
            model = residual_block(model)
            
        model = residual_block(model, half_time=True)
        
        model = Pool2DLayer(model, pool_size=(8,1), stride=(1,1))
        model = Conv2DLayer(model, filter_size=(1,1), num_filters=4, W=lasagne.init.HeNormal())
        model = FinalReshapeLayer(model)
        model = NonlinearityLayer(model, nonlinearity=softmax)
        return lasagne.layers.get_all_layers(model)

