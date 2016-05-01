from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer, NonlinearityLayer
from braindecode.veganlasagne.residual_net import residual_block
from lasagne.nonlinearities import identity
from braindecode.veganlasagne.batch_norm import batch_norm
import lasagne
from braindecode.veganlasagne.layers import FinalReshapeLayer
from lasagne.layers.noise import DropoutLayer
from copy import deepcopy
from lasagne.layers.shape import DimshuffleLayer

class ResNet(object):
    def __init__(self, in_chans, input_time_length,
            projection,  n_layers_per_block,
            n_first_filters, first_filter_length, final_pool_length,
            nonlinearity,
            batch_norm_alpha,
            batch_norm_epsilon,
            drop_before_pool,
            final_aggregator,
            final_nonlin,
            survival_prob,
            split_first_layer,
            add_after_nonlin):
        assert survival_prob <= 1 and survival_prob >= 0
        self.__dict__.update(locals())
        del self.self

    def get_layers(self):
        def resnet_residual_block(model, 
            increase_units_factor=None, half_time=False):
            """Calling with correct attributes from this object. """
            return residual_block(model, 
                batch_norm_epsilon=self.batch_norm_epsilon,
                batch_norm_alpha=self.batch_norm_alpha,
                increase_units_factor=increase_units_factor, 
                half_time=half_time,
                nonlinearity=self.nonlinearity,
                projection=self.projection,
                survival_prob=self.survival_prob,
                add_after_nonlin=self.add_after_nonlin)

        model = InputLayer([None, self.in_chans, self.input_time_length, 1])
        
        if self.split_first_layer:
            # shift channel dim out
            model = DimshuffleLayer(model, (0,3,2,1))
            # first timeconv
            model = Conv2DLayer(model,
                num_filters=self.n_first_filters, 
                filter_size=(self.first_filter_length,1),
                stride=(1,1), nonlinearity=identity, 
                 pad='same', W=lasagne.init.HeNormal(gain='relu'))
            # now spatconv
            model = batch_norm(Conv2DLayer(model,
                num_filters=self.n_first_filters, 
                filter_size=(1,self.in_chans),
                stride=(1,1), nonlinearity=self.nonlinearity, 
                 pad=0, W=lasagne.init.HeNormal(gain='relu')),
                 epsilon=self.batch_norm_epsilon,
                 alpha=self.batch_norm_alpha)
        else:
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
        
        if self.drop_before_pool:
            model = DropoutLayer(model, p=0.5)
        # Replacement for global mean pooling
        if self.final_aggregator == 'pool':
            model = Pool2DLayer(model, pool_size=(self.final_pool_length,1),
                stride=(1,1), mode='average_exc_pad')
            model = Conv2DLayer(model, filter_size=(1,1), num_filters=4,
                W=lasagne.init.HeNormal(), nonlinearity=identity)
        elif self.final_aggregator == 'conv':
            model = Conv2DLayer(model, filter_size=(self.final_pool_length,1), 
                num_filters=4, W=lasagne.init.HeNormal(), nonlinearity=identity)
        else:
            raise ValueError("Unknown final aggregator {:s}".format(
                self.final_aggregator))
            
        model = FinalReshapeLayer(model)
        model = NonlinearityLayer(model, nonlinearity=self.final_nonlin)
        model = set_survival_probs_to_linear_decay(model, self.survival_prob)
        return lasagne.layers.get_all_layers(model)

def set_survival_probs_to_linear_decay(model, survival_prob):
    model = deepcopy(model)
    all_layers = lasagne.layers.get_all_layers(model)
    random_switch_layers = [l for l in all_layers 
        if hasattr(l, '_survival_prob') and 
            l.__class__.__name__ == 'RandomSwitchLayer']
    
    n_switch_layers = len(random_switch_layers)
    
    for i_switch_layer, layer in enumerate(random_switch_layers):
        #http://arxiv.org/pdf/1603.09382v2.pdf#page=6
        # Note this goes from 1 to .. (l-1) / l * p_l
        # not from 1 to p_l as paper says...
        # Seems to be same in their code though?
        # https://github.com/yueatsprograms/Stochastic_Depth/blob/e55dc1d74ba22ba7c56331aa1db35db048e3881e/main.lua#L98-L107
        this_prob = 1 - ((i_switch_layer / float(n_switch_layers)) * 
            (1.0 - survival_prob))
        layer._survival_prob = this_prob
    return model
