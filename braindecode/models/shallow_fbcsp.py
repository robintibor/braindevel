import lasagne
from lasagne.layers.input import InputLayer
from lasagne.layers.shape import DimshuffleLayer
from lasagne.layers.conv import Conv2DLayer
from lasagne.nonlinearities import identity, softmax
from lasagne.layers.special import NonlinearityLayer
from lasagne.layers.pool import Pool2DLayer
from lasagne.layers.noise import DropoutLayer
from braindecode.veganlasagne.layers import (Conv2DAllColsLayer,
    StrideReshapeLayer, FinalReshapeLayer)
from braindecode.veganlasagne.batch_norm import BatchNormLayer
from braindecode.veganlasagne.nonlinearities import square, safe_log


class ShallowFBCSPNet(object):
    def __init__(self, in_chans, input_time_length, n_classes,
            n_filters_time=40,
            filter_time_length=25,
            n_filters_spat=40,
            pool_time_length=75,
            pool_time_stride=15,
            final_dense_length=30,
            conv_nonlin=square,
            pool_mode='average_exc_pad',
            pool_nonlin=safe_log,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            drop_prob=0.5):
        self.__dict__.update(locals())
        del self.self
        
    def get_layers(self):
        l = InputLayer([None, self.in_chans, self.input_time_length, 1])
        if self.split_first_layer:
            l = DimshuffleLayer(l, pattern=[0,3,2,1])
            l = Conv2DLayer(l,
                num_filters=self.n_filters_time,
                filter_size=[self.filter_time_length, 1], 
                nonlinearity=identity,
                name='time_conv')
            l = Conv2DAllColsLayer(l, 
                num_filters=self.n_filters_spat,
                filter_size=[1,-1],
                nonlinearity=identity,
                name='spat_conv')
        else: #keep channel dim in first dim, so it will also be convolved over
            l = Conv2DLayer(l,
                num_filters=self.num_filters_time,
                filter_size=[self.filter_time_length, 1], 
                nonlinearity=identity,
                name='time_conv')
        if self.batch_norm:
            l = BatchNormLayer(l, epsilon=1e-4, alpha=self.batch_norm_alpha,
                nonlinearity=self.conv_nonlin)
        else:
            l = NonlinearityLayer(l, nonlinearity=self.conv_nonlin)
    
        l = Pool2DLayer(l, 
            pool_size=[self.pool_time_length, 1],
            stride=[1,1],
            mode=self.pool_mode)
        l = NonlinearityLayer(l, self.pool_nonlin)
        l = StrideReshapeLayer(l, n_stride=self.pool_time_stride)
        l = DropoutLayer(l, p=self.drop_prob)

        l = Conv2DLayer(l, num_filters=self.n_classes,
            filter_size=[self.final_dense_length, 1], nonlinearity=identity,
            name='final_dense')
        l = FinalReshapeLayer(l)
        l = NonlinearityLayer(l, softmax)
        return lasagne.layers.get_all_layers(l)