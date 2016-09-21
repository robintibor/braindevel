from lasagne.layers.input import InputLayer
from lasagne.layers.special import NonlinearityLayer
from lasagne.layers.shape import DimshuffleLayer, SliceLayer, FlattenLayer
from lasagne.layers.dnn import Pool3DDNNLayer
from lasagne.nonlinearities import sigmoid
import lasagne
import theano.tensor as T
from braindecode.veganlasagne.nonlinearities import safe_log
from braindecode.veganlasagne.recurrent import BandpassLayer
from braindecode.veganlasagne.tensor_dot import TensorDotLayer

class BandpassSquareClassify(object):
    def __init__(self, n_examples, n_time_steps, n_chans,
                 n_filters, n_filt_order, truncate_gradient,
                 n_pool_len=200, n_spat_filters=20):
        self.__dict__.update(locals())
        del self.self

    def get_layers(self):
        in_l = InputLayer((self.n_examples, self.n_time_steps, self.n_chans))
        in_bandpass = InputLayer((self.n_examples, self.n_time_steps, self.n_chans, self.n_filters))


        l_bandpass = BandpassLayer([in_l, in_bandpass], n_filt_order=self.n_filt_order,
                                  truncate_gradient=self.truncate_gradient)
        
        # out comes examples x timesteps x chans x filters
        l_spat_filt = TensorDotLayer(l_bandpass, n_filters=self.n_spat_filters,
            axis=2)
        # still examples x timesteps x chans x filters
        l_square = NonlinearityLayer(l_spat_filt, T.sqr)
        # now adding empty chan dim so we can make pooling per output chan
        l_shape_pad = DimshuffleLayer(l_square, (0,'x',1,2,3))

        # examples x convchans x timesteps x chans x filters
        l_pooled = Pool3DDNNLayer(l_shape_pad, pool_size=(self.n_pool_len,1,1), 
            stride=1, mode='average_exc_pad')

        l_log = NonlinearityLayer(l_pooled, safe_log)

        # removing empty convchan dim again
        l_sliced = SliceLayer(l_log,indices=0,axis=1)
        # now examples x timesteps x chans x filters 
        l_flat = FlattenLayer(l_sliced,outdim=3)
        # now examples x timesteps x features (chans * filters)

        l_dense = TensorDotLayer(l_flat,n_filters=1, axis=2)
        # now examples x timesteps x 1
        l_nonlin = NonlinearityLayer(l_dense, sigmoid)
        return lasagne.layers.get_all_layers(l_nonlin)