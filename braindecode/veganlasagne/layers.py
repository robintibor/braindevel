from lasagne.layers import Conv2DLayer
from lasagne import init
from lasagne import nonlinearities
import theano.tensor as T
import lasagne
import numpy as np

class Conv2DAllColsLayer(Conv2DLayer):
    """Convolutional layer always convolving over the full height
    of the layer before. See Conv2DLayer of lasagne for arguments.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 convolution=T.nnet.conv2d, **kwargs):
        input_shape = incoming.output_shape
        assert filter_size[1] == -1, ("Please specify second dimension as -1"
            " , this dimension wil be replaced by number of cols of input shape")
        filter_size = [filter_size[0], input_shape[3]]
        super(Conv2DAllColsLayer, self).__init__(incoming, num_filters, 
            filter_size, stride=stride,
             pad=pad, untie_biases=untie_biases,
             W=W, b=b, nonlinearity=nonlinearity,
             convolution=convolution, **kwargs)

def reshape_for_stride_theano(topo_var, topo_shape, n_stride, 
        invalid_fill_value=0):
    assert topo_shape[3] == 1
    out_length = int(np.ceil(topo_shape[2] / float(n_stride)))
    reshaped_out=[]
    n_filt = topo_shape[1]
    reshape_shape = (topo_var.shape[0], n_filt, out_length, 1)
    for i_stride in xrange(n_stride):
        reshaped_this = T.ones(reshape_shape, dtype=np.float32) * invalid_fill_value
        i_length = int(np.ceil((topo_shape[2] - i_stride) / float(n_stride)))
        reshaped_this = T.set_subtensor(reshaped_this[:,:,:i_length], 
            topo_var[:,:,i_stride::n_stride])
        reshaped_out.append(reshaped_this)
    reshaped_out = T.concatenate(reshaped_out)
    return reshaped_out

def get_output_shape_after_stride(input_shape, n_stride):
    time_length_after = int(np.ceil(input_shape[2] / float(n_stride)))
    output_shape = [None, input_shape[1], time_length_after, 1]
    return output_shape

class StrideReshapeLayer(lasagne.layers.Layer):
    def __init__(self, incoming, n_stride, invalid_fill_value=0, **kwargs):
        self.n_stride = n_stride
        self.invalid_fill_value = invalid_fill_value
        super(StrideReshapeLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        return reshape_for_stride_theano(input, self.input_shape,self.n_stride,
            invalid_fill_value=self.invalid_fill_value)

    def get_output_shape_for(self, input_shape):
        assert input_shape[3] == 1, "Not tested for nonempty last dim"
        return get_output_shape_after_stride(input_shape, self.n_stride)
    
class FinalReshapeLayer(lasagne.layers.Layer):
    def __init__(self, incoming, remove_invalids=True, **kwargs):
        self.remove_invalids = remove_invalids
        super(FinalReshapeLayer,self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        # Put all samples into their own "batch row"
        # afterwards tensor should have dims #predsamples x #classes x 1 x 1
        # fill value should never be needed, so it shd be safe to set to nan
        input = reshape_for_stride_theano(input, self.input_shape,
            n_stride=self.input_shape[2], invalid_fill_value=np.nan)
        # Reshape/flatten into #predsamples x #classes
        input = input.dimshuffle(1,0,2,3).reshape((self.input_shape[1],-1)).T
        if self.remove_invalids:
            # remove invalid values (possibly nans still contained before)
            lengths_3rd_dim = get_3rd_dim_shapes_without_NaNs(self)
            input_var = lasagne.layers.get_all_layers(self)[0].input_var
            input = input[:input_var.shape[0] * np.sum(lengths_3rd_dim)]
        return input
        
    def get_output_shape_for(self, input_shape):
        assert input_shape[3] == 1, "Not tested for nonempty last dim"
        return [None, input_shape[1]]
    
def get_3rd_dim_shapes_without_NaNs(layer):
    all_layers = lasagne.layers.get_all_layers(layer)
    cur_lengths = np.array([all_layers[0].output_shape[2]])
    # todelay: maybe redo this by using get_output_shape_for function?
    for l in all_layers:
        if hasattr(l, 'filter_size'):
            cur_lengths = cur_lengths - l.filter_size[0] + 1
        if hasattr(l, 'pool_size'):
            cur_lengths = cur_lengths - l.pool_size[0] + 1
        if hasattr(l, 'n_stride'):
            # maybe it should be floor not ceil?
            cur_lengths = np.array([int(np.ceil((length - i_stride) / 
                                               float(l.n_stride)))
                for length in cur_lengths for i_stride in range(l.n_stride)])
    return cur_lengths