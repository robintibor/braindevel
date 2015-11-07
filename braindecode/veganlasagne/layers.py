from lasagne.layers import Conv2DLayer
from lasagne import init
from lasagne import nonlinearities
import theano.tensor as T
import lasagne
import numpy as np
import theano

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
    assert topo_shape[3] == 1, ("Not tested for nonempty third dim, "
        "might work though")
    # collect all new "rows", create a different
    # out tensor for each offset from 0 to stride (exclusive),
    # e.g. 0,1,2 for stride 3
    # Then concatenate them together again
    # from different variants (using scan, using output preallocation 
    # + set_subtensor)
    # this was the fastest, but only by a few percent
    
    n_third_dim = int(np.ceil(topo_shape[2] / float(n_stride)))
    reshaped_out = []
    reshape_shape = (topo_var.shape[0], topo_shape[1], n_third_dim, topo_shape[3])
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
        # before we have sth like this (example where there was only a stride 2
        # in the computations before, and input lengh just 5)
        # showing with 1-based indexing here, sorry ;)
        # batch 1 sample 1, batch 1 sample 3, batch 1 sample 5
        # batch 2 sample 1, batch 2 sample 3, batch 2 sample 5
        # batch 1 sample 2, batch 1 sample 4, batch 1 NaN/invalid
        # batch 2 sample 2, batch 2 sample 4, batch 2 NaN/invalid
        # and this matrix for each filter/class... so if we transpose this matrix for
        # each filter, we get 
        # batch 1 sample 1, batch 2 sample 1, batch 1 sample 2, batch 2 sample 2
        # batch 1 sample 2, ...
        # ...
        # after flattening past the filter dim we then have
        # batch 1 sample 1, batch 2 sample1, ..., batch 1 sample 2, batch 2 sample 2
        # which is our final output shape:
        # (sample 1 for all batches), (sample 2 for all batches), etc
        # any further reshaping should happen outside of theano to speed up compilation
         
        # Reshape/flatten into #predsamples x #classes
        input = input.dimshuffle(1,2,0,3).reshape((self.input_shape[1],
            -1)).T
        if self.remove_invalids:
            # remove invalid values (possibly nans still contained before)
            n_sample_preds = get_n_sample_preds(self)
            input_var = lasagne.layers.get_all_layers(self)[0].input_var
            input = input[:input_var.shape[0] * n_sample_preds]
        return input
        
    def get_output_shape_for(self, input_shape):
        assert input_shape[3] == 1, ("Not tested and thought about " 
            "for nonempty last dim, likely not to work")
        return [None, input_shape[1]]
    
def get_3rd_dim_shapes_without_Invalids(layer):
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

def get_n_sample_preds(layer):
    return np.sum(get_3rd_dim_shapes_without_Invalids(layer))
