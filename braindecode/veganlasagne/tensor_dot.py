from lasagne.layers import Layer
import theano.tensor as T
import lasagne

class TensorDotLayer(Layer):
    def __init__(self, incoming, n_filters, axis, W=lasagne.init.Normal(),
                 **kwargs):
        super(TensorDotLayer, self).__init__(incoming, **kwargs)
        self.axis = axis
        axis_length = incoming.output_shape[self.axis]
        self.n_filters = n_filters
        self.W = self.add_param(W, (self.n_filters, axis_length), name='W',
                                            regularizable=True, trainable=True)
        
    def get_output_shape_for(self, input_shape):
        """ input_shapes[0] should be examples x time x chans
            input_shapes[1] should be examples x time x chans x filters
        """
        out_shape = list(input_shape)
        out_shape[self.axis] = self.n_filters
        return tuple(out_shape)

    def get_output_for(self, input, **kwargs):
        # 1 for W is axis where weight entries are for one weight..
        out = T.tensordot(input, self.W, axes=(self.axis,1))
        n_dims = len(self.output_shape)
        reshuffle_arr = range(self.axis) + [n_dims-1] + range(self.axis,n_dims-1)
        out = out.dimshuffle(*reshuffle_arr)
        return out