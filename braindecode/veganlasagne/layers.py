from lasagne.layers import Conv2DLayer
from lasagne import init
from lasagne import nonlinearities
import theano.tensor as T

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
