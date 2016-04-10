from braindecode.veganlasagne.batch_norm import batch_norm
from lasagne.layers import Conv2DLayer
import lasagne
from lasagne.nonlinearities import rectify
from lasagne.layers.special import NonlinearityLayer, ExpressionLayer
from lasagne.layers.merge import ElemwiseSumLayer
from lasagne.layers.shape import PadLayer
from braindecode.veganlasagne.layers import StrideReshapeLayer


# create a residual learning building block with two stacked 3x3 convlayers as in paper
def residual_block(l, increase_units_factor=None, 
        half_time=False, projection=False):
    assert projection is False, "projection method not adapted"
    input_num_filters = l.output_shape[1]
    if increase_units_factor is not None:
        out_num_filters = int(input_num_filters*increase_units_factor)
        assert (out_num_filters - input_num_filters) % 2 == 0, ("Need even "
            "number of extra channels in order to be able to pad correctly")
    else:
        out_num_filters = input_num_filters

    stack_1 = batch_norm(Conv2DLayer(l, num_filters=out_num_filters, filter_size=(3,3), 
                                   stride=(1,1), nonlinearity=rectify, pad='same', 
                                   W=lasagne.init.HeNormal(gain='relu')))
    if half_time:
        stack_1 = StrideReshapeLayer(stack_1,n_stride=2)
    stack_2 = batch_norm(Conv2DLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), 
                                   stride=(1,1), nonlinearity=None, pad='same', 
                                   W=lasagne.init.HeNormal(gain='relu')))

    # add shortcut connections
    identity = l
    if half_time:
        # identity shortcut, as option A in paper
        identity = StrideReshapeLayer(l,n_stride=2)
    if increase_units_factor is not None:
        n_extra_chans = out_num_filters - input_num_filters
        identity = PadLayer(identity, [n_extra_chans//2,0,0], batch_ndim=1)
        
    block = NonlinearityLayer(ElemwiseSumLayer([stack_2, identity]),nonlinearity=rectify)

    return block

def multiple_residual_blocks(incoming, n_layers, projection=False):
    pass