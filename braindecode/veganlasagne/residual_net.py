from braindecode.veganlasagne.batch_norm import batch_norm
from lasagne.layers import Conv2DLayer
import lasagne
from lasagne.layers.special import NonlinearityLayer, ExpressionLayer
from lasagne.layers.merge import ElemwiseSumLayer
from lasagne.layers.shape import PadLayer
from braindecode.veganlasagne.layers import StrideReshapeLayer
from braindecode.veganlasagne.random_switch import RandomSwitchLayer
import theano.tensor as T


# create a residual learning building block with two stacked 3x3 convlayers as in paper
def residual_block(l, batch_norm_alpha, batch_norm_epsilon,
    nonlinearity, survival_prob,
    increase_units_factor=None, half_time=False, projection=False,
    ):
    assert survival_prob <= 1 and survival_prob >= 0
    input_num_filters = l.output_shape[1]
    if increase_units_factor is not None:
        out_num_filters = int(input_num_filters*increase_units_factor)
        assert (out_num_filters - input_num_filters) % 2 == 0, ("Need even "
            "number of extra channels in order to be able to pad correctly")
    else:
        out_num_filters = input_num_filters

    stack_1 = batch_norm(Conv2DLayer(l, num_filters=out_num_filters, filter_size=(3,3), 
                                   stride=(1,1), nonlinearity=nonlinearity, pad='same', 
                                   W=lasagne.init.HeNormal(gain='relu')),
                                   epsilon=batch_norm_epsilon,
                                   alpha=batch_norm_alpha)
    if half_time:
        stack_1 = StrideReshapeLayer(stack_1,n_stride=2)
    stack_2 = batch_norm(Conv2DLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), 
                                   stride=(1,1), nonlinearity=None, pad='same', 
                                   W=lasagne.init.HeNormal(gain='relu')),
                                   epsilon=batch_norm_epsilon,
                                   alpha=batch_norm_alpha)

    # add shortcut connections
    shortcut = l
    if half_time:
        # note since we are only reshaping
        # this is ok both for later identity and later projection
        # 1x1 conv of projection is same if we do it before or after this reshape
        # (would not be true if it was anything but 1x1 conv(!))
        shortcut = StrideReshapeLayer(shortcut,n_stride=2)
    if increase_units_factor is not None:
        if projection:
            # projection shortcut, as option B in paper
            shortcut = batch_norm(Conv2DLayer(shortcut, 
                num_filters=out_num_filters, 
                filter_size=(1,1), stride=(1,1), nonlinearity=None, 
                pad='same', b=None),
               epsilon=batch_norm_epsilon,
               alpha=batch_norm_alpha)
        else:
            # identity shortcut, as option A in paper
            n_extra_chans = out_num_filters - input_num_filters
            shortcut = PadLayer(shortcut, [n_extra_chans//2,0,0], batch_ndim=1)
    block = NonlinearityLayer(ElemwiseSumLayer([stack_2, shortcut]),
        nonlinearity=nonlinearity)
    if survival_prob != 1:
        # Hack to make both be broadcastable along empty third dim
        # Otherwise I get an error that they are of different type:
        # shortcut: TensorType(False,False,False,True)
        # block: TensorType4d(32) or sth
        shortcut = ExpressionLayer(shortcut, lambda x: T.addbroadcast(x, 3))
        block = ExpressionLayer(block, lambda x: T.addbroadcast(x, 3))
        block = RandomSwitchLayer(block, shortcut, survival_prob)
    return block

