from lasagne.layers import Conv2DLayer
from lasagne import init
from lasagne import nonlinearities
import theano.tensor as T
import lasagne
import numpy as np
from collections import deque
from copy import copy

def get_used_input_length(final_layer):
    """ Determine how much input in the 0-axis
     the layer actually uses,
    assuming valid convolutions/poolings"""
    
    all_layers = lasagne.layers.get_all_layers(final_layer)
    all_layers = all_layers[::-1]
    # determine start size
    for layer in all_layers:
        if (len(layer.output_shape) == 4):
            n_samples = layer.output_shape[2]
            break
    for layer in all_layers:
        if hasattr(layer, 'stride'):
            n_samples = (n_samples - 1) * layer.stride[0] + 1
        if hasattr(layer, 'pool_size'):
            n_samples = n_samples + layer.pool_size[0] - 1
        if hasattr(layer, 'filter_size'):
            n_samples = n_samples + layer.filter_size[0] - 1
    return n_samples

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
    # Create a different
    # out tensor for each offset from 0 to stride (exclusive),
    # e.g. 0,1,2 for stride 3
    # Then concatenate them together again.
    # From 4 different variants (this, using scan, using output preallocation 
    # + set_subtensor, using scan + output preallocation + set_subtensor)
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
    
def get_3rd_dim_shapes_without_invalids(layer):
    all_layers = lasagne.layers.get_all_layers(layer)
    return get_3rd_dim_shapes_without_invalids_for_layers(all_layers)

def get_3rd_dim_shapes_without_invalids_for_layers(all_layers):
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
    paths = get_all_paths(layer)
    preds_per_path = [np.sum(get_3rd_dim_shapes_without_invalids_for_layers(
        layers)) for layers in paths]
    # all path should have same length
    assert len(np.unique(preds_per_path)) == 1
    return preds_per_path[0]

def get_all_paths(layer, treat_as_input=None):
    """
    This function gathers all paths through the net ending at the given final layer.
    ----------
    layer : Layer or list
        the :class:`Layer` instance for which to gather all layers feeding
        into it, or a list of :class:`Layer` instances.
    treat_as_input : None or iterable
        an iterable of :class:`Layer` instances to treat as input layers
        with no layers feeding into them. They will show up in the result
        list, but their incoming layers will not be collected (unless they
        are required for other layers as well).
    Returns
    -------
    list of list
        a list of lists of :class:`Layer` instances feeding into the given
        instance(s) either directly or indirectly, and the given
        instance(s) themselves, in topological order.
    """
    # We perform a depth-first search. We add a layer to the result list only
    # after adding all its incoming layers (if any) or when detecting a cycle.
    # We use a LIFO stack to avoid ever running into recursion depth limits.
    try:
        queue = deque(layer)
    except TypeError:
        queue = deque([layer])
    seen = set()
    done = set()

    # If treat_as_input is given, we pretend we've already collected all their
    # incoming layers.
    if treat_as_input is not None:
        seen.update(treat_as_input)

    paths_queue = deque()
    paths_queue.appendleft((queue, seen, done))
    all_paths = []
    while paths_queue:
        result = []
        queue, seen, done = paths_queue.pop()
        while queue:
            # Peek at the leftmost node in the queue.
            layer = queue[0]
            if layer is None:
                # Some node had an input_layer set to `None`. Just ignore it.
                queue.popleft()
            elif layer not in seen:
                # We haven't seen this node yet: Mark it and queue all incomings
                # to be processed first. If there are no incomings, the node will
                # be appended to the result list in the next iteration.
                seen.add(layer)
                if hasattr(layer, 'input_layers'):
                    for input_layer in layer.input_layers:
                        # Create a new queue for each input layer
                        # they will be used outside of that path
                        this_queue = copy(queue)
                        this_queue.appendleft(input_layer)
                        this_path_parts = (this_queue, copy(seen), copy(done))
                        paths_queue.appendleft(this_path_parts)

                    queue, seen, done = paths_queue.pop()

                elif hasattr(layer, 'input_layer'):
                    queue.appendleft(layer.input_layer)
            else:
                # We've been here before: Either we've finished all its incomings,
                # or we've detected a cycle. In both cases, we remove the layer
                # from the queue and append it to the result list.
                queue.popleft()
                if layer not in done:
                    result.append(layer)
                    done.add(layer)
        all_paths.append(result)


    return all_paths