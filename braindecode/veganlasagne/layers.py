from lasagne.layers import Conv2DLayer, NonlinearityLayer, Layer
from lasagne import init
from lasagne import nonlinearities
import theano.tensor as T
import lasagne
import numpy as np
from collections import deque
from copy import copy
from copy import deepcopy
import sys
from lasagne.utils import as_tuple
from lasagne.theano_extensions import padding
import theano
from lasagne.nonlinearities import identity

def print_layers(final_layer):
    """Print all layers, including all output shapes """
    all_layers = lasagne.layers.get_all_layers(final_layer)
    for i, layer in enumerate(all_layers):
        print("{:2d} {:25s} {:s}".format(i, layer.__class__.__name__,
            layer.output_shape))
class BiasLayer(Layer):
    def __init__(self, incoming, bias_value, **kwargs):
        super(BiasLayer, self).__init__(incoming, **kwargs)
        self.b = self.add_param(bias_value, bias_value.shape, name="b",
                                    regularizable=False)

    def get_output_for(self, input, **kwargs):
        if input.ndim == 4:
            return input + self.b.dimshuffle('x',0,'x','x')
        elif input.ndim == 2:
            return input # TODO: somehow fix the error in this case
        # was input + self.b.dimshuffle('x',0) before
        else:
            raise ValueError("Unknown input ndim {:d}".format(input.ndim))
    
def split_out_biases(model):
    """Splits out nonlinearities of layers with weights into nonlinearity
    layers """
    copied_model = deepcopy(model)
    all_layers = lasagne.layers.get_all_layers(copied_model)
    new_final_layer = False
    for i_layer in xrange(len(all_layers)):
        layer = all_layers[i_layer]
        if (hasattr(layer, 'b') and (not isinstance(layer, BiasLayer)) and
                ((i_layer == len(all_layers) - 1) or
                (not isinstance(all_layers[i_layer+1], BiasLayer)))):
            if not (len(layer.output_shape) == 2):
                # hack atm due to bug with softmax optimization
                old_b_val = deepcopy(layer.b.get_value())
                bias_layer = BiasLayer(layer, old_b_val)
                layer.b = None
                if i_layer < len(all_layers) - 1:
                    next_layer = all_layers[i_layer+1]
                    next_layer.input_layer = bias_layer
                else:
                    new_final_layer = True

    if new_final_layer == True:
        return bias_layer
    else:
        return all_layers[-1]
    
def split_out_nonlinearities(model):
    """Splits out nonlinearities of layers with weights into nonlinearity
    layers """
    copied_model = deepcopy(model)

    all_layers = lasagne.layers.get_all_layers(copied_model)

    new_final_layer = False

    for i_layer in xrange(len(all_layers)):
        layer = all_layers[i_layer]
        if (hasattr(layer, 'nonlinearity') and
                layer.nonlinearity.__name__ != 'identity' and 
                layer.nonlinearity.__name__ != 'linear' and 
                hasattr(layer, 'W')):
            old_nonlin = layer.nonlinearity
            new_nonlin_layer = NonlinearityLayer(layer, old_nonlin)
            if i_layer < len(all_layers) - 1:
                next_layer = all_layers[i_layer+1]
                next_layer.input_layer = new_nonlin_layer
            else:
                new_final_layer = True
            layer.nonlinearity = identity

    if new_final_layer ==True:
        all_layers = lasagne.layers.get_all_layers(new_nonlin_layer)
    else:
        all_layers = lasagne.layers.get_all_layers(all_layers[-1])
    return all_layers[-1]

def calculate_predictions(data, start,stop,stride, window_len, pred_fn):
    preds = []
    for i_end_sample in xrange(start,stop,stride):
        i_start_sample = i_end_sample - window_len + 1
        preds.append(pred_fn(data[i_start_sample:i_end_sample+1].T[None,:,:,None]))
    return np.array(preds).squeeze()
    
def create_pred_fn(model):
    inputs = create_suitable_theano_input_var(model)
    output = lasagne.layers.get_output(model, deterministic=True,
        inputs=inputs, input_var=inputs)
    pred_fn = theano.function([inputs], output)
    return pred_fn

def transform_to_normal_net(final_layer):
    """ Transforms cnt/parallel prediction net to a normal net."""
    # copy old model, need to set sys recursion limit up to make sure it can be copied
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(50000)
    new_final_layer = deepcopy(final_layer)
    sys.setrecursionlimit(old_limit)

    # remove stride reshape layers and instead set strides to conv/pool layers again
    # also replace final reshape with flattening layer
    all_layers = lasagne.layers.get_all_layers(new_final_layer)
    last_stride = None
    for i_layer in xrange(len(all_layers) - 1,-1,-1):
        if hasattr(all_layers[i_layer], 'remove_invalids'):
            # replace with final reshape with flattening layer
            all_layers[i_layer + 1].input_layer = lasagne.layers.FlattenLayer(all_layers[i_layer - 1])
        if hasattr(all_layers[i_layer], 'n_stride'):
            # just remove stride reshape layers
            all_layers[i_layer + 1].input_layer = all_layers[i_layer - 1]
            last_stride = all_layers[i_layer].n_stride
        if hasattr(all_layers[i_layer], 'stride') and last_stride is not None:
            # should work for pool and conv layers as both use stride member variable
            # for stride
            all_layers[i_layer].stride = (last_stride, all_layers[i_layer].stride[1])
            last_stride = None

    # Fix shapes by iterating through layers... 
    # Use original input window, only
    # 3rd dim has to be fixed other dims can remain the same
    all_layers = lasagne.layers.get_all_layers(new_final_layer)
    input_time_window = get_model_input_window(final_layer)
    all_layers[0].shape = [None,all_layers[0].shape[1],input_time_window,all_layers[0].shape[3]]
    for l in all_layers[1:]:
        l.input_shape = l.input_layer.output_shape
    
    return new_final_layer

def create_suitable_theano_input_var(layer):
    if len(get_input_shape(layer)) == 2:
        inputs = T.fmatrix()
    else:
        inputs = T.ftensor4()
    return inputs

def create_suitable_theano_output_var(layer):
    if len(layer.output_shape) > 2:
        outputs = T.ftensor4()
    else:
        outputs = T.fmatrix()
    return outputs

def get_input_var(final_layer):
    return lasagne.layers.get_all_layers(final_layer)[0].input_var

def get_input_shape(final_layer):
    return lasagne.layers.get_all_layers(final_layer)[0].shape

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
            assert layer.pad == (0,0)
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
    if topo_shape[0] is not None:
        zero_dim_len = topo_shape[0]
    else:
        zero_dim_len = topo_var.shape[0]
    reshape_shape = (zero_dim_len, topo_shape[1], n_third_dim, topo_shape[3])
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
    if input_shape[0] is None:
        trials_after = None
    else:
        trials_after = int(input_shape[0] * n_stride)
        
    output_shape = (trials_after, input_shape[1], time_length_after, 1)
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
    def __init__(self, incoming, remove_invalids=True, flatten=True,
             **kwargs):
        self.remove_invalids = remove_invalids
        self.flatten = flatten
        super(FinalReshapeLayer,self).__init__(incoming, **kwargs)

    def get_output_for(self, input, input_var=None, **kwargs):
        # need input_var to determine number of trials
        # cannot use input var of entire net since maybe you want to
        # get output given an activation for a later layer...
        #
        # before we have sth like this (example where there was only a stride 2
        # in the computations before, and input length just 5)
        # showing with 1-based indexing here, sorry ;)
        # trial 1 sample 1, trial 1 sample 3, trial 1 sample 5
        # trial 2 sample 1, trial 2 sample 3, trial 2 sample 5
        # trial 1 sample 2, trial 1 sample 4, trial 1 NaN/invalid
        # trial 2 sample 2, trial 2 sample 4, trial 2 NaN/invalid
        # and this matrix for each filter/class... so if we transpose this matrix for
        # each filter, we get 
        # trial 1 sample 1, trial 2 sample 1, trial 1 sample 2, trial 2 sample 2
        # trial 1 sample 2, ...
        # ...
        # after flattening past the filter dim we then have
        # trial 1 sample 1, trial 2 sample1, ..., trial 1 sample 2, trial 2 sample 2
        # which is our output shape which allows to remove invalids easily:
        # (sample 1 for all trials), (sample 2 for all trials), etc
         
        # After removal of invalids,
        #  we reshape again to (trial 1 all samples), (trial 2 all samples)
        
        # Reshape/flatten into #predsamples x #classes
        n_classes = self.input_shape[1]
        input = input.dimshuffle(1,2,0,3).reshape((n_classes, -1)).T
        if input_var is None:
            input_var = lasagne.layers.get_all_layers(self)[0].input_var
        input_shape = lasagne.layers.get_all_layers(self)[0].shape
        if input_shape[0] is not None:
            trials = input_shape[0]
        else:
            trials = input_var.shape[0]
            
        if self.remove_invalids:
            # remove invalid values (possibly nans still contained before)
            n_sample_preds = get_n_sample_preds(self)
                
            input = input[:trials * n_sample_preds]
        
        # reshape to (trialsxsamples) again, i.e.
        # (trial1 all samples), (trial 2 all samples), ...
        # By doing this:
        # transpose to classes x (samplepreds*trials)
        # then reshape to classes x sample preds x trials, 
        # dimshuffle to classes x trials x sample preds to flatten again to
        # final output:
        # (trial 1 all samples), (trial 2 all samples), ...
        
        # if not flatten, instead reshape to:
        #  trials x classes/filters x sample preds x emptydim
        input = input.T.reshape((self.input_shape[1], 
            -1, trials))
        if self.flatten:
            input = input.dimshuffle(0,2,1).reshape((n_classes, -1)).T
        else:
            input = input.dimshuffle(2,0,1,'x')
            
        
        return input
        
    def get_output_shape_for(self, input_shape):
        assert input_shape[3] == 1, ("Not tested and thought about " 
            "for nonempty last dim, likely not to work")
        return [None, input_shape[1]]
    
def get_3rd_dim_shapes_without_invalids(layer):
    all_layers = lasagne.layers.get_all_layers(layer)
    return get_3rd_dim_shapes_without_invalids_for_layers(all_layers)

def get_3rd_dim_shapes_without_invalids_for_layers(all_layers):
    # handle as special case that first layer is dim shuffle layer that shuffles last 2 dims.. 
    # hack for the stupiding kaggle code
    if hasattr(all_layers[1],  'pattern') and all_layers[1].pattern == (0,1,3,2):
        all_layers = all_layers[1:]

    cur_lengths = np.array([all_layers[0].output_shape[2]])
    # todelay: maybe redo this by using get_output_shape_for function?
    for l in all_layers:
        if hasattr(l, 'filter_size'):
            if l.pad == (0,0):
                cur_lengths = cur_lengths - l.filter_size[0] + 1
            elif l.pad == ((l.filter_size[0] - 1) / 2, (l.filter_size[1] - 1) / 2):
                cur_lengths = cur_lengths
            else:
                print l
                raise ValueError("Not implemented this padding:", l.pad, l, l.filter_size)
                
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
    assert len(np.unique(preds_per_path)) == 1, ("All paths, should have same "
        "lengths, pathlengths are" + str(preds_per_path))
    return preds_per_path[0]


def get_input_time_length(layer):
    return lasagne.layers.get_all_layers(layer)[0].shape[2]

def get_model_input_window(cnt_model):
    return get_input_time_length(cnt_model) - get_n_sample_preds(cnt_model) + 1

def get_model_input_windows_in_folder(folder_name):
    models = get_models_in_folder(folder_name)
    return [get_model_input_window(m) for m in models]

def get_models_in_folder(folder_name):
    from glob import glob
    files = glob(folder_name + '/*[0-9].pkl')
    sorted_files = sorted(files,
        key=lambda f: int(f.split('/')[-1].split('.')[0]))
    models = [np.load(f) for f in sorted_files]
    return models

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

def unfold_filters(channel_weights, kernel_weights):
    """ Unfolds bc and b01 weights into prober bc01 weights """
    return T.batched_tensordot(channel_weights.dimshuffle(0,1,'x'),
        kernel_weights.dimshuffle(0,'x', 1,2), axes=((2), (1)))

class SeparableConv2DLayer(Conv2DLayer):
    """ Doing a separable convolution (channel and 2d-filter convolution) by unfolding weights.
    Basically regularizes weights to be a combination of channel weights and 2d-kernels.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 convolution=T.nnet.conv2d, **kwargs):
        super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size,
            **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, 2)
        self.stride = as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.convolution = convolution

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')

        if pad == 'valid':
            self.pad = (0, 0)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, 2, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2], self.
                                output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.
        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels + 
                self.filter_size[0] * self.filter_size[1])


    def get_unfolded_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1])
    
    def get_output_for(self, input, input_shape=None, **kwargs):
        # The optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape
            
        n_chans = self.input_shape[1]
        channel_weights = self.W[:, :n_chans]
        kernel_weights = self.W[:,n_chans:]
        kernel_weights = kernel_weights.reshape((self.num_filters,
            self.filter_size[0], self.filter_size[1]))
        
        unfolded_W = unfold_filters(channel_weights, kernel_weights)

        if self.stride == (1, 1) and self.pad == 'same':
            # simulate same convolution by cropping a full convolution
            conved = self.convolution(input, unfolded_W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=self.get_unfolded_W_shape(),
                                      border_mode='full')
            crop_x = self.filter_size[0] // 2
            crop_y = self.filter_size[1] // 2
            conved = conved[:, :, crop_x:-crop_x or None,
                            crop_y:-crop_y or None]
        else:
            # no padding needed, or explicit padding of input needed
            if self.pad == 'full':
                border_mode = 'full'
                pad = [(0, 0), (0, 0)]
            elif self.pad == 'same':
                border_mode = 'valid'
                pad = [(self.filter_size[0] // 2,
                        self.filter_size[0] // 2),
                       (self.filter_size[1] // 2,
                        self.filter_size[1] // 2)]
            else:
                border_mode = 'valid'
                pad = [(self.pad[0], self.pad[0]), (self.pad[1], self.pad[1])]
            if pad != [(0, 0), (0, 0)]:
                input = padding.pad(input, pad, batch_ndim=2)
                input_shape = (input_shape[0], input_shape[1],
                               None if input_shape[2] is None else
                               input_shape[2] + pad[0][0] + pad[0][1],
                               None if input_shape[3] is None else
                               input_shape[3] + pad[1][0] + pad[1][1])
            conved = self.convolution(input, unfolded_W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=self.get_unfolded_W_shape(),
                                      border_mode=border_mode)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)