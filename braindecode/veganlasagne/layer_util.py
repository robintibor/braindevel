import lasagne
import numpy as np
from braindecode.analysis.kaggle import (transform_to_time_activations,
    transform_to_cnt_activations)
import logging
from braindecode.veganlasagne.layers import create_pred_fn,\
    get_input_time_length, get_n_sample_preds
from braindecode.datahandling.batch_iteration import compute_trial_start_end_samples
log = logging.getLogger(__name__)

def compute_trial_acts(model, i_layer, iterator, train_set):
    """Compute activations per trial per sample for given layer of the model.
    
    Parameters
    ----------
    model: Lasagne layer
        Final layer of the model.
    i_layer: int
        Index of layer to compute activations for.
    iterator: DatasetIterator
        Iterator to get batches from.
    train_set: Dataset (Cnt)
        Dataset to use.
    
    Returns
    -------
    trial_acts: 3darray of float
        Activations per trial per sample. #trialx#channelx#sample
    """
    # compute number of inputs per trial
    i_trial_starts, i_trial_ends =  compute_trial_start_end_samples(
                train_set.y, check_trial_lengths_equal=True,
                input_time_length=iterator.input_time_length)
    # +1 since trial ends is inclusive
    n_trial_len = i_trial_ends[0] - i_trial_starts[0] + 1
    n_inputs_per_trial = int(np.ceil(n_trial_len / float(iterator.n_sample_preds)))
    log.info("Create theano function...")
    all_layers = lasagne.layers.get_all_layers(model)
    all_out_fn = create_pred_fn(all_layers[i_layer])
    assert(iterator.input_time_length == get_input_time_length(model))
    assert(iterator.n_sample_preds == get_n_sample_preds(model))
    log.info("Compute activations...")
    all_outs_per_batch = [all_out_fn(batch[0]) 
          for batch in iterator.get_batches(train_set, False)]
    batch_sizes = [len(batch[0]) for batch in iterator.get_batches(train_set, False)]
    all_outs_per_batch = np.array(all_outs_per_batch)
    n_trials = len(i_trial_starts)
    log.info("Transform to trial activations...")
    trial_acts = get_trial_acts(all_outs_per_batch,
                                  batch_sizes, n_trials=n_trials, 
                                n_inputs_per_trial=n_inputs_per_trial,
                                  n_trial_len=n_trial_len, 
                                n_sample_preds=iterator.n_sample_preds)
    log.info("Done.")
    return trial_acts

def get_receptive_field_size(layer):
    """Receptive field size of a single output of the given layer.
    
    Parameters
    ----------
    layer: Lasagne layer
        Layer to compute receptive field size of the outputs from.
        
    Returns
    -------
    receptive_field_size:
        How many samples one output has "seen"/is influenced by.
    """

    all_layers = lasagne.layers.get_all_layers(layer)

    in_layer = all_layers[0]


    receptive_field_end = np.arange(in_layer.shape[2])
    for layer in all_layers:
        if hasattr(layer, 'filter_size'):
            receptive_field_end = receptive_field_end[layer.filter_size[0]-1:]
        if hasattr(layer, 'pool_size'):
            receptive_field_end = receptive_field_end[layer.pool_size[0]-1:]
        if hasattr(layer,'stride'):
            receptive_field_end = receptive_field_end[::layer.stride[0]]
        if hasattr(layer,'n_stride'):
            receptive_field_end = receptive_field_end[::layer.n_stride]

    return receptive_field_end[0] + 1

def get_trial_acts(all_outs_per_batch, batch_sizes, n_trials, n_inputs_per_trial,
                   n_trial_len, n_sample_preds):
    """Compute trial activations from activations of a specific layer.
    
    Parameters
    ----------
    all_outs_per_batch: list of 1darray
        All activations of a specific layer for all batches from the iterator.
    batch_sizes: list
        All batch sizes of all batches.
    n_trials: int
    n_inputs_per_trial: int
        How many inputs/rows are used to predict all samples of one trial. 
        Depends on trial length, number of predictions per input window.
    n_trial_len: int
        Number of samples per trial
    n_sample_preds: int
        Number of predictions per input window.
    
    Returns
    --------
    trial_acts: 3darray (final empty dim removed)
        Activations of this layer for all trials.
    
    """
    time_acts = transform_to_time_activations(all_outs_per_batch,batch_sizes)
    trial_batch_acts = np.concatenate(time_acts, axis=0).reshape(n_trials,n_inputs_per_trial,
        time_acts[0].shape[1], time_acts[0].shape[2], 1)
    trial_acts = [transform_to_cnt_activations(t[np.newaxis], n_sample_preds, 
                                                   n_samples = n_trial_len)
                      for t in trial_batch_acts]
    trial_acts = np.array(trial_acts)
    return trial_acts


def model_structure_equal(final_layer_1, final_layer_2):
    """Compare if two networks have the same structure, i.e. same layers
    with same sizes etc. Ignores if they have different parameters."""
    all_equal = True
    all_layers_1 = lasagne.layers.get_all_layers(final_layer_1)
    all_layers_2 = lasagne.layers.get_all_layers(final_layer_2)
    if len(all_layers_1) != len(all_layers_2):
        log.warn("Unequal number of layers: {:d} and {:d}".format(
            len(all_layers_1), len(all_layers_2)))
        return False
    for l1,l2 in zip(all_layers_1, all_layers_2):
        ignore_keys = ['yaml_src', 'input_var', 'input_layer', '_srng', 'b', 'W', 'params',
                      'std', 'beta', 'mean', 'gamma', 'input_layers', 'reshape_layer']
        if l1.__class__.__name__ != l2.__class__.__name__:
            log.warn("Different classnames {:s} and {:s}".format(
                l1.__class__.__name__, l2.__class__.__name__))
            all_equal = False
        for key in l1.__dict__:
            if key in ignore_keys:
                continue
            if l1.__dict__[key] != l2.__dict__[key]:
                all_equal = False
                log.warn("Different attributes:\n{:s}: {:s} and {:s}".format(
                    key, l1.__dict__[key], l2.__dict__[key]))
    return all_equal

def print_layers(model):
    """Print all layers, including all output shapes """
    print(layers_to_str(model))

def layers_to_str(final_layer):
    all_layers_str = ""
    all_layers = lasagne.layers.get_all_layers(final_layer)
    cur_shape = None
    for i, layer in enumerate(all_layers):
        layer_str = "{:25s}".format(layer.__class__.__name__)
        # Add filter sizes and strides
        filter_size = None
        if hasattr(layer, 'filter_size'):
            filter_size = layer.filter_size
        if hasattr(layer, 'pool_size'):
            filter_size = layer.pool_size
        if filter_size is not None:
            filter_str = "{:d}x{:d}".format(filter_size[0],
                filter_size[1])
            if (hasattr(layer,'stride') and layer.stride != (1,1)):
                filter_str += " ::{:d} ::{:d}".format(layer.stride[0],
                    layer.stride[1])
            if (hasattr(layer,'dilation') and layer.dilation != (1,1)):
                filter_str += " ::{:d} ::{:d}".format(layer.dilation[0],
                    layer.dilation[1])
            layer_str += "{:15s}".format(filter_str)
        # Also for stride reshape layer
        if hasattr(layer, 'n_stride'):
            filter_str = "    ::{:d} ::1".format(layer.n_stride)
            layer_str += "{:15s}".format(filter_str)
        layer_str = "{:2d} {:50s}".format(i, layer_str)
        # Possibly add nonlinearities
        if (hasattr(layer, 'nonlinearity') and 
            hasattr(layer.nonlinearity, 'func_name') and 
            layer.nonlinearity.func_name != 'linear'):
            layer_str += " {:15s}".format(layer.nonlinearity.func_name)
        elif (hasattr(layer, 'nonlinearity') and 
            not hasattr(layer.nonlinearity, 'func_name') and
            hasattr(layer.nonlinearity, 'name')):
            layer_str += " {:15s}".format(layer.nonlinearity.name)
        elif (hasattr(layer, 'merge_function')):
            if hasattr(layer.merge_function, 'func_name'):
                layer_str += " {:15s}".format(layer.merge_function.func_name)
            elif hasattr(layer.merge_function, 'name'):
                layer_str += " {:15s}".format(layer.merge_function.name)
        elif (hasattr(layer, 'pool_size')):
            layer_str += " {:15s}".format(layer.mode)
        else:
            layer_str += " {:15s}".format("")
        # Possibly add changing output shape
        if layer.output_shape != cur_shape:
            layer_str += " {:s}".format(layer.output_shape)
            cur_shape = layer.output_shape
        all_layers_str += layer_str + "\n"
    return all_layers_str