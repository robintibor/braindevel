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
    trial_acts: 4darray of float
        Activations per trial per sample.
    """
    
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
    trial_starts, trial_ends = compute_trial_start_end_samples(train_set.y)
    assert len(np.unique(trial_ends - trial_starts)) == 1
    n_trials = len(trial_starts)
    n_trial_len = np.unique(trial_ends - trial_starts)[0]
    log.info("Transform to trial activations...")
    trial_acts = get_trial_acts(all_outs_per_batch,
                                  batch_sizes, n_trials=n_trials, 
                                n_inputs_per_trial=2,
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