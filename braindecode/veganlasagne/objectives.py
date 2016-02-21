import lasagne
import theano.tensor as T
from braindecode.veganlasagne.layers import get_n_sample_preds
from lasagne.objectives import categorical_crossentropy
from theano.tensor.shared_randomstreams import RandomStreams
from lasagne.random import get_rng

def weighted_binary_cross_entropy(preds, targets, imbalance_factor):
    factor_no_target = (imbalance_factor + 1) / (2.0 *  imbalance_factor)
    factor_target = (imbalance_factor + 1) / 2.0
    loss = lasagne.objectives.binary_crossentropy(preds, targets)
    loss = factor_no_target * loss + loss * targets * factor_target
    return loss

def sum_of_losses(preds, targets, final_layer, loss_expressions):
    all_losses = []
    for expression in loss_expressions:
        try:
            loss = expression(preds, targets)
        except TypeError:
            loss = expression(preds, targets, final_layer)
        if loss.ndim > 1:
            loss = loss.mean()
        all_losses.append(loss)
        
    total_loss = sum(all_losses)
    return total_loss

def weight_decay(preds, targets, final_layer, factor):
    params = lasagne.layers.get_all_params(final_layer, regularizable = True)
    loss = factor * sum(T.sum(param ** 2) for param in params)
    return loss

def tied_losses_cnt_model(preds, targets, final_layer, n_pairs):
    n_sample_preds = get_n_sample_preds(final_layer)
    n_classes = final_layer.output_shape[1]
    return tied_losses(preds, n_sample_preds, n_classes, n_pairs)

def tied_losses(preds, n_sample_preds, n_classes, n_pairs):
    preds_per_trial_row = preds.reshape((-1, n_sample_preds, n_classes))
    _srng = RandomStreams(get_rng().randint(1, 2147462579))
    rand_inds = _srng.choice([n_pairs  * 2], n_sample_preds, replace=False)
    loss = categorical_crossentropy(preds_per_trial_row[:,rand_inds[:n_pairs]],
        preds_per_trial_row[:,rand_inds[n_pairs:]])
    return loss

def tied_neighbours_cnt_model(preds, targets, final_layer):
    n_sample_preds = get_n_sample_preds(final_layer)
    n_classes = final_layer.output_shape[1]
    return tied_neighbours(preds, n_sample_preds, n_classes)

def tied_neighbours(preds, n_sample_preds, n_classes):
    preds_per_trial_row = preds.reshape((-1, n_sample_preds, n_classes))
    _srng = RandomStreams(get_rng().randint(1, 2147462579))
    loss = categorical_crossentropy(preds_per_trial_row[:,1:],
        preds_per_trial_row[:,:-1])
    return loss


    