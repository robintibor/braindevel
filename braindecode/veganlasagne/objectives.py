import lasagne
from braindecode.veganlasagne.layers import get_n_sample_preds
from lasagne.objectives import categorical_crossentropy, binary_crossentropy
from theano.tensor.shared_randomstreams import RandomStreams
from lasagne.random import get_rng
import theano.tensor as T

def weighted_binary_cross_entropy(preds, targets, imbalance_factor,
        binary_func=lasagne.objectives.binary_crossentropy):
    factor_no_target = (imbalance_factor + 1) / (2.0 *  imbalance_factor)
    factor_target = (imbalance_factor + 1) / 2.0
    loss = binary_func(preds, targets)
    loss = ((factor_no_target * loss * (1-targets)) + 
        (loss * targets * factor_target))
    return loss

def weighted_thresholded_binary_cross_entropy(preds, targets, imbalance_factor,
        lower_threshold):
    loss = weighted_binary_cross_entropy(preds, targets,
        imbalance_factor=imbalance_factor,)
    # preds that are below 0.2 where there is no target, are ignored
    loss_mask = T.or_(T.gt(preds,lower_threshold), T.eq(targets, 1))
    loss = loss * loss_mask
    return loss

def safe_categorical_crossentropy(predictions, targets, eps=1e-8):
    predictions = (T.gt(predictions, eps) * predictions + 
        T.le(predictions, eps) * predictions + eps)
    return categorical_crossentropy(predictions, targets)
def safe_binary_crossentropy(predictions, targets, eps=1e-4):
    # add eps for predictions that are smaller than eps
    predictions = predictions + T.le(predictions, eps) * eps
    # remove eps for predictions that are larger than 1 - eps
    predictions = predictions - T.ge(predictions, 1 - eps) * eps
    return binary_crossentropy(predictions, targets)
    
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

def tied_losses_image_mask(preds, targets):
    """ Should return one tied loss per image"""
    return -T.sum(preds[:,:,1:,1:] * T.log(preds[:,:,:-1,:-1]), axis=(1,2,3))

def weight_decay(preds, targets, final_layer, factor):
    loss = lasagne.regularization.regularize_network_params(final_layer,
        lasagne.regularization.l2)
    return loss * factor

def tied_losses_cnt_model(preds, targets, final_layer, n_pairs):
    n_sample_preds = get_n_sample_preds(final_layer)
    n_classes = final_layer.output_shape[1]
    return tied_losses(preds, n_sample_preds, n_classes, n_pairs)

def tied_losses(preds, n_sample_preds, n_classes, n_pairs):
    preds_per_trial_row = preds.reshape((-1, n_sample_preds, n_classes))
    _srng = RandomStreams(get_rng().randint(1, 2147462579))
    rand_inds = _srng.choice([n_pairs  * 2], n_sample_preds, replace=False)
    part_1 = preds_per_trial_row[:,rand_inds[:n_pairs]]
    part_2 = preds_per_trial_row[:,rand_inds[n_pairs:]]
    # Have to now ensure first values are larger zero
    # for numerical stability :/
    eps = 1e-4
    part_1 = T.maximum(part_1, eps)
    loss = categorical_crossentropy(part_1, part_2)
    return loss

def tied_neighbours_cnt_model(preds, targets, final_layer):
    n_sample_preds = get_n_sample_preds(final_layer)
    n_classes = final_layer.output_shape[1]
    return tied_neighbours(preds, n_sample_preds, n_classes)

def tied_neighbours(preds, n_sample_preds, n_classes):
    eps = 1e-8
    #preds = T.clip(preds, eps, 1-eps)
    preds_per_trial_row = preds.reshape((-1, n_sample_preds, n_classes))
    earlier_neighbours = preds_per_trial_row[:,:-1]
    later_neighbours = preds_per_trial_row[:,1:]
    # Have to now ensure first values are larger zero
    # for numerical stability :/
    # Example of problem otherwise:
    """
    a = T.fmatrix()
    b = T.fmatrix()
    soft_out_a =softmax(a)
    soft_out_b =softmax(b)
    
    loss = categorical_crossentropy(soft_out_a[:,1:],soft_out_b[:,:-1])
    neigh_fn = theano.function([a,b], loss)
    
    neigh_fn(np.array([[0,1000,0]], dtype=np.float32), 
        np.array([[0.1,0.9,0.3]], dtype=np.float32))
    -> inf
    """
    
    # renormalize(?)
    
    earlier_neighbours = (T.gt(earlier_neighbours, eps) * earlier_neighbours + 
        T.le(earlier_neighbours, eps) * earlier_neighbours + eps)
    loss = categorical_crossentropy(earlier_neighbours, later_neighbours)
    return loss
   

def tied_neighbours_cnt_model_logdomain(preds, targets, final_layer):
    n_sample_preds = get_n_sample_preds(final_layer)
    n_classes = final_layer.output_shape[1]
    # just for checkup
    # TODOREMOVE!!
    #return categorical_crossentropy_logdomain(preds, targets)
    return tied_neighbours_logdomain(preds, n_sample_preds, n_classes) 

def tied_neighbours_logdomain(preds, n_sample_preds, n_classes):
    preds_per_trial_row = preds.reshape((-1, n_sample_preds, n_classes))
    earlier_neighbours = preds_per_trial_row[:,:-1]
    later_neighbours = preds_per_trial_row[:,1:]
    # TODOREMOVE? now using kl divergence...
    # have to exponentiate later neighbours that have role of
    # targets in categorical crossentropy (and are therefore assumed
    # to not have been logarithmized)
    loss = categorical_crossentropy_logdomain(earlier_neighbours, 
        T.exp(later_neighbours))
    loss = T.sum(T.exp(earlier_neighbours) * (earlier_neighbours - later_neighbours),
        axis=1)
    return loss

def categorical_crossentropy_logdomain(log_predictions, targets):
    """From https://github.com/Lasagne/Lasagne/issues/332#issuecomment-122328992"""
    return -T.sum(targets * log_predictions, axis=1)
