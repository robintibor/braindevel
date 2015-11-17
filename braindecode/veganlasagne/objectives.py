import lasagne
import theano.tensor as T

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
    