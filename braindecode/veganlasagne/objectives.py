import lasagne
def weighted_binary_cross_entropy(preds, targets, imbalance_factor):
    factor_no_target = (imbalance_factor + 1) / (2.0 *  imbalance_factor)
    factor_target = (imbalance_factor + 1) / 2.0
    loss = lasagne.objectives.binary_crossentropy(preds, targets)
    loss = factor_no_target * loss + loss * targets * factor_target
    return loss