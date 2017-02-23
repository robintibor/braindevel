import theano.tensor as T
def masked_loss_func(loss_function):
    """ Return function that computes loss only for those targets that are not -1."""
    def masked_loss_fn(predictions, targets): 
        assert targets.ndim == 1
        target_mask = T.neq(targets, -1) 
        valid_inds = T.nonzero(target_mask)
        return loss_function(predictions[valid_inds], targets[valid_inds])
    return masked_loss_fn
