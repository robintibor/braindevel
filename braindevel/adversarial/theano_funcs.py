import theano.tensor as T
import lasagne
import theano
from lasagne.updates import adam
from theano.ifelse import ifelse
import numpy as np

def mean_or_zero_if_empty(losses):
    return ifelse(T.eq(losses.size,0), np.array(0).astype(losses.dtype),
        T.mean(losses))

def filter_unequal_1(var, targets):
    valid_inds = T.nonzero(T.all(T.neq(targets,-1), axis=1))
    return var[valid_inds]

def create_pred_loss_train_adv_fn(final_layer, final_adv_time,
        main_loss_expression, adv_loss_expression,
        adv_weight_time, learning_rate=1e-3):
    in_sym = T.ftensor4()
    targets_mrk = T.fmatrix()
    targets_time = T.fmatrix()
    out_mrk, out_time = lasagne.layers.get_output(
        [final_layer, final_adv_time],
        input_var=in_sym,
      inputs=in_sym, deterministic=False)
    out_mrk = filter_unequal_1(out_mrk, targets_mrk)
    out_time = filter_unequal_1(out_time, targets_time)
    
    valid_t_mrk =  filter_unequal_1(targets_mrk, targets_mrk)
    loss_mrk = main_loss_expression(out_mrk, valid_t_mrk)
    valid_t_time=  filter_unequal_1(targets_time, targets_time)
    loss_time = main_loss_expression(out_time, valid_t_time)
    
    class_params = lasagne.layers.get_all_params(final_layer,
        trainable=True)
    time_params = lasagne.layers.get_all_params(final_adv_time,
        trainable=True)
    only_time_params = [p for p in time_params if p not in class_params]
    
    all_layers_mrk = lasagne.layers.get_all_layers(final_layer)
    all_layers_time = lasagne.layers.get_all_layers(final_adv_time)
    only_time_layers = [l for l in all_layers_time if l not in all_layers_mrk]
    adv_loss_time = adv_loss_expression(out_time,
        1 - valid_t_time)
    
    total_loss_mrk = mean_or_zero_if_empty(loss_mrk) + (
        adv_weight_time * mean_or_zero_if_empty(adv_loss_time))
    total_loss_mrk = total_loss_mrk + (1e-5 * 
        lasagne.regularization.regularize_network_params(
            final_layer,
            lasagne.regularization.l2))
    
    updates_mrk = adam(total_loss_mrk, class_params,
        learning_rate=learning_rate)
    
    total_loss_time = mean_or_zero_if_empty(loss_time) + 1e-5 * (
        lasagne.regularization.regularize_layer_params(only_time_layers,
        lasagne.regularization.l2))
    
    updates_time = adam(total_loss_time, only_time_params,
        learning_rate=learning_rate)
    
    for updated_param in updates_time:
        assert updated_param not in updates_mrk
    
    all_updates = updates_mrk.copy()
    all_updates.update(updates_time)
    
    pred_loss_train_fn = theano.function([in_sym, targets_mrk, targets_time],
        [out_mrk, out_time,
        total_loss_mrk, total_loss_time],
         updates=all_updates)
    
    test_outs_mrk, test_outs_time =lasagne.layers.get_output(
        [final_layer, final_adv_time],
        input_var=in_sym,
        inputs=in_sym, deterministic=True)
    test_outs_mrk = filter_unequal_1(test_outs_mrk, targets_mrk)
    test_outs_time = filter_unequal_1(test_outs_time, targets_time)
    
    test_loss_mrk = main_loss_expression(test_outs_mrk, valid_t_mrk)
    test_loss_time = main_loss_expression(test_outs_time, valid_t_time)
    test_adv_loss_time = adv_loss_expression(test_outs_time,
        1 - valid_t_time)
    test_loss_mrk = mean_or_zero_if_empty(test_loss_mrk) +  (
        adv_weight_time * mean_or_zero_if_empty(test_adv_loss_time))
    pred_loss_fn = theano.function([in_sym, targets_mrk, targets_time],
        [test_outs_mrk, test_outs_time,
        test_loss_mrk, test_loss_time, test_adv_loss_time],)
    return pred_loss_train_fn, pred_loss_fn, all_updates.keys()
