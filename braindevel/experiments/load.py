import numpy as np
import lasagne
from braindevel.experiments.experiment import create_experiment

def load_model(basename):
    """Load model with params from .yaml and .npy files."""
    exp = create_experiment(basename + '.yaml')
    params = np.load(basename + '.npy')
    model = exp.final_layer
    set_param_values_backwards_compatible(model, params)
            
    return model

def load_exp_and_model(basename, set_invalid_to_NaN=True, seed=9859295):
    """ Loads experiment and model for analysis, sets invalid fillv alues to NaN."""
    model = load_model(basename)
    exp = create_experiment(basename + '.yaml', seed=seed)
    if set_invalid_to_NaN:
        all_layers = lasagne.layers.get_all_layers(model)
        # mark nans to be sure you are doing correct transformations
        # also necessary for transformations to cnt and time activations
        for l in all_layers:
            if hasattr(l, 'invalid_fill_value'):
                l.invalid_fill_value = np.nan
    return exp, model

def set_param_values_backwards_compatible(final_layer, param_values):
    """Backwards compatible for old batch norm layer params."""
    old_batch_norm_layer_used = False
    for param, param_val in zip(lasagne.layers.get_all_params(final_layer), param_values):
        if param.get_value().shape == param_val.shape:
            param.set_value(param_val)
        # account for change in batch norm layer
        elif param.get_value().ndim == 1 and param_val.ndim == 4:
            old_batch_norm_layer_used = True
            assert param.get_value().shape[0] == param_val.shape[1]
            if param.name == 'inv_std': # was std before, now inv std
                # assuming epsilon was always 1e-4 :)
                #epsilon = 1e-4
                #param_val = 1.0 / (param_val + epsilon)
                pass
            else:
                assert param.name in ['mean', 'beta', 'gamma'], (
                    "Unexpected param name {:s}".format(
                    param.name))
            param.set_value(param_val[0,:,0,0])
        else:
            raise ValueError("Different shapes for parameters, constructed model:"
                            "{:s}, param value: {:s}".format(
                str(param.get_value().shape), str(param_val.shape)))
            
    for l in lasagne.layers.get_all_layers(final_layer):
        if (l.__class__.__name__ == 'BatchNormLayer' and
                old_batch_norm_layer_used):
            print("Correcting for old batch norm layer")
            false_mean = l.mean.get_value()
            false_inv_std = l.inv_std.get_value()
            false_beta = l.beta.get_value()
            false_gamma = l.gamma.get_value()
            l.mean.set_value(false_beta)
            l.inv_std.set_value(1.0 / (false_gamma + l.epsilon))
            l.beta.set_value(false_mean)
            l.gamma.set_value(false_inv_std)
    

