from hyperoptim.parse import cartesian_dict_of_lists_product,\
    product_of_list_of_lists_of_dicts

def lalalala_fn():
    return "hi"

def get_templates():
    return  {}

def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    grid_params = dictlistprod({
        'n_chans': ['a', 'b', 'c'],
    })
    
    return grid_params

def sample_config_params(rng, params):
    return params