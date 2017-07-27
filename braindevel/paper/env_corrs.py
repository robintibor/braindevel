import numpy as np
import os.path
from braindevel.results.results import ResultPool
from braindevel.paper import unclean_sets

default_folder = 'data/models/paper/ours/cnt/deep4/car/'
default_params = dict(cnt_preprocessors="$cz_zero_resample_car_demean")

def load_default_topo_corrs(i_layer=26, with_square=True):
    # default folder and filename
    return load_topo_corrs(folder=default_folder, params=default_params, 
        i_layer=i_layer,
        with_square=with_square)

def load_analysis_results(folder, params, file_name_end):
    result_pool = ResultPool()
    result_pool.load_results(folder, params=params)
    result_file_names = result_pool.result_file_names()
    results = result_pool.result_objects() # sort by dataset filename
    sort_order = np.argsort([r.parameters['dataset_filename'] for r in results])
    result_file_names = np.array(result_file_names)[sort_order]
    results = np.array(results)[sort_order]
    analysis_result_per_person = []
    clean_mask = []
    for file_name, result in zip(result_file_names, results):
        analysis_result_file = file_name.replace('.result.pkl', 
            file_name_end)
        assert os.path.isfile(analysis_result_file)
        analysis_result = np.load(analysis_result_file)
        if any(s in result.parameters['dataset_filename'] for s in unclean_sets):
            clean_mask.append(False)
        else:
            clean_mask.append(True)
        analysis_result_per_person.append(analysis_result)
    analysis_result_per_person = np.array(analysis_result_per_person)
    clean_mask = np.array(clean_mask)
    return analysis_result_per_person, clean_mask


def load_trained_untrained_analysis_results(folder, params, file_name_end_trained,
    file_name_end_untrained):
    results_per_person, clean_mask = load_analysis_results(folder, params, 
        file_name_end_trained)
    rand_results_per_person, clean_mask_2 = load_analysis_results(folder, params, 
        file_name_end_untrained)
    assert np.array_equal(clean_mask, clean_mask_2)
    return results_per_person, rand_results_per_person, clean_mask

def load_topo_corrs(folder, params, i_layer, with_square=True):
    file_name_end = '{:d}.npy'.format(i_layer)
    if with_square:
        file_name_end = 'square.' + file_name_end
        
    file_name_end_trained = '.env_corrs.{:s}'.format(file_name_end)
    file_name_end_untrained = '.env_rand_corrs.{:s}'.format(file_name_end)
    topo_corrs_per_person, clean_mask = load_analysis_results(folder, params, 
        file_name_end_trained)
    rand_corrs_per_person, clean_mask_2 = load_analysis_results(folder, params, 
        file_name_end_untrained)
    assert np.array_equal(clean_mask, clean_mask_2)
    return topo_corrs_per_person, rand_corrs_per_person, clean_mask

def load_topo_class_corrs(folder=default_folder, params=default_params, 
        with_square=True):
    file_name_end = 'class.npy'
    if with_square:
        file_name_end = 'square.' + file_name_end
        
    file_name_end_trained = '.env_corrs.{:s}'.format(file_name_end)
    file_name_end_untrained = '.env_rand_corrs.{:s}'.format(file_name_end)
    topo_corrs_per_person, clean_mask = load_analysis_results(folder, params, 
        file_name_end_trained)
    rand_corrs_per_person, clean_mask_2 = load_analysis_results(folder, params, 
        file_name_end_untrained)
    assert np.array_equal(clean_mask, clean_mask_2)
    return topo_corrs_per_person, rand_corrs_per_person, clean_mask

def load_default_topo_corrs_for_layers(i_all_layers=(8,14,20,26),
        with_square=True):
    # default folder and filename
    return load_topo_corrs_for_layers(folder=default_folder, params=default_params, 
        i_all_layers=i_all_layers,
        with_square=with_square)
    
def load_topo_corrs_for_layers(folder, params, i_all_layers,
        with_square):
    topo_corrs_by_layer = []
    rand_corrs_by_layer= []
    for i_layer in i_all_layers:
        topo_corrs, rand_corrs, clean_mask = load_topo_corrs(folder,
                   params, i_layer, with_square)
        topo_corrs_by_layer.append(topo_corrs)
        rand_corrs_by_layer.append(rand_corrs)
    return topo_corrs_by_layer, rand_corrs_by_layer, clean_mask