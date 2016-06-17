import numpy as np
import os.path
from braindecode.results.results import ResultPool
from braindecode.paper import unclean_sets

def load_topo_corrs(folder, params, i_layer):
    result_pool = ResultPool()
    result_pool.load_results(folder, params=params)
    result_file_names = result_pool.result_file_names()
    results = result_pool.result_objects()
    # sort by dataset filename
    sort_order = np.argsort([r.parameters['dataset_filename'] for r in results])

    result_file_names = np.array(result_file_names)[sort_order]
    results = np.array(results)[sort_order]

    topo_corrs_per_person = []
    rand_corrs_per_person = []
    clean_mask = []
    for file_name, result in zip(result_file_names, results):
        env_corr_filename = file_name.replace('.result.pkl', '.env_corrs.{:d}.npy'.format(i_layer))
        rand_corr_filename = file_name.replace('.result.pkl', '.env_rand_corrs.{:d}.npy'.format(i_layer))

        assert os.path.isfile(env_corr_filename)
        topo_corrs = np.load(env_corr_filename)
        rand_corrs = np.load(rand_corr_filename)
        if any(s in result.parameters['dataset_filename'] for s in unclean_sets):
            clean_mask.append(False)
        else:
            clean_mask.append(True)
        topo_corrs_per_person.append(topo_corrs)
        rand_corrs_per_person.append(rand_corrs)


    rand_corrs_per_person = np.array(rand_corrs_per_person)
    topo_corrs_per_person = np.array(topo_corrs_per_person)
    clean_mask = np.array(clean_mask)
    return topo_corrs_per_person, rand_corrs_per_person, clean_mask

def load_topo_class_corrs(folder, params):
    result_pool = ResultPool()
    result_pool.load_results(folder, params=params)
    result_file_names = result_pool.result_file_names()
    results = result_pool.result_objects()
    # sort by dataset filename
    sort_order = np.argsort([r.parameters['dataset_filename'] for r in results])

    result_file_names = np.array(result_file_names)[sort_order]
    results = np.array(results)[sort_order]

    topo_corrs_per_person = []
    rand_corrs_per_person = []
    clean_mask = []
    for file_name, result in zip(result_file_names, results):
        env_corr_filename = file_name.replace('.result.pkl', '.env_corrs.class.npy')
        env_rand_corr_filename = file_name.replace('.result.pkl', '.env_rand_corrs.class.npy')

        assert os.path.isfile(env_corr_filename)
        assert os.path.isfile(env_rand_corr_filename)
        topo_corrs = np.load(env_corr_filename)
        rand_corrs = np.load(env_rand_corr_filename)
        if any(s in result.parameters['dataset_filename'] for s in unclean_sets):
            clean_mask.append(False)
        else:
            clean_mask.append(True)
        topo_corrs_per_person.append(topo_corrs)
        rand_corrs_per_person.append(rand_corrs)
    topo_corrs_per_person = np.array(topo_corrs_per_person)
    rand_corrs_per_person = np.array(rand_corrs_per_person)
    clean_mask = np.array(clean_mask)
    return topo_corrs_per_person, rand_corrs_per_person, clean_mask

def load_topo_corrs_for_layers(folder, params, i_all_layers):
    topo_corrs_by_layer = []
    rand_corrs_by_layer= []
    for i_layer in i_all_layers:
        topo_corrs, rand_corrs, clean_mask = load_topo_corrs(folder,
                   params, i_layer)
        topo_corrs_by_layer.append(topo_corrs)
        rand_corrs_by_layer.append(rand_corrs)
    return topo_corrs_by_layer, rand_corrs_by_layer, clean_mask