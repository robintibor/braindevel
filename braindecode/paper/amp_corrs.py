import numpy as np
import os.path
from braindecode.results.results import ResultPool
from braindecode.paper import unclean_sets

def load_amp_corrs(with_square):
    res_pool = ResultPool()
    res_pool.load_results('data/models/paper/ours/cnt/deep4/car/',
        params=dict(sensor_names="$all_EEG_sensors", batch_modifier="null",
        low_cut_off_hz="null", first_nonlin="$elu"))
    result_file_names = res_pool.result_file_names()
    results = res_pool.result_objects()
    
    # sort by dataset filename
    sort_order = np.argsort([r.parameters['dataset_filename'] for r in results])
    
    result_file_names = np.array(result_file_names)[sort_order]
    results = np.array(results)[sort_order]
    
    
    
    all_base_names = [name.replace('.result.pkl', '')
        for name in result_file_names]
    clean_mask = []
    all_corrs = dict()
    for i_file, base_name in enumerate(all_base_names):
        if any(s in results[i_file].parameters['dataset_filename'] for s in unclean_sets):
            clean_mask.append(False)
        else:
            clean_mask.append(True)
        for perturb_name in ('rand_mad', 'rand_std', 'shuffle'):
            file_name_end =  '.{:s}.amp_corrs.npy'.format(perturb_name)
            if with_square:
                file_name_end = '.square' + file_name_end
            file_name = base_name + file_name_end
            assert os.path.isfile(file_name)
            this_arr = all_corrs.pop(perturb_name, [])
            this_arr.append(np.load(file_name))
            all_corrs[perturb_name] = this_arr
            
    clean_mask = np.array(clean_mask)
    return all_corrs, clean_mask