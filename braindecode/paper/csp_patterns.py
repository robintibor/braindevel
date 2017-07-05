from braindecode.results.results import ResultPool
import numpy as np
import logging
from braindecode.paper import unclean_sets
log = logging.getLogger(__name__)

def load_patterns(folder='data/models/paper/ours/csp/car/'):
    res_pool = ResultPool()
    res_pool.load_results(folder)

    result_file_names = res_pool.result_file_names()
    results = res_pool.result_objects()

    # sort by dataset filename
    sort_order = np.argsort([r.parameters['dataset_filename'] for r in results])

    result_file_names = np.array(result_file_names)[sort_order]
    results = np.array(results)[sort_order]

    # now sorted
    dataset_names = [r.parameters['dataset_filename'] for r in results]
    all_patterns = []
    clean_mask = []
    all_exps = []
    for file_name, dataset in zip(result_file_names, dataset_names):
        log.info("Loading for {:s}".format(dataset))
        model_file_name = file_name.replace('.result.pkl', '.pkl')
        csp_exp = np.load(model_file_name)
        patterns = csp_exp.binary_csp.patterns
        pattern_arr = patterns_to_single_array(patterns)
        pattern_arr = pattern_arr.squeeze()
        assert not np.any(np.isnan(pattern_arr))
        all_patterns.append(pattern_arr)
        all_exps.append(csp_exp)
        
        if any([s in dataset for s in unclean_sets]):
            clean_mask.append(False)
        else:
            clean_mask.append(True)

    all_patterns = np.array(all_patterns)
    clean_mask = np.array(clean_mask)
    return all_patterns, clean_mask, all_exps

def patterns_to_single_array(patterns):
    pattern_arr = np.ones((patterns.shape[0],
                       patterns.shape[1],
                       patterns.shape[2],
                       patterns[0,0,0].shape[0],
                       patterns[0,0,0].shape[1],)) * np.nan
    for a in xrange(pattern_arr.shape[0]):
        for b in xrange(pattern_arr.shape[1]):
            for c in xrange(pattern_arr.shape[2]):
                pattern_arr[a,b,c] = patterns[a,b,c]
    return pattern_arr