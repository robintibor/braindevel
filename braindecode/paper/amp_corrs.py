import numpy as np
import os.path
import logging
from braindecode.results.results import ResultPool
from braindecode.paper import unclean_sets
from braindecode.analysis.stats import cov_and_var_to_corr
log = logging.getLogger(__name__)


def load_amp_corrs(with_square, with_square_corr, cov_or_corr):
    assert not (with_square and with_square_corr)
    assert cov_or_corr == 'cov' or cov_or_corr == 'corr'
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
        for perturb_name in ('rand_mad', 'rand_std', 'no_dev'):
            file_name_end =  '.{:s}.amp_{:s}s.npy'.format(perturb_name,
                cov_or_corr)
            if with_square:
                file_name_end = '.square' + file_name_end
            if with_square_corr:
                file_name_end = ".corrtosquare" + file_name_end
            file_name = base_name + file_name_end
            assert os.path.isfile(file_name), "Expect {:s} to exist".format(
                file_name)
            this_arr = all_corrs.pop(perturb_name, [])
            this_arr.append(np.load(file_name))
            all_corrs[perturb_name] = this_arr
            
    clean_mask = np.array(clean_mask)
    return all_corrs, clean_mask

def create_meaned_amp_pred_corrs(prefix='',
        folder='data/models/paper/ours/cnt/deep4/car/',
        params='default',
        perturb_names=('no_dev', 'rand_mad', 'rand_std')):
    """This takes computed cov_vars and transforms them to corrs 
    and saves corrs."""
    if params == 'default':
        params = dict(cnt_preprocessors="$cz_zero_resample_car_demean",
            trial_start=1500, trial_stop=4000)
    res_pool = ResultPool()
    res_pool.load_results(folder,
        params=params)
    result_file_names = res_pool.result_file_names()
    results = res_pool.result_objects()
    
    # sort by dataset filename
    sort_order = np.argsort([r.parameters['dataset_filename'] for r in results])
    
    result_file_names = np.array(result_file_names)[sort_order]
    results = np.array(results)[sort_order]
    
    all_base_names = [name.replace('.result.pkl', '')
        for name in result_file_names]
    if prefix != '':
        prefix = '.' + prefix
    for i_file, base_name in enumerate(all_base_names):
        log.info("Loading {:s}".format(results[i_file].parameters['dataset_filename']))
        create_meaned_amp_pred_corrs_for_file(base_name, prefix, perturb_names)

def create_meaned_amp_pred_corrs_for_file(base_name, prefix, perturb_names):
    for perturb_name in perturb_names:
        filename_end = '{:s}.{:s}.amp_cov_vars.npz'.format(prefix, perturb_name)
        filename = base_name + filename_end
        if not os.path.isfile(filename):
            log.warn("File does not exist: {:s}".format(filename))
            continue
        npz_file = np.load(filename)
        this_covs, this_pred_vars, this_amp_vars = np.load(filename)
        this_covs = npz_file['arr_0']
        this_pred_vars = npz_file['arr_1']
        this_amp_vars = npz_file['arr_2']
        this_corrs = transform_to_corrs(this_covs, this_pred_vars, this_amp_vars)
        new_file_name_end = '{:s}.{:s}.amp_cov_var_corrs.npy'.format(prefix, 
            perturb_name)
        new_filename = base_name + new_file_name_end
        assert new_filename != filename
        log.info("Saving {:s}...".format(new_filename))
        np.save(new_filename, np.mean(this_corrs, axis=0))

def transform_to_corrs(this_covs, this_pred_vars, this_amp_vars):
    this_covs = np.array([a for a in this_covs])
    """# Fix mistake in reshapes.. remove this if you recomputed bp perturb corrs
    this_covs = this_covs.reshape(this_covs.shape[0], this_covs.shape[1], -1).reshape(
    this_covs.shape[0], this_covs.shape[1],this_covs.shape[3], this_covs.shape[2])
    this_covs = this_covs.transpose(0,1,3,2)"""
    all_flat_corrs = []
    for one_cov, one_pred_var, one_amp_var in zip(this_covs, this_pred_vars, this_amp_vars):
        flat_cov = one_cov.reshape(one_cov.shape[0],-1)
        flat_corr = cov_and_var_to_corr(flat_cov, one_pred_var, one_amp_var)
        all_flat_corrs.append(flat_corr)
    all_corrs = np.array(all_flat_corrs).reshape(this_covs.shape)
    return all_corrs

def transform_to_corrs_preds_last(this_covs, this_amp_vars, this_pred_vars):
    """expects sensors x freqs x classes shape from this_covs"""
    this_covs = np.array([a for a in this_covs])
    all_flat_corrs = []
    for one_cov, one_pred_var, one_amp_var in zip(this_covs, this_pred_vars, this_amp_vars):
        flat_cov = one_cov.reshape(-1, one_cov.shape[2])
        flat_corr = cov_and_var_to_corr(flat_cov, one_amp_var, one_pred_var)
        all_flat_corrs.append(flat_corr)
    all_corrs = np.array(all_flat_corrs).reshape(this_covs.shape)
    return all_corrs

def load_meaned_amp_pred_corrs(prefix='',
        folder='data/models/paper/ours/cnt/deep4/car/',
        params='default',
        perturb_names=('no_dev','rand_mad', 'rand_std')):
    if params == 'default':
        params = dict(cnt_preprocessors="$cz_zero_resample_car_demean",
            trial_start=1500, trial_stop=4000)
    res_pool = ResultPool()
    res_pool.load_results(folder,
        params=params)
    result_file_names = res_pool.result_file_names()
    results = res_pool.result_objects()
    if prefix != '':
        prefix = '.' + prefix
    
    # sort by dataset filename
    sort_order = np.argsort([r.parameters['dataset_filename'] for r in results])
    
    result_file_names = np.array(result_file_names)[sort_order]
    results = np.array(results)[sort_order]
    
    all_base_names = [name.replace('.result.pkl', '')
        for name in result_file_names]
    clean_mask = []
    all_corrs = dict()
    for i_file, base_name in enumerate(all_base_names):
        # Check that all perturbations exist
        all_perturbations_exist = True
        for perturb_name in perturb_names:
            filename_end =  '{:s}.{:s}.amp_cov_var_corrs.npy'.format(
                prefix, perturb_name)
            filename = base_name + filename_end
            if not os.path.isfile(filename):
                all_perturbations_exist = False
                log.warn("{:s} does not exist".format(
                filename))
        if not all_perturbations_exist:
            log.warn("Skipping {:s} since not all perturbations exist".format(
                base_name))
            continue
                
        # Check that all exist for subject
        if any(s in results[i_file].parameters['dataset_filename'] for s in unclean_sets):
            clean_mask.append(False)
        else:
            clean_mask.append(True)
        for perturb_name in perturb_names:
            filename_end =  '{:s}.{:s}.amp_cov_var_corrs.npy'.format(
                prefix, perturb_name)
            filename = base_name + filename_end
            assert os.path.isfile(filename), ("Expect {:s} to exist".format(
                filename))
            this_arr = all_corrs.pop(perturb_name, [])
            this_corrs = np.load(filename)
            this_arr.append(this_corrs)
            all_corrs[perturb_name] = this_arr
                
    clean_mask = np.array(clean_mask)
    return all_corrs, clean_mask