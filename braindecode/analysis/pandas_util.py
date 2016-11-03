from braindecode.results.results import get_final_misclasses, ResultPool,\
    get_training_times
import pandas as pd
import numpy as np
from braindecode.analysis.stats import perm_mean_diff_test
from pandas.core.index import MultiIndex
from braindecode.scripts.print_results import prettify_word
from collections import OrderedDict
import logging
from braindecode.util import merge_dicts
import os.path
from scipy.stats.morestats import wilcoxon
log = logging.getLogger(__name__)

class MetaDataFrame(pd.DataFrame):
    # from https://github.com/pydata/pandas/issues/2485#issuecomment-174577149
    _metadata = ['attrs']

    @property
    def _constructor(self):
        return MetaDataFrame

    def _combine_const(self, other, *args, **kwargs):
        return super(MetaDataFrame, self)._combine_const(other, *args, **kwargs).__finalize__(self)

def load_results_for_df(df, prefix="/home/schirrmr/motor-imagery/",
        debug_exps=False):
    if not debug_exps:
        result_file_names = [os.path.join(save_path, str(exp_id) + ".result.pkl") 
                 for save_path, exp_id in zip(np.array(df.save_path), np.array(df.index))]
    else:
        assert debug_exps
        result_file_names = [os.path.join(save_path, 'debug', str(exp_id) + ".result.pkl") 
                 for save_path, exp_id in zip(np.array(df.save_path), np.array(df.index))]
    results = [np.load(prefix + name) for name in result_file_names]
    return results

def remove_dollar(param_val):
    if isinstance(param_val, str) and param_val[0] == '$':
        return param_val[1:]
    else:
        return param_val

def load_data_frame(folder, params=None, shorten_headers=True):
    res_pool = ResultPool()
    res_pool.load_results(folder, params=params)
    result_objs = res_pool.result_objects()
    varying_params = res_pool.varying_params()
    constant_params = res_pool.constant_params()
    file_names = res_pool.result_file_names()
    data_frame = to_data_frame(file_names, result_objs, varying_params,
        constant_params, shorten_headers=shorten_headers)
    data_frame.attrs = {'folder': folder, 'params': params}
    return data_frame

def to_data_frame(file_names, result_objs, varying_params, constant_params,
        shorten_headers=True):
    all_params = [merge_dicts(var, constant_params) for var in varying_params]
    file_numbers = [int(f.split('/')[-1].split('.')[0]) for f in file_names]
    
    # remove dollars
    for param_dict in all_params:
        for key, val in param_dict.iteritems():
            param_dict[key] = remove_dollar(val)
    
    param_keys = all_params[0].keys()
    param_vals = [[v[key] for key in param_keys] for v in all_params]
    # transform lists to tuples to make them hashable
    param_vals = [[to_tuple_if_list(v) for v in var_list] for var_list in param_vals]
    param_vals = np.array(param_vals, dtype=object)
    test_accs = (1 - get_final_misclasses(result_objs, 'test')) * 100
    train_accs = (1 - get_final_misclasses(result_objs, 'train')) * 100
    training_times = get_training_times(result_objs)
    # try adding sample accuracies, might exist, might not 
    sample_accs_exist = (hasattr(result_objs[0], 'monitor_channels') and
        'test_sample_misclass' in result_objs[0].monitor_channels)
    if sample_accs_exist:
        test_sample_accs = (1 - get_final_misclasses(result_objs, 'test_sample')) * 100
        train_sample_accs = (1 - get_final_misclasses(result_objs, 'train_sample')) * 100
        vals_and_misclasses =  np.append(param_vals, 
            np.array([training_times, test_accs, test_sample_accs,
                train_accs, train_sample_accs]).T, 
            axis=1)
    else:
        vals_and_misclasses = np.append(param_vals, 
            np.array([training_times, test_accs, train_accs]).T, 
            axis=1)
    if shorten_headers:
        param_keys = [prettify_word(key) for key in param_keys]
    
    if sample_accs_exist:
        all_keys = param_keys + ['time', 'test', 'test_sample', 'train',
            'train_sample']
    else:
        all_keys = param_keys + ['time', 'test', 'train']
        
    data_frame = MetaDataFrame(vals_and_misclasses, index=file_numbers, 
        columns=all_keys)
    data_frame = to_numeric_where_possible(data_frame)
    data_frame.time = pd.to_timedelta(np.round(data_frame.time), unit='s')
    return data_frame

def to_tuple_if_list(val):
    if isinstance(val, list):
        return tuple(val)
    else:
        return val

def get_dfs_for_matched_exps_with_different_vals(df, key):
    """Matched in the sense all other keys/parameters are the same."""
    param_keys = set(df.keys()) - set(['test', 'time', 'train',
    'test_sample', 'train_sample'])


    possible_vals = np.unique(df[key])
    other_param_keys = list(param_keys - set([key]))
    joined_frame = None
    for i_value in range(0, len(possible_vals)):
        val = possible_vals[i_value]
        frame = df[df[key] == val]
        if joined_frame is None:
            joined_frame = frame
        else:
            joined_frame = joined_frame.merge(frame, on=other_param_keys, suffixes=('','_' + str(val)))

    # just keep the joined frame and rename/overwrite columns of interest
    # cleaner might be to delete other val columns but not necessary
    # overwrite train test time correctly
    # df for first value just copy, since it did not get any suffix
    dfs = [joined_frame.copy()]
    for i_value in range(1, len(possible_vals)):
        val = possible_vals[i_value]
        this_df = joined_frame.copy()
        for shared_key in ('train', 'test', 'time'):
            this_df[shared_key] = this_df[shared_key + '_' + str(val)]
        dfs.append(this_df)
    return dfs, possible_vals

def pairwise_compare_frame(df, with_p_vals=False):
    table_vals = []
    table_indices = []
    param_keys = set(df.keys()) - set(['test', 'time', 'train',
        'test_sample', 'train_sample'])
    for key in param_keys:
        if key == 'dataset_filename' or key == 'test_filename' or key == 'subject_id':
            continue
        possible_vals = df[key].unique()
        for i_value_a in range(0, len(possible_vals) - 1):
            for i_value_b in range(i_value_a + 1, len(possible_vals)):
                val_a = possible_vals[i_value_a]
                val_b = possible_vals[i_value_b]
                frame_1 = df[df[key] == val_a]
                frame_2 = df[df[key] == val_b]
                other_param_keys = list(param_keys - set([key]))
                joined_frame = frame_1.merge(frame_2, on=other_param_keys)
                if joined_frame.size == 0:
                    continue
                accuracies_a = np.array(joined_frame.test_x,
                    dtype=np.float64)
                accuracies_b = np.array(joined_frame.test_y,
                    dtype=np.float64)
                mean_a = np.mean(accuracies_a)
                mean_b = np.mean(accuracies_b)
                # Always put better value first in table
                if mean_a >= mean_b:
                    accuracies_1 = accuracies_a
                    accuracies_2 = accuracies_b
                    mean_1 = mean_a 
                    mean_2 = mean_b 
                    val_1 = val_a
                    val_2 = val_b
                else:
                    accuracies_1 = accuracies_b
                    accuracies_2 = accuracies_a
                    mean_1 = mean_b 
                    mean_2 = mean_a 
                    val_1 = val_b
                    val_2 = val_a
                if with_p_vals:
                    if len(accuracies_1) <= 18:
                        diff_perm = perm_mean_diff_test(accuracies_1,
                            accuracies_2) * 100
                    elif len(accuracies_1) <= 62:
                        diff_perm = perm_mean_diff_test(accuracies_1,
                            accuracies_2, n_diffs=2**17) * 100
                    else:
                        _, diff_perm = wilcoxon(accuracies_1,
                            accuracies_2)
                        diff_perm *= 100

                diffs = accuracies_2 - accuracies_1
                diff_std = np.std(diffs)
                diff_mean = np.mean(diffs)
                this_vals = [len(accuracies_1), str(val_1), str(val_2),
                    mean_1, mean_2, diff_mean, diff_std]
                if with_p_vals:
                    this_vals.append(diff_perm)
                table_vals.append(this_vals)
                table_indices.append(key)

    if len(table_vals) == 0:
        return None
    table_vals = np.array(table_vals)
    compare_headers = ['n_exp', 'val_1', 'val_2', 'acc_1', 'acc_2',
                       'diff', 'std']
    if with_p_vals:
        compare_headers.append('p_val')
    compare_frame = pd.DataFrame(table_vals, columns=compare_headers,  
                                 index=(table_indices))
    compare_frame = to_numeric_where_possible(compare_frame)
    compare_frame = round_numeric_columns(compare_frame, 1)
    return compare_frame

def tmean(series):
    """Mean of time and rounding."""
    return pd.Timedelta.round(np.mean(series), 's')

def tstd(series):
    """Std of time and rounding."""
    return pd.Timedelta.round(np.std(series), 's')

def dataset_averaged_frame(data_frame, ignorable_keys=(),
        filename_key=None):
    ignorable_keys = ('test',
        'dataset_filename', 'test_filename', 'time', 'train', 'filename',
        'test_sample', 'train_sample', 'valid') + ignorable_keys
    if filename_key is not None:
        ignorable_keys += (filename_key,)
    
    param_keys = [k for k in data_frame.keys() if k not in ignorable_keys]
    # weird this len(parma_keys)>0 shd always be rue unsure of this
    if len(param_keys) > 0:
        grouped = data_frame.groupby(param_keys)
        # Check for dup
        for name, group in grouped:
            if filename_key is None:
                filename_key = ('filename' if 'filename' in data_frame.keys() 
                    else 'dataset_filename')
            duplicates = group[filename_key][group[filename_key].duplicated()]
            if duplicates.size > 0:
                log.warn("Duplicate filenames:\n{:s}".format(str(duplicates)))
                log.warn("From group {:s}".format(str(name)))
        if  'valid' in data_frame.keys():
            avg_frame = grouped.agg(OrderedDict([('time', [len, tmean, tstd]), 
              ('test', [np.mean, np.std]),
               ('valid', [np.mean, np.std]),
               ('train', [np.mean, np.std]),
               ]))
           
        else: 
            avg_frame = grouped.agg(OrderedDict([('time', [len, tmean, tstd]), 
              ('test', [np.mean, np.std]),
               ('train', [np.mean, np.std]),]))
        # cast from time to int for len
        avg_frame[('time', 'len')] = np.int32(avg_frame[('time', 'len')])
    else:
        # Recreate group result manually for just one group
        avg_frame = pd.DataFrame(columns=pd.MultiIndex(
        levels=[[u'time', u'test', u'train'], 
                [u'len', u'mean', u'std', u'tmean', u'tstd']],
           labels=[[0, 0, 0, 1, 1, 2, 2], [0, 3, 4, 1, 2, 1, 2]]))
        avg_frame[('time', 'len')] = [data_frame.time.size]
        avg_frame[('time', 'tmean')] = [pd.Timedelta.round(
            np.mean(data_frame.time), 's')]
        avg_frame[('time', 'tstd')] = [pd.Timedelta.round(
            np.std(data_frame.time), 's')]
        avg_frame[('test', 'mean')] = [np.mean(data_frame.test)]
        avg_frame[('test', 'std')] = [np.std(data_frame.test)]
        avg_frame[('train', 'mean')] = [np.mean(data_frame.train)]
        avg_frame[('train', 'std')] = [np.std(data_frame.train)]
    avg_frame = round_numeric_columns(avg_frame, 1)
    return avg_frame

def to_numeric_where_possible(df):
    df = df.copy(deep=True)
    for col in df.columns:
        df.loc[:,col] = pd.to_numeric(df.loc[:,col], errors='ignore')
    return df

def round_numeric_columns(df, decimals):
    df = df.copy(deep=True)
    tmp = df.select_dtypes(include=[np.number], exclude=[np.timedelta64])
    df.loc[:, tmp.columns] = np.round(tmp, decimals)
    return df

def round_time_columns(df):
    df = df.copy(deep=True)
    tmp = df.select_dtypes(include=[np.timedelta64])
    df.loc[:, tmp.columns] =  pd.Timedelta.round(tmp, 's')
    return df

def remove_indices_with_same_value(df):
    """Remove indices from a MultiIndex, in case all rows have the same value for that index."""
    df = df.copy(deep=True)
    old_index = df.index
    levels, labels, names = old_index.levels, old_index.labels, old_index.names
    selection_mask = np.array([len(np.unique(l)) > 1 for l in labels])
    new_levels = np.array(levels)[selection_mask]
    new_labels = np.array(labels)[selection_mask]
    new_names = np.array(names)[selection_mask]
    df.index = MultiIndex(levels=new_levels,
               labels=new_labels,
               names=new_names)
    return df

def remove_columns_with_same_value(df, exclude=('train',)):
    cols_multiple_vals = []
    for col in df.columns:
        try:
            has_multiple_vals = len(set(df[col])) > 1
        except TypeError:
            # transform to string in case there are lists
            # since lists not hashable to set
            has_multiple_vals = len(set([str(val) for val in df[col]])) > 1
        cols_multiple_vals.append(has_multiple_vals)
    cols_multiple_vals = np.array(cols_multiple_vals)
    excluded_cols = np.array([c in exclude for c in df.columns])
    df = df.iloc[:,(cols_multiple_vals | excluded_cols)]
    return df

def restrict(df, **params):
    for key, val in params.iteritems():
        df = df[df[key] == val]
    return df

def restrict_or_unset(df, **params):
    for key, val in params.iteritems():
        if key in df.columns:
            if '-' in np.array(df[key]):
                df = df[((df[key]) == val) | (df[key] == '-')]
            else:
                df = df[((df[key]) == val)]
    return df

def restrict_or_missing_col(df, **params):
    for key, val in params.iteritems():
        if key in df.columns:
            df = df[((df[key]) == val)]
    return df

def restrict_if_existing_and_not_unique(df, **params):
    for key, val in params.iteritems():
        if (key in df.columns) and (len(df[key].unique()) > 1):
            df = df[df[key] == val]
    return df


def get_valid_misclass_at_stop(result):
    runtimes_after_first = result.monitor_channels['runtime'][1:]
    i_last_epoch_before_early_stop = np.argmax(np.abs(runtimes_after_first - 
        np.mean(runtimes_after_first))) - 1
    return result.monitor_channels['valid_misclass'][i_last_epoch_before_early_stop]

def add_valid_accuracy_at_stop(df):
    results = load_results_for_df(df)
    misclasses = np.array([get_valid_misclass_at_stop(r) for r in results])
    df['valid_at_stop'] = (1 - misclasses) * 100
    return df

def extract_valid(df):
    results = load_results_for_df(df)
    return np.array([100 * (1 - r.monitor_channels['valid_misclass'][-1]) for r in results])

def extract_train_valid_test_mean(df):
    results = load_results_for_df(df)
    return np.array([
            100 * ( 1 - np.mean((r.monitor_channels['train_misclass'][-1],
                     r.monitor_channels['valid_misclass'][-1],
                     r.monitor_channels['test_misclass'][-1])))
                     for r in results])
def extract_train_valid_mean(df):
    results = load_results_for_df(df)
    return np.array([
            100 * ( 1 - np.mean((r.monitor_channels['train_misclass'][-1],
                     r.monitor_channels['valid_misclass'][-1])))
                     for r in results])

def extract_from_results(df, extract_fn):
    results = load_results_for_df(df)
    return [extract_fn(r) for r in results]
    
def extract_n_epochs_before_early_stop(df):
    results = load_results_for_df(df)
    n_epochs = [len(r.monitor_channels['before_reset_test_misclass']) for r in results]
    return n_epochs

def extract_last_best_epochs(df):
    results = load_results_for_df(df)
    return [extract_last_best_epoch(r) for r in results]

def extract_last_best_epoch(result):
    valid_misclass_til_early_stop = result.monitor_channels['before_reset_valid_misclass']
    n_epochs = len(valid_misclass_til_early_stop)
    # in case of multiple occurences get last one
    best_epoch_from_behind = np.argmin(valid_misclass_til_early_stop[::-1])
    best_epoch = n_epochs - best_epoch_from_behind
    return best_epoch

def extract_n_epochs(df):
    results = load_results_for_df(df)
    n_epochs = [len(r.monitor_channels['test_misclass']) for r in results]
    return n_epochs