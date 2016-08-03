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
log = logging.getLogger(__name__)

class MetaDataFrame(pd.DataFrame):
    # from https://github.com/pydata/pandas/issues/2485#issuecomment-174577149
    _metadata = ['attrs']

    @property
    def _constructor(self):
        return MetaDataFrame

    def _combine_const(self, other, *args, **kwargs):
        return super(MetaDataFrame, self)._combine_const(other, *args, **kwargs).__finalize__(self)

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
    
def pairwise_compare_frame(df, with_p_vals=False):
    table_vals = []
    table_indices = []
    param_keys = set(df.keys()) - set(['test', 'time', 'train'])
    for key in param_keys:
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
                    if len(accuracies_1) < 19:
                        diff_perm = perm_mean_diff_test(accuracies_1,
                            accuracies_2) * 100
                    else:
                        diff_perm = perm_mean_diff_test(accuracies_1,
                            accuracies_2, n_diffs=2**18) * 100

                diffs = accuracies_2 - accuracies_1
                diff_std = np.std(diffs)
                diff_mean = np.mean(diffs)
                this_vals = [len(accuracies_1), str(val_1), str(val_2),
                    mean_1, mean_2, diff_mean, diff_std]
                if with_p_vals:
                    this_vals.append(diff_perm)
                table_vals.append(this_vals)
                table_indices.append(key)

    table_vals = np.array(table_vals)
    compare_headers = ['n_exp', 'val_1', 'val_2', 'acc_1', 'acc_2',
                       'diff', 'std']
    if with_p_vals:
        compare_headers.append('perm')
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

def dataset_averaged_frame(data_frame):
    param_keys = [k for k in data_frame.keys() if k not in ['test',
        'dataset_filename', 'test_filename', 'time', 'train', 'filename',
        'test_sample', 'train_sample']]
    if len(param_keys) > 0:
        grouped = data_frame.groupby(param_keys)
        # Check for dup
        for name, group in grouped:
            filename_key = ('filename' if 'filename' in data_frame.keys() 
                else 'dataset_filename')
            duplicates = group[filename_key][group[filename_key].duplicated()]
            if duplicates.size > 0:
                log.warn("Duplicate filenames:\n{:s}".format(str(duplicates)))
                log.warn("From group {:s}".format(str(name)))
        avg_frame = grouped.agg(OrderedDict([('time', [len, tmean, tstd]), 
              ('test', [np.mean, np.std]),
               ('train', [np.mean, np.std]),]))
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
