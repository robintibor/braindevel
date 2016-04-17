from braindecode.results.results import get_final_misclasses, ResultPool,\
    get_training_times
import pandas as pd
import numpy as np
from braindecode.analysis.stats import perm_mean_diff_test
from pandas.core.index import MultiIndex
from braindecode.scripts.print_results import prettify_word
from collections import OrderedDict
import logging
log = logging.getLogger(__name__)

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
    data_frame = to_data_frame(result_objs, varying_params,
        shorten_headers=shorten_headers)
    return data_frame

def to_data_frame(result_objs, varying_params, shorten_headers=True):
    # remove dollars
    for var_param_dict in varying_params:
        for key, val in var_param_dict.iteritems():
            var_param_dict[key] = remove_dollar(val)
    
    var_param_keys = varying_params[0].keys()
    var_param_vals = [[v[key] for key in var_param_keys] for v in varying_params]
    var_param_vals = np.array(var_param_vals)
    test_accs = (1 - get_final_misclasses(result_objs, 'test')) * 100
    train_accs = (1 - get_final_misclasses(result_objs, 'train')) * 100
    training_times = get_training_times(result_objs)
    vals_and_misclasses = np.append(var_param_vals, 
        np.array([training_times, test_accs, train_accs]).T, 
        axis=1)
    if shorten_headers:
        var_param_keys = [prettify_word(key) for key in var_param_keys]
    data_frame = pd.DataFrame(vals_and_misclasses, 
        columns=var_param_keys + ['time', 'test', 'train'])
    data_frame = to_numeric_where_possible(data_frame)
    data_frame.time = pd.to_timedelta(np.round(data_frame.time), unit='s')
    return data_frame

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

def dataset_averaged_frame(data_frame):
    param_keys = [k for k in data_frame.keys() if k not in ['test', 'dataset_filename', 'test_filename', 'time',
                                                       'train', 'filename']]
    grouped = data_frame.groupby(param_keys)
    # Check for dup
    for name, group in grouped:
        duplicates = group.filename[group.filename.duplicated()]
        if duplicates.size > 0:
            log.warn("Duplicate filenames : {:s}".format(str(duplicates)))
            log.warn("From group {:s}".format(str(name)))
    averaged_frame = grouped.agg(OrderedDict([('time', [len, tmean]), 
          ('test', [np.mean, np.std]),
           ('train', [np.mean, np.std]),]))
    averaged_frame = round_numeric_columns(averaged_frame, 1)
    return averaged_frame

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

def remove_columns_with_same_value(df):
    wanted_cols = np.array([len(np.unique(df[c])) > 1 for c in df.columns])

    df = df.iloc[:,wanted_cols]
    return df