from braindecode.results.results import get_final_misclasses, ResultPool,\
    get_training_times
import pandas as pd
import numpy as np
from braindecode.analysis.stats import perm_mean_diff_test
from pandas.core.index import MultiIndex

def remove_dollar(param_val):
    if isinstance(param_val, str) and param_val[0] == '$':
        return param_val[1:]
    else:
        return param_val

def load_data_frame(folder, params):
    res_pool = ResultPool()
    res_pool.load_results(folder, params=params)
    result_objs = res_pool.result_objects()
    varying_params = res_pool.varying_params()
    # remove dollars
    for var_param_dict in varying_params:
        for key, val in var_param_dict.iteritems():
            var_param_dict[key] = remove_dollar(val)
    var_param_keys = varying_params[0].keys()
    var_param_vals = [[v[key] for key in var_param_keys] for v in varying_params]
    var_param_vals = np.array(var_param_vals)

    accuracies = (1-get_final_misclasses(result_objs)) * 100
    training_times = get_training_times(result_objs)
    vals_and_misclasses = np.append(var_param_vals, 
        np.array([training_times, accuracies]).T, 
        axis=1)
    data_frame = pd.DataFrame(vals_and_misclasses, 
        columns=var_param_keys + ['time', 'test'])
    return to_numeric_where_possible(data_frame)

def pairwise_compare_frame(df):
    table_vals = []
    table_indices = []
    param_keys = set(df.keys()) - set(['test'])
    for key in param_keys:
        possible_vals = df[key].unique()
        for i_value_1 in range(0, len(possible_vals) - 1):
            for i_value_2 in range(i_value_1 + 1, len(possible_vals)):
                val_1 = possible_vals[i_value_1]
                val_2 = possible_vals[i_value_2]
                frame_1 = df[df[key] == val_1]
                frame_2 = df[df[key] == val_2]
                other_param_keys = list(param_keys - set([key]))
                joined_frame = frame_1.merge(frame_2, on=other_param_keys)
                if joined_frame.size == 0:
                    continue
                accuracies_1 = np.array(joined_frame.accuracy_x,
                    dtype=np.float64)
                accuracies_2 = np.array(joined_frame.accuracy_y,
                    dtype=np.float64)
                diff_perm = perm_mean_diff_test(accuracies_1, accuracies_2) * 100
                diffs = accuracies_2 - accuracies_1
                diff_std = np.std(diffs)
                diff_mean = np.mean(diffs)
                mean_a = np.mean(accuracies_1)
                mean_b = np.mean(accuracies_2)
                table_vals.append([len(accuracies_1), str(val_1), str(val_2),
                    mean_a, mean_b, diff_mean, diff_std, diff_perm])
                table_indices.append(key)

    table_vals = np.array(table_vals)
    compare_headers = ['n_exp', 'val_1', 'val_2', 'acc_1', 'acc_2',
                       'diff', 'std', 'perm']
    compare_frame = pd.DataFrame(table_vals, columns=compare_headers,  
                                 index=(table_indices))
    compare_frame = to_numeric_where_possible(compare_frame)
    compare_frame = round_numeric_columns(compare_frame, 1)
    return compare_frame

def to_numeric_where_possible(df):
    df = df.copy(deep=True)
    for col in df.columns:
        df.loc[:,col] = pd.to_numeric(df.loc[:,col], errors='ignore')
    return df

def round_numeric_columns(df, decimals):
    df = df.copy(deep=True)
    tmp = df.select_dtypes(include=[np.number])
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