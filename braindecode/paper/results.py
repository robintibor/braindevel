import numpy as np
import pandas as pd
from braindecode.analysis.pandas_util import restrict, restrict_or_unset,\
    round_numeric_columns
from braindecode.paper import unclean_sets
from braindecode.analysis.stats import perm_mean_diff_test, wilcoxon_signed_rank,\
    sign_test
def clean_datasets(df):
    for name in unclean_sets:
        df = df[np.logical_not(df.dataset_filename.str.contains(name))]
    return df

## Main comparison
def elu_deep_5(df):
    return df[(df.first_nonlin == 'elu') & 
        (df.first_pool_nonlin == 'identity') &
        (df.first_pool_mode == 'max')]
def square_shallow(df):
    return df[(df.first_nonlin == 'square') & 
        (df.post_pool_nonlin == 'safe_log') &
        (df.pool_mode == 'average_exc_pad')]

def above_0(df):
    if 38 in df.high_cut_hz.values:
        df = df[(df.high_cut_hz == 38) & (df.low_cut_hz == 0)]
    else:
        df = df[(df.high_cut_hz == 'null') & (df.low_cut_hz == 0)]
    return df

def above_4(df):
    return df[df.low_cut_hz == 4]

def from_0_to_4(df):
    return df[(df.high_cut_hz == 4) & 
             (df.low_cut_hz == 0)]
    
def deep_5_default(df):
    return df[(df.num_filters_4 == 200) & (df.filter_time_length == 10)]

def csp_above_0(df):
    return df[(df.min_freq == 1) & ((df.max_freq == 34) | (df.max_freq == 118))]

def csp_above_4(df):
    df = df[df.min_freq == 7]
    if 'trial_stop' in df.columns and 'max_freq' in df.columns:
        df = df[(df.trial_stop == 4000) & ((df.max_freq == 34) | (df.max_freq == 118))]
    return df

def csp_0_to_4(df):
    return df[df.max_freq == 3.5]

def tied_loss(df):
    return df[(df.loss_expression == 'tied_neighbours')]

def deep_5_new_default(df):
    df = deep_5_default(df)
    df = df[(df.layers == 'deep_5') & (df.batch_norm == True) &
         (df.layer_names_to_norms == 'layer_names_to_norms') &
         (df.drop_prob == 0.5) & 
         (df.split_first_layer == True)] # | df.split_first_layer == '-')?
    return df

def shallow_cnt_default(df):
    return df[(df.resample_fs ==250) &
         (df.final_dense_length == 30) &
         (df.layers == 'cnt_shallow_square')]

def deep5_cnt_main_comp(df):
    return tied_loss(deep5_main_comp(df))

def deep5_main_comp(df):
    df = restrict_or_unset(df, filter_length_4=10,
                           num_filters_time=25,
                           double_time_convs=False, split_first_layer=True,)
    df = restrict_or_unset(df, batch_modifier='null', first_nonlin='elu',
                          later_nonlin='elu', batch_norm=True)
    df = restrict_or_unset(df, drop_p=0.5, drop_prob=0.5)
    return df
    
def shallow_cnt_main_comp(df):
    df = shallow_main_comp(df)
    df = restrict(df, loss_expression='tied_neighbours')
    return df

def shallow_main_comp(df):
    df = restrict(df, layers='cnt_shallow_square', first_nonlin='square', post_pool_nonlin='safe_log', 
                  pool_mode='average_exc_pad')
    df = restrict_or_unset(df, drop_p=0.5, batch_modifier='null',)
    return df

def main_comp_csp(df):
    return restrict(restrict_or_unset(df, standardize=False, standardize_filt_cnt=False,
                  standardize_epo=False, standardize_cnt=False),
                  low_bound=0)
    
## Without advances experiment
def past(df):
    if 'cnt_shallow_square_no_bnorm' in df.layers.unique():
        df = df[df.layers == 'cnt_shallow_square_no_bnorm']
        df = df[df.drop_p == 0.]
    else:
        df = relu_deep_5(df)
        df = df[(df.layers == 'deep_5') & (df.batch_norm == False) &
         (df.drop_prob == 0.)]
    
    return df

def relu_deep_5(df):
    return df[(df.first_nonlin == 'relu') & 
        (df.first_pool_nonlin == 'identity') &
        (df.first_pool_mode == 'max')]

## Modification experiments

def square_mean_first(df):
    df = df[(df.first_nonlin == 'square') & (df.later_nonlin == 'elu') &
         (df.first_pool_mode == 'average_exc_pad') &
         (df.first_pool_nonlin == 'safe_log')]
    if len(df.layers.unique()) > 1:
        df = df[df.layers=='deep_5']
    return df

def square_max_first(df):
    df = df[(df.first_nonlin == 'square') & (df.later_nonlin == 'elu') &
         (df.first_pool_mode == 'max') &
         (df.first_pool_nonlin == 'safe_log')]
    if len(df.layers.unique()) > 1:
        df = df[df.layers=='deep_5']
    return df

def no_drop(df):
    if 'drop_prob' in df.columns:
        df = df[(df.drop_prob == 0)]
    else:
        df = df[(df.drop_p == 0)]
    return df

def no_bnorm(df):
    if 'batch_norm' in df.columns:
        df = df[df.batch_norm == False]
    else:
        df = df[df.layers == 'cnt_shallow_square_no_bnorm']
    return df

def yes_bnorm(df):
    if 'batch_norm' in df.columns:
        df = df[df.batch_norm == True]
    else:
        df = df[df.layers == 'cnt_shallow_square']
    return df

def yes_drop(df):
    if 'drop_prob' in df.columns:
        df = df[(df.drop_prob == 0.5)]
    if 'drop_p' in df.columns:
        if '-' in df.drop_p.tolist():
            df = df[(df.drop_p == 0.5) | (df.drop_p == '-')]
        else:
            df = df[(df.drop_p == 0.5)]
    return df

def no_tied_loss(df):
    df = df[df.loss_expression == 'categorical_crossentropy']
    df = df[df.trial_start == 1500]
    if hasattr(df, 'split_first_layer') and len(df.split_first_layer.unique()) > 1:
        df = df[df.split_first_layer == True]
    return df

def no_split_first_layer(df):
    if 'split_first_layer' in df.columns:
        df =  df[df.split_first_layer == False]
    else:
        df = df[df.layers == 'cnt_shallow_merged']
    if 'tag' in df.columns and len(df.tag.unique()) > 1:
        df = df[df.tag == 'after_improvement']
    return df

def double_time_convs(df):
    df = df[(df.double_time_convs == True) & 
             (df.filter_length_4 == 6)]
    if len(df.split_first_layer.unique()) > 1:
        df = df[df.split_first_layer == True]
    return df

def single_time_convs(df):
    df = df[(df.double_time_convs == False)]
    return df

def square_mean_sqrt(df):
    return df[(df.post_pool_nonlin == 'safe_sqrt') &
         (df.first_nonlin == 'square')]

def elu_max_shallow(df):
    df = df[(df.first_nonlin == 'elu') &
         (df.pool_mode == 'max')]
    if len(df.drop_p.unique()) > 1:
        df = df[df.drop_p == 0.5]
    return df

def elu_mean_shallow(df):
    df = df[(df.first_nonlin == 'elu') &
         (df.pool_mode == 'average_exc_pad')]
    if len(df.drop_p.unique()) > 1:
        df = df[df.drop_p == 0.5]
    return df

def relu_nonlins(df):
    df = df[(df.first_nonlin == 'relu') & (df.later_nonlin == 'relu')]
    return df
def elu_nonlins(df):
    df = df[(df.first_nonlin == 'elu') & (df.later_nonlin == 'elu')]
    return df

def split_first_layer(df):
    return df[df.split_first_layer == True]

def compare_net_csp(df_net, df_csp, name,freq, dataset, with_csp_acc=False, 
        with_std=False, with_std_error=False, max_n_p_vals=20):
    assert len(df_net) == len(df_csp), (
        "Net ({:d}) and csp ({:d}) should have same length".format(
            len(df_net), len(df_csp)))
    df_merged = df_net.merge(df_csp, on='dataset_filename', suffixes=('_net','_csp'))
    # not really necessary to sort, just to make sure 
    df_merged = df_merged.sort_values(by='dataset_filename')

    test_acc_net = np.array(df_merged['test_net'])
    test_acc_csp = np.array(df_merged['test_csp'])
    if len(test_acc_net) > max_n_p_vals:
        p_val = perm_mean_diff_test(test_acc_net,test_acc_csp, n_diffs=2**max_n_p_vals)
    else:
        p_val = perm_mean_diff_test(test_acc_net,test_acc_csp, n_diffs=None)
    p_val_wilc = wilcoxon_signed_rank(test_acc_net, test_acc_csp)
    p_val_sign = sign_test(test_acc_net, test_acc_csp)
    diff_std = np.std(test_acc_net - test_acc_csp)

    df_out = pd.DataFrame()

    df_out['name'] = [name]
    df_out['freq'] = [freq]
    df_out['dataset'] = [dataset]
    if with_csp_acc:
        df_out['test_csp'] = [np.mean(test_acc_csp)]
        
    df_out['test_net'] = [np.mean(test_acc_net)]
    df_out['diff'] = [np.mean(test_acc_net) - np.mean(test_acc_csp)]
    if with_std:
        df_out['std'] = [diff_std]
    if with_std_error:
        df_out['stderr'] = [diff_std / np.sqrt(len(test_acc_net))]
    df_out = round_numeric_columns(df_out,1)
        
    
        
    df_out['rand'] = [p_val]
    df_out['wilc'] = [p_val_wilc]
    df_out['sign'] = [p_val_sign]
    df_out['time_net'] = [pd.Timedelta.round(np.mean(df_net.time), 's')]

    assert len(df_merged) == len(df_csp), (
        "Merged ({:d}) and csp ({:d}) should have same length".format(
            len(df_merged), len(df_csp)))
    return df_out