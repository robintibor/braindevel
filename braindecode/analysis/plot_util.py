import numpy as np
from matplotlib import  pyplot as plt
from matplotlib import cm
from braindecode.datasets.sensor_positions import (get_C_sensors_sorted,
    get_sensor_pos, tight_C_positions, cap_positions)
from braindecode.results.results import (
    DatasetAveragedResults, compute_confusion_matrix, get_padded_chan_vals)
from copy import deepcopy
from pylearn2.utils import serial
import os.path
from matplotlib import gridspec
import seaborn
from braindecode.analysis.pandas_util import load_results_for_df,\
    extract_from_results, get_dfs_for_matched_exps_with_different_vals
from numpy.random import RandomState

def plot_loss_mean_std_for_exps_with_tube(df):
    test_losses = extract_from_results(df, lambda r: r.monitor_channels['test_loss'])
    test_losses, exps_by_epoch = get_padded_chan_vals(test_losses, pad_by=np.nan)

    valid_losses = extract_from_results(df, lambda r: r.monitor_channels['valid_loss'])
    valid_losses, exps_by_epoch = get_padded_chan_vals(valid_losses, pad_by=np.nan)
    train_losses = extract_from_results(df, lambda r: r.monitor_channels['train_loss'])
    train_losses, exps_by_epoch = get_padded_chan_vals(train_losses, pad_by=np.nan)
    plot_with_tube(range(train_losses.shape[1]),np.nanmean(train_losses, axis=0),
          np.nanstd(train_losses, axis=0))
    plot_with_tube(range(valid_losses.shape[1]),np.nanmean(valid_losses, axis=0),
              np.nanstd(valid_losses, axis=0), color=seaborn.color_palette()[1])
    plot_with_tube(range(test_losses.shape[1]),np.nanmean(test_losses, axis=0),
              np.nanstd(test_losses, axis=0), color=seaborn.color_palette()[2])
    plt.plot(exps_by_epoch / float(exps_by_epoch[0]), color='black', linestyle='dashed')
    

def plot_misclasses_mean_std_for_exps_with_tube(df):
    test_misclasses = extract_from_results(df, lambda r: r.monitor_channels['test_misclass'])
    test_misclasses, exps_by_epoch = get_padded_chan_vals(test_misclasses, pad_by=np.nan)

    valid_misclasses = extract_from_results(df, lambda r: r.monitor_channels['valid_misclass'])
    valid_misclasses, exps_by_epoch = get_padded_chan_vals(valid_misclasses, pad_by=np.nan)
    train_misclasses = extract_from_results(df, lambda r: r.monitor_channels['train_misclass'])
    train_misclasses, exps_by_epoch = get_padded_chan_vals(train_misclasses, pad_by=np.nan)
    plot_with_tube(range(train_misclasses.shape[1]),np.nanmean(train_misclasses, axis=0),
          np.nanstd(train_misclasses, axis=0))
    plot_with_tube(range(valid_misclasses.shape[1]),np.nanmean(valid_misclasses, axis=0),
              np.nanstd(valid_misclasses, axis=0), color=seaborn.color_palette()[1])
    plot_with_tube(range(test_misclasses.shape[1]),np.nanmean(test_misclasses, axis=0),
              np.nanstd(test_misclasses, axis=0), color=seaborn.color_palette()[2])
    plt.plot(exps_by_epoch / float(exps_by_epoch[0]), color='black', linestyle='dashed')
    

def plot_all_matched_vals(df):
    param_keys = set(df.keys()) - set(['test', 'time', 'train',
        'test_sample', 'train_sample'])
    for key in param_keys:
        if len(df[key].unique()) > 1:
            dfs, unique_vals = get_dfs_for_matched_exps_with_different_vals(
                df, key)
            if len(dfs[0]) > 0:
                plt.figure()
                plot_per_sub_unique_vals(df, key, matched=True)
                plt.title(key, fontsize=12)
    
def plot_dfs_test(dfs):
    plot_dfs_vals(dfs, values_fn=lambda df: df.test)

def plot_dfs_vals(dfs, values_fn=lambda df: df.test):
    '''
    
    :param dfs: 
    :type dfs: Pandas Dataframes
    :param values_fn: Function to extract values (default extract test)
    '''
    if values_fn == 'time':
        # in mintues (nanoseconds to minutes)
        values_fn=lambda df: df.time / (1.0e9 * 60.0)
    rng = RandomState(3483948)
    for i_df, this_df in enumerate(dfs):
        vals = values_fn(this_df)
        plt.plot(i_df + rng.randn(len(vals)) * 0.05, vals, linestyle='None', marker='o',
                alpha=0.5)

def plot_per_sub_dfs_test(dfs):
    plot_per_sub_dfs(dfs, values_fn=lambda df: df.test)

def plot_per_sub_dfs(dfs, values_fn):
    per_sub_dfs = get_per_sub_dfs(dfs)
    cp = seaborn.color_palette()
    with seaborn.color_palette(np.repeat(cp,len(dfs),axis=0)):
        plot_dfs_vals(per_sub_dfs, values_fn=values_fn)
    subject_legend_mrks = [plt.Line2D((0,1),(0,0),
                                  color=seaborn.color_palette()[i_color], marker='o', linestyle='Null')
                       for i_color in (0,1,2)]
    
    plt.legend(subject_legend_mrks, ("Subj 1","Subj 2","Subj 3"))
    

def get_per_sub_dfs(dfs):
    # shd be (subject1, firstdf), (subject1, seconddf), ... (subject2, firstdf),...
    return [d[d.subject_id == sid] for sid in (1,2,3) for d in dfs ]
    
def plot_per_sub_unique_vals(df, col_name, values_fn=lambda df: df.test,
        matched=False):
    assert len(df) > 0
    if matched:
        dfs, unique_vals = get_dfs_for_matched_exps_with_different_vals(df,
            col_name)
    else:
        unique_vals = df[col_name].unique()
        dfs = [df[df[col_name] == val] for val in unique_vals]
    plot_per_sub_dfs(dfs, values_fn=values_fn)
    plt.xticks(range(len(dfs) * 3), np.tile(unique_vals,3), rotation=30, ha='right')

def show_misclass_scatter_plot(first_misclasses, second_misclasses, figsize=(4,4)):
    fig = plt.figure(figsize=figsize)
    first_acc = (1-first_misclasses) * 100
    second_acc = (1-second_misclasses) * 100
    plt.plot(first_acc, second_acc, 'o', markersize=5)
    plt.plot([0,100],[0,100], 'k:')
    plt.ylim(20,100)
    plt.xlim(20,100)
    return fig

def plot_heatmap(trial, relevances, sensor_names, sensor_map, figsize=(14, 10)):
    fig = plot_head_signals_tight(trial,
                                  sensor_names,
                                  sensor_map=sensor_map,
                                  figsize=figsize)
    vmin = np.min(relevances)
    vmax = np.max(relevances)
    for i_chan, ax in enumerate(fig.axes):
        chan_relevance = relevances[i_chan].squeeze()
        plotlim = [ax.get_xlim()[0] - 0.5, ax.get_xlim()[1] + 0.5] + list(ax.get_ylim())
        ax.imshow([chan_relevance], cmap=cm.Reds, interpolation='nearest',
            extent=plotlim, aspect='auto',
                 vmin=vmin, vmax=vmax)
    return fig

def plot_mean_and_std(data, axis=0, color=None):
    if color is None:
        color = seaborn.color_palette()[0]
    std = np.std(data, axis=axis)
    mean = np.mean(data, axis=axis)
    plt.plot(mean, color=color)
    ax = plt.gca()
    ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2, color=color)

def plot_with_tube(x,y,deviation, axis=0, color=None):
    if color is None:
        color = seaborn.color_palette()[0]
    plt.plot(x,y, color=color)
    ax = plt.gca()
    ax.fill_between(x, y - deviation, y + deviation, alpha=0.2, color=color)

def plot_mean_std_misclasses_over_time(misclasses):
    padded_misclasses, _ = get_padded_chan_vals(misclasses['train'])
    plot_mean_and_std(padded_misclasses, color=seaborn.color_palette()[0])
    padded_misclasses, _ = get_padded_chan_vals(misclasses['valid'])
    plot_mean_and_std(padded_misclasses, color=seaborn.color_palette()[1])
    padded_misclasses, n_exps_by_epoch = get_padded_chan_vals(misclasses['test'])
    plot_mean_and_std(padded_misclasses, color=seaborn.color_palette()[2])
    plt.plot(n_exps_by_epoch / float(n_exps_by_epoch[0]), color='black', lw=1)
    plt.ylim(0,1)
    
def plot_misclasses_over_time(misclasses, alpha=1, lw=0.75):
    for single_misclass in misclasses['train']:
        plt.plot(single_misclass, color=seaborn.color_palette()[0], alpha=alpha, lw=lw)
    for single_misclass in misclasses['valid']:
        plt.plot(single_misclass, color=seaborn.color_palette()[1], alpha=alpha, lw=lw)
    for single_misclass in misclasses['test']:
        plt.plot(single_misclass, color=seaborn.color_palette()[2], alpha=alpha, lw=lw)
    
def plot_multiple_head_signals_tight(signals, sensor_names=None, 
    figsize=(12, 7), plot_args=None, hspace=0.35, sensor_map=tight_C_positions,
        tsplot=False):
    reshaped_signals=np.array(signals).transpose(1,2,0)
    return plot_head_signals_tight(reshaped_signals, sensor_names, figsize, 
        plot_args, hspace, sensor_map, tsplot)
    

def plot_head_signals(signals, sensor_names=None, figsize=(12, 7),
    plot_args=None):
    assert sensor_names is None or len(signals) == len(sensor_names), ("need "
        "sensor names for all sensor matrices")
    if sensor_names is None:
        sensor_names = map(str, range(len(signals)))
    if plot_args is None:
        plot_args = dict()
    figure = plt.figure(figsize=figsize)
    sensor_positions = [get_sensor_pos(name) for name in sensor_names]
    sensor_positions = np.array(sensor_positions)  # sensors x 2(row and col)
    maxima = np.max(sensor_positions, axis=0)
    minima = np.min(sensor_positions, axis=0)
    max_row = maxima[0]
    max_col = maxima[1]
    min_row = minima[0]
    min_col = minima[1]
    rows = max_row - min_row + 1
    cols = max_col - min_col + 1
    first_ax = None
    for i in xrange(0, len(signals)):
        sensor_name = sensor_names[i]
        sensor_pos = sensor_positions[i]
        assert np.all(sensor_pos == get_sensor_pos(sensor_name))
        # Transform to flat sensor pos
        row = sensor_pos[0]
        col = sensor_pos[1]
        subplot_ind = (row - min_row) * cols + col - min_col + 1  # +1 as matlab uses based indexing
        if first_ax is None:
            ax = figure.add_subplot(rows, cols, subplot_ind)
            first_ax = ax
        else:
            ax = figure.add_subplot(rows, cols, subplot_ind, sharey=first_ax,
                sharex=first_ax)
        signal = signals[i]
        ax.plot(signal, **plot_args)
        ax.set_title(sensor_name)
        ax.set_yticks([])
        if len(signal) == 600:
            ax.set_xticks([150, 300, 450])
            ax.set_xticklabels([])
        ax.xaxis.grid(True)
        # make line at zero
        ax.axhline(y=0, ls=':', color="grey")
    return figure

def plot_head_signals_tight_with_tube(signals, deviation,
    sensor_names=None, figsize=(12, 7),
        plot_args=None, hspace=0.35, sensor_map=tight_C_positions,
        tsplot=False, color=None):
    if color is None:
        color = seaborn.color_palette()[0]
    fig = plot_head_signals_tight(signals, sensor_names, 
        figsize=figsize,plot_args=plot_args, hspace=hspace,
        sensor_map=sensor_map, tsplot=tsplot)
    for i, ax in enumerate(fig.axes):
        ax.fill_between(range(signals.shape[1]),
            signals[i].squeeze() - deviation[i].squeeze(),  
            signals[i].squeeze() + deviation[i].squeeze(),  
            alpha=0.2, color=color)
    return fig
    

def plot_head_signals_tight(signals, sensor_names=None, figsize=(12, 7),
        plot_args=None, hspace=0.35, sensor_map=tight_C_positions,
        tsplot=False):
    assert sensor_names is None or len(signals) == len(sensor_names), ("need "
        "sensor names for all sensor matrices")
    assert sensor_names is not None
    if plot_args is None:
        plot_args = dict()
    figure = plt.figure(figsize=figsize)
    sensor_positions = [get_sensor_pos(name, sensor_map) for name in sensor_names]
    sensor_positions = np.array(sensor_positions)  # sensors x 2(row and col)
    maxima = np.max(sensor_positions, axis=0)
    minima = np.min(sensor_positions, axis=0)
    max_row = maxima[0]
    max_col = maxima[1]
    min_row = minima[0]
    min_col = minima[1]
    rows = max_row - min_row + 1
    cols = max_col - min_col + 1
    first_ax = None
    for i in xrange(0, len(signals)):
        sensor_name = sensor_names[i]
        sensor_pos = sensor_positions[i]
        assert np.all(sensor_pos == get_sensor_pos(sensor_name, sensor_map))
        # Transform to flat sensor pos
        row = sensor_pos[0]
        col = sensor_pos[1]
        subplot_ind = (row - min_row) * cols + col - min_col + 1  # +1 as matlab uses based indexing
        if first_ax is None:
            ax = figure.add_subplot(rows, cols, subplot_ind)
            first_ax = ax
        else:
            ax = figure.add_subplot(rows, cols, subplot_ind, sharey=first_ax,
                sharex=first_ax)
        signal = signals[i]
        if tsplot is False:
            ax.plot(signal, **plot_args)
        else:
            seaborn.tsplot(signal.T, ax=ax, **plot_args)
        ax.set_title(sensor_name)
        ax.set_yticks([])
        if len(signal) == 600:
            ax.set_xticks([150, 300, 450])
            ax.set_xticklabels([])
        else:
            ax.set_xticks([])
            
            
        ax.xaxis.grid(True)
        # make line at zero
        ax.axhline(y=0, ls=':', color="grey")
        figure.subplots_adjust(hspace=hspace)
    return figure

def plot_head_signals_tight_two_signals(signals1, signals2,
    sensor_names=None, figsize=(10, 8), plot_args=None):
    assert len(signals1) == len(signals2)
    assert sensor_names is not None
    both_signals = [signals1, signals2]
    return plot_head_signals_tight_multiple_signals(both_signals,
        sensor_names=sensor_names, figsize=figsize,
        plot_args=plot_args)

def plot_head_signals_tight_multiple_signals(all_signals, sensor_names=None,
    figsize=(10, 8), plot_args=None):
    assert sensor_names is not None
    assert all([len(signals) == len(all_signals[0]) for signals in all_signals])
    if plot_args is None:
        plot_args = dict()
    figure = plt.figure(figsize=figsize)
    sensor_positions = [get_sensor_pos(name, tight_C_positions) for name in sensor_names]
    sensor_positions = np.array(sensor_positions)  # sensors x 2(row and col)
    maxima = np.max(sensor_positions, axis=0)
    minima = np.min(sensor_positions, axis=0)
    max_row = maxima[0, 0]
    max_col = maxima[1, 0]
    min_row = minima[0, 0]
    min_col = minima[1, 0]
    rows = max_row - min_row + 1
    cols = max_col - min_col + 1
    first_ax = None
    
    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(rows, cols, wspace=0.3, hspace=0.5)
    
    for i in xrange(0, len(all_signals[0])):
        sensor_name = sensor_names[i]
        sensor_pos = sensor_positions[i]
        assert np.all(sensor_pos == get_sensor_pos(sensor_name, tight_C_positions))
        row = sensor_pos[0]
        col = sensor_pos[1]
        inner_grid = gridspec.GridSpecFromSubplotSpec(len(all_signals), 1,
                subplot_spec=outer_grid[row - min_row, col - min_col], wspace=0.0, hspace=0.0)
        for signal_type in xrange(len(all_signals)):
            signal = all_signals[signal_type][i]
            if first_ax is None:
                ax = plt.Subplot(figure, inner_grid[signal_type, 0])
                first_ax = ax
            else:
                ax = plt.Subplot(figure, inner_grid[signal_type, 0], sharey=first_ax, sharex=first_ax)
            
            if signal_type == 0:
                ax.set_title(sensor_name, fontsize=10)
    
            ax.plot(signal, **plot_args)
            ax.xaxis.grid(True)
            # make line at zero
            ax.axhline(y=0, ls=':', color="grey")
            figure.add_subplot(ax)
        
        
    if len(signal) == 600:
        plt.xticks([150, 300, 450], [])
    else:
        plt.xticks([])
    
    plt.yticks([])
    return figure
        
def plot_sensor_signals(signals, sensor_names=None, figsize=None,
        yticks=None, plot_args=None, sharey=True, highlight_zero_line=True,
        xvals=None, fontsize=9, x_ticks_x_offset=-0.035):
    """
    Plot signals of all sensors below each other, one row per sensor.
    
    Parameters
    ----------
    yticks: 1d-array or str or None
        Which yticks to show, either supply yticks or "minmax", "keep", "onlymax"
        or None (Default= None, means no ticks)
    """
    assert sensor_names is None or len(signals) == len(sensor_names), ("need "
        "sensor names for all sensor matrices")
    if sensor_names is None:
        sensor_names = map(str, range(len(signals)))  
    num_sensors = signals.shape[0]
    if plot_args is None:
        plot_args = dict()
    if figsize is None:
        figsize = (7, np.maximum(num_sensors // 4, 1))
    figure, axes = plt.subplots(num_sensors, sharex=True, sharey=sharey,
        figsize=figsize)
    for sensor_i in xrange(num_sensors):
        if num_sensors > 1:
            ax = axes[sensor_i]
        else:
            ax = axes
        if xvals is None:
            ax.plot(signals[sensor_i], **plot_args)
        else:
            ax.plot(xvals, signals[sensor_i], **plot_args)
        if yticks is None:
            ax.set_yticks([])
        elif (isinstance(yticks, list)): 
            ax.set_yticks(yticks)
        elif yticks == "minmax":
            ymin, ymax = ax.get_ylim()
            ax.set_yticks((ymin, ymax - ymax / 10.0))
        elif yticks == "onlymax":
            ymin, ymax = ax.get_ylim()
            ax.set_yticks([ymax])
        elif yticks == "keep": 
            pass
        else:
            raise ValueError("Unknown yticks value {:s}".format(str(yticks)))
        ax.text(x_ticks_x_offset, 0.4, sensor_names[sensor_i], fontsize=fontsize,
            transform=ax.transAxes,
            horizontalalignment='right')
        if (highlight_zero_line):
            # make line at zero
            ax.axhline(y=0, ls=':', color="grey")
    max_ylim = np.max(np.abs(plt.ylim()))
    plt.ylim(-max_ylim, max_ylim)
    figure.subplots_adjust(hspace=0)
    return figure

def plot_misclasses_for_exps(df, **plot_args):
    results = load_results_for_df(df)
    for result in results:
        plot_misclasses_for_result(result, **plot_args)

def plot_losses_for_exps(df, **plot_args):
    results = load_results_for_df(df)
    for result in results:
        plot_loss_for_result(result, **plot_args)

def plot_chan_for_all_exps(df, val):
    results = load_results_for_df(df)
    all_exps_vals = [r.monitor_channels[val] for r in results]
    for exp_vals in all_exps_vals:
        plt.plot(exp_vals)

def plot_misclasses_for_file(result_file_path):
    assert result_file_path.endswith('result.pkl')
    result = serial.load(result_file_path)
    plot_misclasses_for_result(result)
    figure = plt.gcf()
    figure.suptitle("Misclass: " + result_file_path, fontsize=14)
    return figure

def plot_misclasses_loss_for_file(result_file_path):
    assert result_file_path.endswith('result.pkl')
   
    result = serial.load(result_file_path)
    plot_misclass_loss_for_result(result)
    
    figure = plt.gcf()
    figure.suptitle("Misclass/Loss: " + result_file_path)
    return figure

def plot_misclass_loss_for_result(result):
    fig = plt.figure()
    plot_misclasses_for_result(result)
    plot_loss_for_result(result)
    return fig

def plot_loss_for_file(result_file_path, start=None, stop=None):
    assert result_file_path.endswith('result.pkl')
    result = serial.load(result_file_path)
    plot_loss_for_result(result, start, stop)
    figure = plt.gcf()
    figure.suptitle("Loss: " + result_file_path)
    return figure

def plot_misclasses_for_result(result, linewidth=1, **plot_args):
    set_names = ('train', 'valid', 'test')
    for i_set, setname in enumerate(set_names):
        plt.plot(result.monitor_channels['{:s}_misclass'.format(setname)],
                linewidth=linewidth,
                color=seaborn.color_palette()[i_set], **plot_args)
        before_key = 'before_reset_{:s}_misclass'.format(setname)
        if before_key in result.monitor_channels:
            plt.plot(result.monitor_channels[before_key],
                linestyle='--', linewidth=linewidth,
                color=seaborn.color_palette()[i_set], **plot_args)
        plt.legend(plt.gca().get_lines()[0:6:2], set_names, fontsize=12)
        plt.xlabel("Epochs")
        plt.ylabel("Misclass")
        plt.title("Misclass")
        plt.ylim(0,1)

        
def plot_loss_for_result(result, linewidth=1, **plot_args):
    set_names = ('train', 'valid', 'test')
    for i_set, setname in enumerate(set_names):
        plt.plot(result.monitor_channels['{:s}_loss'.format(setname)],
                linewidth=linewidth, color=seaborn.color_palette()[i_set],
                **plot_args)
        before_key = 'before_reset_{:s}_loss'.format(setname)
        if before_key in result.monitor_channels:
            plt.plot(result.monitor_channels[before_key],
                linestyle='--', linewidth=linewidth,
                color=seaborn.color_palette()[i_set], **plot_args)
    plt.legend(plt.gca().get_lines()[0:6:2], set_names, fontsize=12)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")


def add_early_stop_boundary(monitor_channels):
    """Plot early stop boundary as black vertical line into plot
    Determine it by epoch with largest difference in runtime."""
    runtimes_after_first = monitor_channels['runtime'][1:]
    # this might be last epoch after early stop
    # as i saw it
    i_last_epoch_before_early_stop = np.argmax(np.abs(runtimes_after_first - 
        np.mean(runtimes_after_first)))
    plt.axvline(i_last_epoch_before_early_stop, color='black', lw=1)



def plot_train_valid_test_epochs(train, valid, test, figure=None):
    if figure is None:
        figure = plt.figure()
    plt.plot(train)
    plt.plot(valid)
    plt.plot(test)
    plt.legend(('train', 'valid', 'test'))
    return figure


def plot_confusion_matrix_for_averaged_result(result_folder, result_nr):
    """ Plot confusion matrix for averaged dataset result."""
    result_objects = DatasetAveragedResults.load_result_objects_for(
        result_folder, result_nr)
    confusion_mat = compute_confusion_matrix(result_objects)
    plot_confusion_matrix(confusion_mat)

def plot_confusion_matrix_for_result(result_folder, result_nr):
    """ Plot confusion matrix for dataset result with given nr."""
    filename = str(result_nr) + ".result.pkl"
    result_object = serial.load(os.path.join(result_folder, filename))
    confusion_mat = compute_confusion_matrix([result_object])
    plot_confusion_matrix(confusion_mat)    


def plot_confusion_matrix(confusion_mat, class_names=None, figsize=None, colormap=cm.bwr,
        textcolor='black', vmin=None, vmax=None):
    # TODELAY: split into several functions
    # transpose to get confusion matrix same way as matlab
    confusion_mat = confusion_mat.T
    n_classes = confusion_mat.shape[0]
    if class_names is None:
        class_names = [str(i_class + 1) for i_class in xrange(n_classes)]
        
    # norm by number of targets (targets are columns after transpose!)
    #normed_conf_mat = confusion_mat / np.sum(confusion_mat,
    #    axis=0).astype(float)
    # norm by all targets
    normed_conf_mat = confusion_mat / float(np.sum(confusion_mat))
    augmented_conf_mat = deepcopy(normed_conf_mat)
    augmented_conf_mat = np.vstack([augmented_conf_mat, [np.nan] * n_classes])
    augmented_conf_mat = np.hstack([augmented_conf_mat, [[np.nan]] * (n_classes + 1)])
    
    fig = plt.figure(figsize=figsize)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = np.max(normed_conf_mat)
    ax.imshow(np.array(augmented_conf_mat), cmap=colormap,
        interpolation='nearest', alpha=0.6,vmin=vmin, vmax=vmax)
    width = len(confusion_mat)
    height = len(confusion_mat[0])
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate("{:d}\n".format(confusion_mat[x][y]),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=12,
                        color=textcolor,
                        fontweight='bold')
            
            ax.annotate("\n\n{:4.1f}%".format(
                        (confusion_mat[x][y] / float(np.sum(confusion_mat))) * 100),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=10,
                        color=textcolor,
                        fontweight='bold')
    
    # Add values for target correctness etc.
    for x in xrange(width):
        y = len(confusion_mat)
        correctness = confusion_mat[x][x] / float(np.sum(confusion_mat[x, :]))
        ax.annotate("{:5.2f}%".format(correctness * 100),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=12)
        ax.annotate("\n\n\n(correct)",
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=8)
        
    
    for y in xrange(height):
        x = len(confusion_mat)
        correctness = confusion_mat[y][y] / float(np.sum(confusion_mat[:, y]))
        ax.annotate("{:5.2f}%".format(correctness * 100),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=12)
        ax.annotate("\n\n\n(correct)",
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=8)
        
    overall_correctness = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat).astype(float)
    ax.annotate("{:5.2f}%".format(overall_correctness * 100),
                        xy=(len(confusion_mat), len(confusion_mat)),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=12)
    ax.annotate("\n\n\n(correct)",
                    xy=(len(confusion_mat), len(confusion_mat)),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=8)
    
    plt.xticks(range(width), class_names, fontsize=12)
    plt.yticks(range(height), class_names, fontsize=12, rotation=90)
    plt.grid(False)
    plt.ylabel('Predictions', fontsize=15)
    plt.xlabel('Targets', fontsize=15)
    
    return fig

def plot_most_activated_neurons(activations, layers, num_neurons, plotfunction, figsize=(13, 7)):
    sum_per_neuron = np.sum(np.array(activations), axis=0)
    layer_1_sums = sum_per_neuron[1]
    strongest_neurons = np.argsort(layer_1_sums)[::-1][0:num_neurons]
    plotfunction(strongest_neurons, layers, num_neurons, figsize)
    
def plot_most_variant_neurons(activations, layers, num_neurons, plotfunction, figsize=(13, 7)):
    var_per_neuron = np.var(np.array(activations), axis=0)
    layer_1_var = var_per_neuron[1]
    variant_neurons = np.argsort(layer_1_var)[::-1][0:num_neurons]
    plotfunction(variant_neurons, layers, num_neurons, figsize)
    
def plot_only_class_probs(neuron_ids, layers, num_neurons, figsize):
    for neuron_i in neuron_ids:
        plot_class_probs(layers[2].get_weights()[neuron_i])
         
def plot_trials(predictions, dataset, trial_inds, figsize):
    for trial_i in trial_inds:
        trial = dataset.get_topological_view()[trial_i]
        plot_chan_matrices(trial, get_C_sensors_sorted(), figname="Trial %d" % trial_i, figsize=figsize)
        plot_class_probs(predictions[trial_i])
        plot_class_probs(dataset.y[trial_i])
        
def plot_only_preds_and_class_probs(predictions, dataset, trial_inds, figsize):
    for trial_i in trial_inds:
        plot_class_probs(predictions[trial_i])
        plot_class_probs(dataset.y[trial_i])

def plot_correct_part(reconstruction, inputs, sensor_names, **kwargs):
    input_correct = np.ma.masked_where(np.sign(reconstruction) != np.sign(inputs), inputs)
    plot_chan_matrices(input_correct, sensor_names, figname="Reconstruction Correct", **kwargs)

def plot_incorrect_part(reconstruction, inputs, sensor_names, **kwargs):
    input_correct = np.ma.masked_where(np.sign(reconstruction) == np.sign(inputs), inputs)
    plot_chan_matrices(input_correct, sensor_names, figname="Reconstruction Incorrect", **kwargs)

def plot_class_probs(probs, value_minmax=None):
    if value_minmax is None:
        value_minmax = np.max(np.abs(probs))
    fig = plt.figure(figsize=(2, 6))
    plt.imshow(np.atleast_2d(probs), interpolation='nearest', cmap=cm.bwr,
                          origin='lower', vmin=-value_minmax, vmax=value_minmax)
    # hide normal x/y ticks but show some ticks for orientation in case two classes have almost same color
    fig.axes[0].get_xaxis().set_ticklabels([])
    fig.axes[0].get_yaxis().set_ticks([])
    fig.axes[0].get_xaxis().set_ticks([0.5, 1.5, 2.5])
                
def plot_chan_matrices(matrices, sensor_names, figname='', figure=None,
    figsize=(8, 4.5), yticks=(), yticklabels=(),
    correctness_matrices=None, colormap=cm.coolwarm,
    sensor_map=cap_positions, vmax=None, vmin=None,
    share_y_axes=True):
    """ figsize ignored if figure given """
    assert len(matrices) == len(sensor_names), "need sensor names for all sensor matrices"
    if figure is None:
        figure = plt.figure(figsize=figsize)
    sensor_positions = [get_sensor_pos(name, sensor_map) for name in sensor_names]
    sensor_positions = np.array(sensor_positions)  # #sensors x 2(row and col) x1(for some reason:)) 
    maxima = np.max(sensor_positions, axis=0)
    minima = np.min(sensor_positions, axis=0)
    max_row = maxima[0]
    max_col = maxima[1]
    min_row = minima[0]
    min_col = minima[1]
    rows = max_row - min_row + 1
    cols = max_col - min_col + 1
    mean_abs_weight = np.mean(np.abs(matrices))
    if (correctness_matrices is not None):
        mean_abs_weight = np.mean(np.abs(matrices * correctness_matrices))
    if figname != '':
        figure.suptitle(figname, fontsize=14)  # + ", pixel abs mean(x100):  {:.3f}".format(mean_abs_weight * 100),
    
    first_ax = None
    for i in xrange(0, len(matrices)):
        sensor_name = sensor_names[i]
        sensor_pos = sensor_positions[i]
        assert np.all(sensor_pos == get_sensor_pos(sensor_name, sensor_map))
        # Transform to flat sensor pos
        row = sensor_pos[0]
        col = sensor_pos[1]
        subplot_ind = (row - min_row) * cols + col - min_col + 1  # +1 as matlab uses based indexing
        
        if vmin is None:
            vmin = -2 * mean_abs_weight
        if vmax is None:
            vmax = 2 * mean_abs_weight
        if first_ax is None or not share_y_axes:
            ax = figure.add_subplot(rows, cols, subplot_ind)
            first_ax = ax
        elif share_y_axes:
            ax = figure.add_subplot(rows, cols, subplot_ind, sharey=first_ax)
            
        chan_matrix = matrices[i]
        ax.set_title(sensor_name)
        if (correctness_matrices is None):
            # ax.pcolor(chan_matrix.T,
            #     cmap=cm.bwr, #origin='lower', #interpolation='nearest',#aspect='auto'
            #    vmin=-mean_abs_weight * 2, vmax= mean_abs_weight * 2)
            ax.pcolorfast(chan_matrix.T,
                 cmap=colormap,  # origin='lower', #interpolation='nearest',#aspect='auto'
                vmin=vmin, vmax=vmax)
            # ax.imshow(chan_matrix.T,
            #    interpolation='nearest', cmap=cm.bwr, origin='lower', 
            #    vmin=-mean_abs_weight * 2, vmax= mean_abs_weight * 2,
            #    aspect='auto')#"""
        else:
            # Show correct and incorrect inputs with different colors
            # weighted also by degree of correctness
            correctness_mat = correctness_matrices[i]
            ax.imshow(
                np.ma.masked_where(correctness_mat < 0, chan_matrix * correctness_mat).T,
                interpolation='nearest', cmap=cm.bwr, origin='lower',
                vmin=vmin, vmax=vmax)
            # use yellow/brown for red(positive) incorrect values
            # (mask out correct or blue=negative values)
            # also take minus the values as otherwise they go from
            # minus-something to 0, that makes colormaps 
            # more complicated to use :) 
            ax.imshow(
                np.ma.masked_where(
                    np.logical_or(correctness_mat > 0, chan_matrix < 0),
                    - (chan_matrix * correctness_mat)).T,
                interpolation='nearest', cmap=cm.YlOrBr, origin='lower',
                vmin=0, vmax=vmax)
            # use purple for blue(negative) incorrect values
            ax.imshow(
                np.ma.masked_where(
                    np.logical_or(correctness_mat > 0, chan_matrix >= 0),
                    chan_matrix * correctness_mat).T,
                interpolation='nearest', cmap=cm.PuRd, origin='lower',
                vmin=0, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(color='k', linewidth=0.1, linestyle=':')
    return figure
