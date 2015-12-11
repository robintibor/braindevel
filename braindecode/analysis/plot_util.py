import numpy as np
from matplotlib import  pyplot
from matplotlib import cm
from braindecode.datasets.sensor_positions import (get_C_sensors_sorted, 
    get_sensor_pos, tight_C_positions, cap_positions)
from braindecode.results.results import (
    DatasetAveragedResults, compute_confusion_matrix)
from copy import deepcopy
from pylearn2.utils import serial
import os.path
from matplotlib import gridspec
import seaborn

def plot_head_signals(signals, sensor_names=None, figsize=(12,7), 
    plot_args=None):
    
    assert sensor_names is None or len(signals) == len(sensor_names), ("need "
        "sensor names for all sensor matrices")
    if sensor_names is None:
        sensor_names = map(str, range(len(signals)))
    if plot_args is None:
        plot_args = dict()
    figure = pyplot.figure(figsize=figsize)
    sensor_positions = [get_sensor_pos(name) for name in sensor_names]
    sensor_positions = np.array(sensor_positions) #sensors x 2(row and col)
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
        subplot_ind = (row - min_row) * cols + col - min_col + 1 # +1 as matlab uses based indexing
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
            ax.set_xticks([150,300,450])
            ax.set_xticklabels([])
        ax.xaxis.grid(True)
        # make line at zero
        ax.axhline(y=0,ls=':', color="grey")
    return figure

def plot_head_signals_tight(signals, sensor_names=None, figsize=(12,7),
        plot_args=None, hspace=0.35, sensor_map=tight_C_positions,
        tsplot=False):
    assert sensor_names is None or len(signals) == len(sensor_names), ("need "
        "sensor names for all sensor matrices")
    assert sensor_names is not None
    if plot_args is None:
        plot_args = dict()
    figure = pyplot.figure(figsize=figsize)
    sensor_positions = [get_sensor_pos(name, sensor_map) for name in sensor_names]
    sensor_positions = np.array(sensor_positions) #sensors x 2(row and col)
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
        subplot_ind = (row - min_row) * cols + col - min_col + 1 # +1 as matlab uses based indexing
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
            ax.set_xticks([150,300,450])
            ax.set_xticklabels([])
        else:
            ax.set_xticks([])
            
            
        ax.xaxis.grid(True)
        # make line at zero
        ax.axhline(y=0,ls=':', color="grey")
        figure.subplots_adjust(hspace=hspace)
    return figure

def plot_head_signals_tight_two_signals(signals1, signals2, 
    sensor_names=None,  figsize=(10,8), plot_args=None):
    assert len(signals1) == len(signals2)
    assert sensor_names is not None
    both_signals = [signals1, signals2]
    return plot_head_signals_tight_multiple_signals(both_signals,
        sensor_names=sensor_names, figsize=figsize,
        plot_args=plot_args)

def plot_head_signals_tight_multiple_signals(all_signals, sensor_names=None,
    figsize=(10,8), plot_args=None):
    assert sensor_names is not None
    assert all([len(signals) == len(all_signals[0]) for signals in all_signals])
    if plot_args is None:
        plot_args = dict()
    figure = pyplot.figure(figsize=figsize)
    sensor_positions = [get_sensor_pos(name, tight_C_positions) for name in sensor_names]
    sensor_positions = np.array(sensor_positions) #sensors x 2(row and col)
    maxima = np.max(sensor_positions, axis=0)
    minima = np.min(sensor_positions, axis=0)
    max_row = maxima[0,0]
    max_col = maxima[1,0]
    min_row = minima[0,0]
    min_col = minima[1,0]
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
                subplot_spec=outer_grid[row-min_row,col-min_col], wspace=0.0, hspace=0.0)
        for signal_type in xrange(len(all_signals)):
            signal = all_signals[signal_type][i]
            if first_ax is None:
                ax = pyplot.Subplot(figure, inner_grid[signal_type,0])
                first_ax = ax
            else:
                ax = pyplot.Subplot(figure, inner_grid[signal_type,0], sharey=first_ax, sharex=first_ax)
            
            if signal_type == 0:
                ax.set_title(sensor_name, fontsize=10)
    
            ax.plot(signal, **plot_args)
            ax.xaxis.grid(True)
            # make line at zero
            ax.axhline(y=0,ls=':', color="grey")
            figure.add_subplot(ax)
        
        
    if len(signal) == 600:
        pyplot.xticks([150,300,450], [])
    else:
        pyplot.xticks([])
    
    pyplot.yticks([])
    return figure
    
        
def plot_sensor_signals(signals, sensor_names=None, figsize=None, 
        yticks=None, plotargs=[], sharey=True, highlight_zero_line=True,
        xvals=None,fontsize=9):
    assert sensor_names is None or len(signals) == len(sensor_names), ("need "
        "sensor names for all sensor matrices")
    if sensor_names is None:
        sensor_names = map(str, range(len(signals)))  
    num_sensors = signals.shape[0]
    if figsize is None:
        figsize = (7, np.maximum(num_sensors // 4, 1))
    figure, axes = pyplot.subplots(num_sensors, sharex=True, sharey=sharey,
        figsize=figsize)
    for sensor_i in xrange(num_sensors):
        if num_sensors > 1:
            ax = axes[sensor_i]
        else:
            ax = axes
        if xvals is None:
            ax.plot(signals[sensor_i], *plotargs)
        else:
            ax.plot(xvals, signals[sensor_i], *plotargs)
        if yticks is None:
            ax.set_yticks([])
        elif (isinstance(yticks, list)): 
            ax.set_yticks(yticks)
        elif yticks == "minmax":
            ymin, ymax = ax.get_ylim()
            ax.set_yticks((ymin, ymax - ymax/10.0))
        elif yticks == "onlymax":
            ymin, ymax = ax.get_ylim()
            ax.set_yticks([ymax])
        elif yticks == "keep": 
            pass
        ax.text(-0.035, 0.4, sensor_names[sensor_i], fontsize=fontsize,
            transform=ax.transAxes,
            horizontalalignment='right')
        if (highlight_zero_line):
            # make line at zero
            ax.axhline(y=0,ls=':', color="grey")
    max_ylim = np.max(np.abs(pyplot.ylim()))
    pyplot.ylim(-max_ylim, max_ylim)
    figure.subplots_adjust(hspace=0)
    return figure

def plot_misclasses_for_file(result_file_path):
    assert result_file_path.endswith('result.pkl')
    result = serial.load(result_file_path)
    fig = plot_misclasses_for_result(result)
    fig.suptitle("Misclass: " + result_file_path)
    return fig

def plot_misclasses_loss_for_file(result_file_path):
    assert result_file_path.endswith('result.pkl')
    fig = pyplot.figure()
    result = serial.load(result_file_path)
    plot_misclasses_for_result(result, fig)
    plot_loss_for_result(result, fig)
    
    fig.suptitle("Misclass/Loss: " + result_file_path)
    return fig


def plot_loss_for_file(result_file_path, figure=None, start=None, stop=None):
    assert result_file_path.endswith('result.pkl')
    if figure is None:
        figure = pyplot.figure()
    result = serial.load(result_file_path)
    figure = plot_loss_for_result(result,figure,start,stop)
    figure.suptitle("Loss: " + result_file_path)
    return figure

def add_early_stop_boundary(monitor_channels):
    """Plot early stop boundary as black vertical line into plot
    Determine it by epoch with largest difference in runtime."""
    runtimes_after_first = monitor_channels['runtime'][1:]
    i_last_epoch_before_early_stop = np.argmax(np.abs(runtimes_after_first - 
        np.mean(runtimes_after_first)))
    pyplot.axvline(i_last_epoch_before_early_stop, color='black', lw=1)


def plot_misclasses_for_result(result, figure=None):
    fig =  plot_train_valid_test_epochs(result.monitor_channels['train_misclass'],
        result.monitor_channels['valid_misclass'],
        result.monitor_channels['test_misclass'],
        figure=figure)
    pyplot.ylim(0, 1)
    add_early_stop_boundary(result.monitor_channels)
    return fig

def plot_train_valid_test_epochs(train, valid,test, figure=None):
    if figure is None:
        figure = pyplot.figure()
    pyplot.plot(train)
    pyplot.plot(valid)
    pyplot.plot(test)
    pyplot.legend(('train', 'valid', 'test'))
    return figure

def plot_loss_for_result(result, figure=None, start=None, stop=None):
    fig = plot_train_valid_test_epochs(result.monitor_channels['train_loss'],
        result.monitor_channels['valid_loss'],
        result.monitor_channels['test_loss'],
        figure=figure)
    
    add_early_stop_boundary(result.monitor_channels)
    return fig

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


def plot_confusion_matrix(confusion_mat, class_names=None, figsize=None, colormap=cm.bwr):
    # TODELAY: split into several functions
    # transpose to get confusion matrix same way as matlab
    confusion_mat = confusion_mat.T
    n_classes = confusion_mat.shape[0]
    if class_names is None:
        class_names = [str(i_class + 1) for i_class in xrange(n_classes)]
        
    # norm by number of targets (targets are columns after transpose!)
    normed_conf_mat = confusion_mat / np.sum(confusion_mat, 
        axis=0).astype(float)
    augmented_conf_mat = deepcopy(normed_conf_mat)
    augmented_conf_mat = np.vstack([augmented_conf_mat, [np.nan] *n_classes])
    augmented_conf_mat = np.hstack([augmented_conf_mat, [[np.nan]] * (n_classes+1)])
    
    fig = pyplot.figure(figsize=figsize)
    pyplot.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.imshow(np.array(augmented_conf_mat), cmap=colormap,
        interpolation='nearest', alpha=0.6)
    width = len(confusion_mat)
    height = len(confusion_mat[0])
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate("{:d}\n".format(confusion_mat[x][y]),
                        xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=12,
                        color='white',
                        fontweight='bold')
            
            ax.annotate("\n\n{:4.1f}%".format(
                        (confusion_mat[x][y] / float(np.sum(confusion_mat)))*100),
                        xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=10,
                        color='white',
                        fontweight='bold')
    
    # Add values for target correctness etc.
    for x in xrange(width):
        y = len(confusion_mat)
        correctness = confusion_mat[x][x] / float(np.sum(confusion_mat[x,:]))
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
        correctness = confusion_mat[y][y] / float(np.sum(confusion_mat[:,y]))
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
    
    pyplot.xticks(range(width), class_names, fontsize=12)
    pyplot.yticks(range(height), class_names, fontsize=12)
    pyplot.grid(False)
    pyplot.ylabel('Predictions', fontsize=15)
    pyplot.xlabel('Targets', fontsize=15)
    
    return fig

def plot_most_activated_neurons(activations, layers, num_neurons, plotfunction, figsize=(13,7)):
    sum_per_neuron = np.sum(np.array(activations), axis=0)
    layer_1_sums = sum_per_neuron[1]
    strongest_neurons = np.argsort(layer_1_sums)[::-1][0:num_neurons]
    plotfunction(strongest_neurons, layers, num_neurons, figsize)
    
def plot_most_variant_neurons(activations, layers, num_neurons, plotfunction, figsize=(13,7)):
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
    fig = pyplot.figure(figsize=(2,6))
    pyplot.imshow(np.atleast_2d(probs), interpolation='nearest', cmap=cm.bwr,
                          origin='lower', vmin=-value_minmax, vmax=value_minmax)
    # hide normal x/y ticks but show some ticks for orientation in case two classes have almost same color
    fig.axes[0].get_xaxis().set_ticklabels([])
    fig.axes[0].get_yaxis().set_ticks([])
    fig.axes[0].get_xaxis().set_ticks([0.5,1.5,2.5])

def plot_chan_matrices(matrices, sensor_names, figname='', figure=None,
    figsize=(8,4.5), yticks = None, yticklabels=None, 
    correctness_matrices = None, colormap=cm.coolwarm,
    sensor_map=cap_positions):
    """ figsize ignored if figure given """
    # for now hack it here... giving freq labels with 2 hz width if likely
    # that this is correct ind of input
    # TODELAY: do this properly
    if yticks == None and yticklabels == None and matrices.shape[2] > 1:
        freq_bins = np.fft.rfftfreq(n=250,d=1/500.0)
        wanted_ticks = 5
        step_size = matrices.shape[2] // wanted_ticks
        freq_bins = freq_bins[:matrices.shape[2]:step_size]
        yticks = freq_bins / 2
        yticklabels = freq_bins
    
    assert len(matrices) == len(sensor_names), "need sensor names for all sensor matrices"
    if figure is None:
        figure = pyplot.figure(figsize=figsize)
    sensor_positions = [get_sensor_pos(name, sensor_map) for name in sensor_names]
    sensor_positions = np.array(sensor_positions) # #sensors x 2(row and col) x1(for some reason:)) 
    maxima = np.max(sensor_positions, axis =0)
    minima = np.min(sensor_positions, axis =0)
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
        figure.suptitle(figname,fontsize=14)# + ", pixel abs mean(x100):  {:.3f}".format(mean_abs_weight * 100),
    
    first_ax = None
    for i in xrange(0, len(matrices)):
        sensor_name = sensor_names[i]
        sensor_pos = sensor_positions[i]
        assert np.all(sensor_pos == get_sensor_pos(sensor_name, sensor_map))
        # Transform to flat sensor pos
        row = sensor_pos[0]
        col = sensor_pos[1]
        subplot_ind = (row - min_row) * cols + col - min_col + 1 # +1 as matlab uses based indexing
        
        if first_ax is None:
            ax = figure.add_subplot(rows, cols, subplot_ind)
            first_ax = ax
        else:
            ax = figure.add_subplot(rows, cols, subplot_ind,sharey=first_ax)
            
        chan_matrix = matrices[i]
        ax.set_title(sensor_name)
        if (correctness_matrices is None):
            #ax.pcolor(chan_matrix.T,
            #     cmap=cm.bwr, #origin='lower', #interpolation='nearest',#aspect='auto'
            #    vmin=-mean_abs_weight * 2, vmax= mean_abs_weight * 2)
            ax.pcolorfast(chan_matrix.T,
                 cmap=colormap, #origin='lower', #interpolation='nearest',#aspect='auto'
                vmin=-mean_abs_weight * 2, vmax= mean_abs_weight * 2)
            #ax.imshow(chan_matrix.T,
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
                vmin=-mean_abs_weight * 2, vmax= mean_abs_weight * 2)
            # use yellow/brown for red(positive) incorrect values
            # (mask out correct or blue=negative values)
            # also take minus the values as otherwise they go from
            # minus-something to 0, that makes colormaps 
            # more complicated to use :) 
            ax.imshow(
                np.ma.masked_where(
                    np.logical_or(correctness_mat > 0, chan_matrix < 0),
                    -(chan_matrix * correctness_mat)).T,
                interpolation='nearest', cmap=cm.YlOrBr, origin='lower',
                vmin=0, vmax= mean_abs_weight * 2)
            # use purple for blue(negative) incorrect values
            ax.imshow(
                np.ma.masked_where(
                    np.logical_or(correctness_mat > 0, chan_matrix >= 0), 
                    chan_matrix * correctness_mat).T,
                interpolation='nearest', cmap=cm.PuRd, origin='lower',
                vmin=0, vmax= mean_abs_weight * 2)
        ax.set_xticks([])
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(color='k', linewidth=0.1, linestyle=':')
    return figure
