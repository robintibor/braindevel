# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from braindecode.mywyrm.plot import ax_scalp
from braindecode.paper import map_i_class_pair, resorted_class_names
from copy import deepcopy
from braindecode.datasets.sensor_positions import CHANNEL_10_20_APPROX
from braindecode.mywyrm.plot import add_ears, get_channelpos

def plot_freq_bands_corrs_topo(corrs, freqs, freq_bands, sensor_names, merge_func):
    """Expects corrs classes already resorted."""
    freq_amp_corrs = []
    freq_strs = [u"{:d}—{:d} Hz".format(low,high) for low, high in freq_bands]
    for i_freq, (freq_low, freq_high) in enumerate(freq_bands):
        i_freq_start = np.searchsorted(freqs,freq_low) - 1
        i_freq_stop = np.searchsorted(freqs,freq_high)
        freq_amp_corrs.append([])
        for i_class in xrange(4):
            freq_amp_corrs[-1].append(merge_func(corrs[:,i_class,:,
                i_freq_start:i_freq_stop], axis=(0,2)))
    freq_amp_corrs = np.array(freq_amp_corrs)
    fig,axes = plot_scalp_grid(freq_amp_corrs, sensor_names, scale_individually=True,
                   col_names=resorted_class_names, row_names=freq_strs, figsize=(14,8),
                              fontsize=30)
    fig.tight_layout()
    fig.subplots_adjust(hspace=-0.3,wspace=0.)
    cbar = add_colorbar_to_scalp_grid(fig, axes, label='Correlation',
                                     ticklabelsize=28,
                                     labelsize=32)
def plot_freq_classes_corrs(corrs, freqs, merge_func):
    """Expects corrs classes already resorted."""
    # draw image
    plt.figure(figsize=(8,1.2))
    freq_classes_corrs = merge_func(corrs, axis=(0,2))
    im = plt.imshow(freq_classes_corrs, cmap=cm.coolwarm, interpolation='nearest',
          aspect='auto', vmin=-np.max(np.abs(freq_classes_corrs)), 
                vmax=np.max(np.abs(freq_classes_corrs)))
    plt.xticks(range(freq_classes_corrs.shape[1])[::20], freqs[::20].astype(np.int32))
    plt.yticks(range(4), resorted_class_names)
    cbar = plt.colorbar(im)#, orientation='horizontal')
    cbar.set_ticks(np.round(np.linspace(cbar.get_clim()[0], cbar.get_clim()[1], 3), 4))
    cbar.set_label('Correlation')
    plt.xlabel('Frequency [Hz]')
    plt.tight_layout()

def plot_csp_patterns(wanted_patterns, sensor_names, i_fb=3, freq_str=u"7—13 Hz"):
    """
    THIS WAS ONLY FOR PAPER; REMOVE LATER :D see function below
    Expects filterband x classpair  x sensor x 2 (pattern).
    Expects classpairs in original order, i.e. 
    Hand(R)/Hand(L), Hand(R)/Rest, Hand(R)/Feet,
    Hand(L)/Rest, Hand(L)/Feet, Feet/Rest"""
    fb_patterns = wanted_patterns[i_fb]
    fig = plot_csp_patterns_(fb_patterns, sensor_names)
    
    plt.text(0.27,0.5, freq_str, transform=fig.transFigure, fontsize=14,
        rotation=90, va='center')
    None


def plot_csp_patterns_(all_patterns, sensor_names,
        original_class_names=('Hand (R)', 'Hand (L)', 'Rest', 'Feet')):
    """Expects filterband x classpair  x sensor x 2 (pattern).
    Expects classpairs in original order, i.e. 
    Hand(R)/Hand(L), Hand(R)/Rest, Hand(R)/Feet,
    Hand(L)/Rest, Hand(L)/Feet, Feet/Rest"""
    fig = plt.figure(figsize=(12,2))
    for i_class_pair in range(6):
        i_wanted_class_pair, wanted_class_pair, reverse_filters = map_i_class_pair(i_class_pair)
        pair_patterns = all_patterns[i_wanted_class_pair]
        if reverse_filters:
            pair_patterns = pair_patterns[:,::-1]
        for i_sub_pattern in range(2):
            pattern = pair_patterns[:,i_sub_pattern]
            ax = plt.subplot(2,6, i_class_pair+(i_sub_pattern * 6)+1)
            if i_sub_pattern == 0 and i_class_pair == 0:
                scalp_line_width = 1
                #ax.set_ylabel(u"{:.0f}—{:.0f} Hz".format(*filterbands[i_fb]))
            else:
                scalp_line_width = 0

            ax_scalp(pattern,sensor_names, colormap=cm.PRGn, ax=ax,
                    vmin=-np.max(np.abs(pattern)), vmax=np.max(np.abs(pattern)),
                    scalp_line_width=scalp_line_width)
            if i_sub_pattern == 0:
                # reversefilters is 0 if not to be reversed and 1 if to be revrsed
                ax.set_title(original_class_names[wanted_class_pair[reverse_filters]])
            else:
                ax.set_xlabel(original_class_names[wanted_class_pair[1-reverse_filters]])
    fig.subplots_adjust(wspace=-0.7,hspace=-0)
    add_colorbar_to_scalp_grid(fig,np.array(fig.axes),'', shrink=1)
    return fig
    
def plot_scalp_grid(data, sensor_names, scale_per_row=False,
                         scale_per_column=False, 
                         scale_individually=False, figsize=None, 
                         row_names=None, col_names=None,
                         vmin=None, vmax=None,
                        chan_pos_list=CHANNEL_10_20_APPROX,
                        colormap=cm.coolwarm,
                        fontsize=16):
    """
    data: 3darray
        freqs x classes x sensors
    """
    assert np.sum([scale_per_row, scale_per_column, scale_individually]) < 2, (
               "Can have only one way of scaling...")
    if vmin is None:
        assert vmax is None
        max_abs_val = np.max(np.abs(data))
        vmin = -max_abs_val
        vmax = max_abs_val
        
    n_rows = data.shape[0]
    n_cols = data.shape[1]
    if figsize is None:
        figsize = (n_rows*3, n_cols*2)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,figsize=figsize)
    for i_row in xrange(n_rows):
        if scale_per_row:
            max_abs_val = np.max(np.abs(data[i_row]))
            vmin = -max_abs_val
            vmax = max_abs_val
        for i_col in xrange(n_cols):
            this_data = data[i_row,i_col]
            if scale_per_column:
                max_abs_val = np.max(np.abs(data[:,i_col]))
                vmin = -max_abs_val
                vmax = max_abs_val
            if scale_individually:
                max_abs_val = np.max(np.abs(this_data))
                vmin = -max_abs_val
                vmax = max_abs_val
            scalp_line_style = 'solid'
            if i_row == 0 and i_col == 0:
                scalp_line_width = 1
            else:
                scalp_line_width=0
            if n_rows > 1:
                ax = axes[i_row][i_col]
            else:
                ax = axes[i_col]
            ax_scalp(this_data,sensor_names, colormap=colormap,ax=ax,
                    scalp_line_width=scalp_line_width, scalp_line_style=scalp_line_style,
                    vmin=vmin, vmax=vmax, chan_pos_list=chan_pos_list,
                    zorder=10)
            if col_names is not None and i_row == 0:
                ax.set_title(col_names[i_col], fontsize=fontsize)
            if row_names is not None and i_col == 0:
                ax.set_ylabel(row_names[i_row], fontsize=fontsize)
    fig.subplots_adjust(hspace=-0.3,wspace=0.)
    return fig, axes

def add_colorbar_to_scalp_grid(fig, axes, label, min_max_ticks=True, shrink=0.9,
    ticklabelsize=14, 
    labelsize=16,
        **colorbar_args):
    cbar = fig.colorbar(fig.axes[2].images[0], ax=axes.ravel().tolist(),
        shrink=shrink, **colorbar_args)
    if min_max_ticks:
        clim =cbar.get_clim()
        cbar.set_ticks((clim[0],0,clim[1]))
        cbar.set_ticklabels(('min','0','max'))
    cbar.ax.tick_params(labelsize=ticklabelsize)
    cbar.set_label(label, fontsize=labelsize)
    return cbar

# see http://stackoverflow.com/a/31397438/1469195
def cmap_map(function, cmap, name='colormap_mod', N=None, gamma=None):
    """
    Modify a colormap using `function` which must operate on 3-element
    arrays of [r, g, b] values.

    You may specify the number of colors, `N`, and the opacity, `gamma`,
    value of the returned colormap. These values default to the ones in
    the input `cmap`.

    You may also specify a `name` for the colormap, so that it can be
    loaded using plt.get_cmap(name).
    """
    from matplotlib.colors import LinearSegmentedColormap as lsc
    if N is None:
        N = cmap.N
    if gamma is None:
        gamma = cmap._gamma
    cdict = cmap._segmentdata
    # Cast the steps into lists:
    step_dict = {key: map(lambda x: x[0], cdict[key]) for key in cdict}
    # Now get the unique steps (first column of the arrays):
    step_list = np.unique(sum(step_dict.values(), []))
    # 'y0', 'y1' are as defined in LinearSegmentedColormap docstring:
    y0 = cmap(step_list)[:, :3]
    y1 = y0.copy()[:, :3]
    # Go back to catch the discontinuities, and place them into y0, y1
    for iclr, key in enumerate(['red', 'green', 'blue']):
        for istp, step in enumerate(step_list):
            try:
                ind = step_dict[key].index(step)
            except ValueError:
                # This step is not in this color
                continue
            y0[istp, iclr] = cdict[key][ind][1]
            y1[istp, iclr] = cdict[key][ind][2]
    # Map the colors to their new values:
    y0 = np.array(map(function, y0))
    y1 = np.array(map(function, y1))
    # Build the new colormap (overwriting step_dict):
    for iclr, clr in enumerate(['red', 'green', 'blue']):
        step_dict[clr] = np.vstack((step_list, y0[:, iclr], y1[:, iclr])).T
    return lsc(name, step_dict, N=N, gamma=gamma)

def plot_confusion_matrix_paper(confusion_mat, p_val_vs_csp,
                                p_val_vs_other_net,
                                class_names=None, figsize=None, colormap=cm.bwr,
        textcolor='black', vmin=None, vmax=None,
                               fontweight='normal',
                           rotate_row_labels=90,
                           rotate_col_labels=0,
                           with_f1_score=False):
    # TODELAY: split into several functions
    # transpose to get confusion matrix same way as matlab
    confusion_mat = confusion_mat.T
    # then have to transpose pvals also
    p_val_vs_csp = p_val_vs_csp.T
    p_val_vs_other_net = p_val_vs_other_net.T
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
    # see http://stackoverflow.com/a/31397438/1469195
    def brighten(x, ):
        return (1 - ((1 - x) * 0.4))
    brightened_cmap = cmap_map(brighten, colormap)
    ax.imshow(np.array(augmented_conf_mat), cmap=brightened_cmap,
        interpolation='nearest', vmin=vmin, vmax=vmax)
    width = len(confusion_mat)
    height = len(confusion_mat[0])
    for x in xrange(width):
        for y in xrange(height):
            if x == y:
                this_font_weight = 'bold'
            else:
                this_font_weight = fontweight
            annotate_str = "{:d}".format(confusion_mat[x][y])
            if p_val_vs_csp[x][y] < 0.05:
                annotate_str += " *"
            else:
                annotate_str += "  "
            if p_val_vs_csp[x][y] < 0.01:
                annotate_str += u"*"
            if p_val_vs_csp[x][y] < 0.001:
                annotate_str += u"*"
                
            if p_val_vs_other_net[x][y] < 0.05:
                annotate_str += u" ◊"
            if p_val_vs_other_net[x][y] < 0.01:
                annotate_str += u"◊"
            if p_val_vs_other_net[x][y] < 0.001:
                annotate_str += u"◊"
            annotate_str += "\n"
            ax.annotate(annotate_str.format(confusion_mat[x][y]),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=12,
                    color=textcolor,
                    fontweight=this_font_weight)
            if x != y or (not with_f1_score):
                ax.annotate("\n\n{:4.1f}%".format(
                            (confusion_mat[x][y] / float(np.sum(confusion_mat))) * 100),
                            xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center', fontsize=10,
                            color=textcolor,
                            fontweight=this_font_weight)
            else:
                assert x == y
                precision = confusion_mat[x][x] / float(np.sum(
                    confusion_mat[x, :]))
                sensitivity = confusion_mat[x][x] / float(np.sum(
                    confusion_mat[:, y]))
                f1_score = 2 * precision * sensitivity / (precision + sensitivity)
                
                ax.annotate("\n{:4.1f}%\n{:4.1f}% (F)".format(
                            (confusion_mat[x][y] / float(np.sum(confusion_mat))) * 100,
                            f1_score * 100),
                            xy=(y, x+0.1),
                            horizontalalignment='center',
                            verticalalignment='center', fontsize=10,
                            color=textcolor,
                            fontweight=this_font_weight)
    
    # Add values for target correctness etc.
    for x in xrange(width):
        y = len(confusion_mat)
        correctness = confusion_mat[x][x] / float(np.sum(confusion_mat[x, :]))
        annotate_str = ""
        if p_val_vs_csp[x][y] < 0.05:
            annotate_str += " *"
        else:
            annotate_str += "  "
        if p_val_vs_csp[x][y] < 0.01:
            annotate_str += u"*"
        if p_val_vs_csp[x][y] < 0.001:
            annotate_str += u"*"

        if p_val_vs_other_net[x][y] < 0.05:
            annotate_str += u" ◊"
        if p_val_vs_other_net[x][y] < 0.01:
            annotate_str += u"◊"
        if p_val_vs_other_net[x][y] < 0.001:
            annotate_str += u"◊"
        annotate_str += "\n{:5.2f}%".format(correctness * 100)
        ax.annotate(annotate_str,
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=12)
        
    
    for y in xrange(height):
        x = len(confusion_mat)
        correctness = confusion_mat[y][y] / float(np.sum(confusion_mat[:, y]))
        annotate_str = ""
        if p_val_vs_csp[x][y] < 0.05:
            annotate_str += " *"
        else:
            annotate_str += "  "
        if p_val_vs_csp[x][y] < 0.01:
            annotate_str += u"*"
        if p_val_vs_csp[x][y] < 0.001:
            annotate_str += u"*"

        if p_val_vs_other_net[x][y] < 0.05:
            annotate_str += u" ◊"
        if p_val_vs_other_net[x][y] < 0.01:
            annotate_str += u"◊"
        if p_val_vs_other_net[x][y] < 0.001:
            annotate_str += u"◊"
        annotate_str += "\n{:5.2f}%".format(correctness * 100)
        ax.annotate(annotate_str,
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=12)
        
    overall_correctness = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat).astype(float)
    ax.annotate("{:5.2f}%".format(overall_correctness * 100),
                        xy=(len(confusion_mat), len(confusion_mat)),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=12,
               fontweight='bold')
    
    plt.xticks(range(width), class_names, fontsize=12, rotation=rotate_col_labels)
    plt.yticks(range(height), class_names, fontsize=12, rotation=rotate_row_labels)
    plt.grid(False)
    plt.ylabel('Predictions', fontsize=15)
    plt.xlabel('Targets', fontsize=15)
    
    # n classes is also shape of matrix/size
    ax.text(-1.1, n_classes, "Sensitivity", ha='center', va='center',
           fontsize=13)
    ax.text(n_classes, -1.1, "Precision", ha='center', va='center', rotation=90,#270,
           fontsize=13)
    
    return fig

def plot_conf_mat(conf_mat, p_val_vs_csp, p_val_vs_other_net, label,
        class_names=resorted_class_names,
                 add_colorbar=True,
                 figsize=(6,6),
                 vmin=0,
                 vmax=0.1,
                 rotate_row_labels=90,
                 rotate_col_labels=0,
                 with_f1_score=False):
    fig = plot_confusion_matrix_paper(conf_mat, p_val_vs_csp, p_val_vs_other_net,
                                figsize=figsize, 
                                class_names=class_names, 
                                      #colormap=seaborn.cubehelix_palette(8, as_cmap=True),#, start=.5, rot=-.75),
                                      colormap=cm.OrRd,
                           vmin=vmin, vmax=vmax,
                           rotate_row_labels=rotate_row_labels,
                           rotate_col_labels=rotate_col_labels,
                           with_f1_score=with_f1_score)
    plt.title(label, fontsize=20,y=1.04)
    
    cbar = plt.colorbar(fig.axes[0].images[0], shrink=0.9)
    ticks = np.linspace(0, vmax, 5, endpoint=True)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks * 100)
    cbar.set_label('Trials [%]',labelpad=10, fontsize=14)
    if not add_colorbar:
        # hack to have same size of figure but remove colorbar parts
        cbar.set_ticks([])
        cbar.set_label('')
        fig.axes[1].cla()
        fig.axes[1].axis('off')
        
    None
    

def scalp_with_circles(v, channels, 
    ax=None, annotate=False,
    vmin=None, vmax=None, colormap=None,
    scalp_line_width=1,
    scalp_line_style='solid',
    chan_pos_list=CHANNEL_10_20_APPROX,
    interpolation='bilinear'):
    """Draw a scalp plot.

    Draws a scalp plot on an existing axes. The method takes an array of
    values and an array of the corresponding channel names. It matches
    the channel names with an internal list of known channels and their
    positions to project them correctly on the scalp.

    .. warning:: The behaviour for unkown channels is undefined.

    Parameters
    ----------
    v : 1d-array of floats
        The values for the channels
    channels : 1d array of strings
        The corresponding channel names for the values in ``v``
    ax : Axes, optional
        The axes to draw the scalp plot on. If not provided, the
        currently activated axes (i.e. ``gca()``) will be taken
    annotate : Boolean, optional
        Draw the channel names next to the channel markers.
    vmin, vmax : float, optional
        The display limits for the values in ``v``. If the data in ``v``
        contains values between -3..3 and ``vmin`` and ``vmax`` are set
        to -1 and 1, all values smaller than -1 and bigger than 1 will
        appear the same as -1 and 1. If not set, the maximum absolute
        value in ``v`` is taken to calculate both values.
    colormap : matplotlib.colors.colormap, optional
        A colormap to define the color transitions.

    Returns
    -------
    ax : Axes
        the axes on which the plot was drawn

    See Also
    --------
    ax_colorbar

    """
    if ax is None:
        ax = plt.gca()
    assert len(v) == len(channels), "Should be as many values as channels"
    assert interpolation=='bilinear' or interpolation=='nearest'
    if vmin is None:
        # added by me (robintibor@gmail.com)
        assert vmax is None
        vmin, vmax = -np.max(np.abs(v)), np.max(np.abs(v))
    # what if we have an unknown channel?
    points = [get_channelpos(c, chan_pos_list) for c in channels]
    for c in channels:
        assert get_channelpos(c, chan_pos_list) is not None, ("Expect " + c + " "
            "to exist in positions")
    values = [v[i] for i in range(len(points))]
   
    for (x,y),z in zip(points, values):
        if z > 0:
            fill = 'red'
        else:
            fill = False
        ax.add_artist(plt.Circle((x, y), 0.03, linestyle=scalp_line_style,
            linewidth=0.2, fill=fill, facecolor=cm.coolwarm(z)))
        #plt.plot(x,y,marker='x', markersize=5)
            
    
    # paint the head
    ax.add_artist(plt.Circle((0, 0), 1, linestyle=scalp_line_style,
        linewidth=scalp_line_width, fill=False))
    # add a nose
    ax.plot([-0.1, 0, 0.1], [1, 1.1, 1], color='black', 
        linewidth=scalp_line_width, linestyle=scalp_line_style,)
    # add ears
    add_ears(ax, scalp_line_width, scalp_line_style)
    
    # set the axes limits, so the figure is centered on the scalp
    ax.set_ylim([-1.05, 1.15])
    ax.set_xlim([-1.15, 1.15])
    
    # hide the frame and ticks
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # draw the channel names
    if annotate:
        for i in zip(channels, list(zip(x, y))):
            ax.annotate(" " + i[0], i[1],horizontalalignment="center",
                verticalalignment='center')
    ax.set_aspect(1)
