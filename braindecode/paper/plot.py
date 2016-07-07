# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from braindecode.mywyrm.plot import ax_scalp
from braindecode.paper import map_i_class_pair

def plot_csp_patterns(wanted_patterns, sensor_names, i_fb=3):
    """Expects filterband x classpair  x sensor x 2 (pattern)"""
    original_class_names = ('Hand (R)', 'Hand (L)', 'Rest', 'Feet')
    fig = plt.figure(figsize=(12,2))
    for i_class_pair in range(6):
        i_wanted_class_pair, wanted_class_pair, reverse_filters = map_i_class_pair(i_class_pair)
        pair_patterns = wanted_patterns[i_fb,i_wanted_class_pair]
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
    plt.text(0.27,0.5,u"7–13 Hz", transform=fig.transFigure, fontsize=14, rotation=90, va='center')
    None

def plot_scalp_grid(data, sensor_names, scale_per_row=False,
                         scale_per_column=False, 
                         scale_individually=False, figsize=None, 
                         row_names=None, col_names=None,
                         vmin=None, vmax=None,
                        colormap=cm.coolwarm,
                        add_colorbar=True):
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
                    vmin=vmin, vmax=vmax)
            if col_names is not None and i_row == 0:
                ax.set_title(col_names[i_col], fontsize=16)
            if row_names is not None and i_col == 0:
                ax.set_ylabel(row_names[i_row], fontsize=16)
    fig.subplots_adjust(hspace=-0.3,wspace=0.)
    return fig, axes

def add_colorbar_to_scalp_grid(fig, axes, label, min_max_ticks=True, shrink=0.9,
        **colorbar_args):
    cbar = fig.colorbar(fig.axes[2].images[0], ax=axes.ravel().tolist(),
        shrink=shrink, **colorbar_args)
    if min_max_ticks:
        clim =cbar.get_clim()
        cbar.set_ticks((clim[0],0,clim[1]))
        cbar.set_ticklabels(('min','0','max'))
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label, fontsize=16)
    return cbar
