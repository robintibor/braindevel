import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from braindecode.mywyrm.plot import ax_scalp

def plot_scalp_grid(data, sensor_names, scale_per_row=False,
                         scale_per_column=False, 
                         scale_individually=False, figsize=None, 
                         row_names=None, col_names=None,
                         vmin=None, vmax=None,
                        colormap=cm.coolwarm,
                        add_colorbar=True):
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
