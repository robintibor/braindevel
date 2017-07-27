import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import interpolate
from matplotlib import patches
from matplotlib.path import Path
from braindevel.datasets.sensor_positions import CHANNEL_10_20_APPROX

def get_channelpos(channame, chan_pos_list):
    
    if chan_pos_list[0] == 'angle':
        return get_channelpos_from_angle(channame, chan_pos_list[1:])
    elif chan_pos_list[0] == 'cartesian':
        channame = channame.lower()
        for name, coords in chan_pos_list[1:]:
            if name.lower() == channame:
                return coords[0], coords[1]
        return None
    else:
        raise ValueError("Unknown first element "
            "{:s} (should be type of positions)".format(chan_pos_list[0]))

def get_channelpos_from_angle(channame, chan_pos_list=CHANNEL_10_20_APPROX):
    """Return the x/y position of a channel.

    This method calculates the stereographic projection of a channel
    from ``CHANNEL_10_20``, suitable for a scalp plot.

    Parameters
    ----------
    channame : str
        Name of the channel, the search is case insensitive.

    Returns
    -------
    x, y : float or None
        The projected point on the plane if the point is known,
        otherwise ``None``

    Examples
    --------

    >>> plot.get_channelpos_from_angle('C2')
    (0.1720792096741632, 0.0)
    >>> # the channels are case insensitive
    >>> plot.get_channelpos_from_angle('c2')
    (0.1720792096741632, 0.0)
    >>> # lookup for an invalid channel
    >>> plot.get_channelpos_from_angle('foo')
    None

    """
    channame = channame.lower()
    for i in chan_pos_list:
        if i[0].lower() == channame:
            # convert the 90/4th angular position into x, y, z
            p = i[1]
            ea, eb = p[0] * (90 / 4), p[1] * (90 / 4)
            ea = ea * math.pi / 180
            eb = eb * math.pi / 180
            x = math.sin(ea) * math.cos(eb)
            y = math.sin(eb)
            z = math.cos(ea) * math.cos(eb)
            # Calculate the stereographic projection.
            # Given a unit sphere with radius ``r = 1`` and center at
            # the origin. Project the point ``p = (x, y, z)`` from the
            # sphere's South pole (0, 0, -1) on a plane on the sphere's
            # North pole (0, 0, 1).
            #
            # The formula is:
            #
            # P' = P * (2r / (r + z))
            #
            # We changed the values to move the point of projection
            # further below the south pole
            mu = 1 / (1.3 + z)
            x *= mu
            y *= mu
            return x, y
    return None

def ax_scalp(v, channels, 
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
    z = [v[i] for i in range(len(points))]
    # calculate the interpolation
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    # interpolate the in-between values
    xx = np.linspace(min(x), max(x), 500)
    yy = np.linspace(min(y), max(y), 500)
    if  interpolation == 'bilinear':
        xx_grid, yy_grid = np.meshgrid(xx, yy)
        f = interpolate.LinearNDInterpolator(list(zip(x, y)), z)
        zz = f(xx_grid, yy_grid)
    else:
        assert interpolation == 'nearest'
        f = interpolate.NearestNDInterpolator(list(zip(x, y)), z)
        assert len(xx) == len(yy)
        zz = np.ones((len(xx), len(yy)))
        for i_x in xrange(len(xx)):
            for i_y in xrange(len(yy)):
                # somehow this is correct. don't know why :(
                zz[i_y,i_x] = f(xx[i_x], yy[i_y])
                #zz[i_x,i_y] = f(xx[i_x], yy[i_y])
        assert not np.any(np.isnan(zz))
    
    # plot map
    image = ax.imshow(zz, vmin=vmin, vmax=vmax, cmap=colormap,
        extent=[min(x),max(x),min(y),max(y)], origin='lower',
        interpolation=interpolation)
    #image = ax.contourf(xx, yy, zz, 100, vmin=vmin, vmax=vmax,
    #    cmap=colormap)
    if scalp_line_width > 0:
        # paint the head
        ax.add_artist(plt.Circle((0, 0), 1, linestyle=scalp_line_style,
            linewidth=scalp_line_width, fill=False))
        # add a nose
        ax.plot([-0.1, 0, 0.1], [1, 1.1, 1], color='black', 
            linewidth=scalp_line_width, linestyle=scalp_line_style)
        # add ears
        add_ears(ax, scalp_line_width, scalp_line_style)
    # add markers at channels positions
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
    return image

def add_ears(ax, linewidth, linestyle):
    start_x = np.cos(10* np.pi/180.0)
    start_y = np.sin(10 * np.pi/180.0)
    end_x = np.cos(-15* np.pi/180.0)
    end_y = np.sin(-15* np.pi/180.0)
    verts = [
        (start_x, start_y), 
        (start_x+0.05, start_y+0.05), # out up
        (start_x+0.1, start_y), # further out, back down
        (start_x+0.11, (end_y * 0.7 + start_y * 0.3)), #midpoint
        (end_x+0.14, end_y), # down out start
        (end_x+0.05, end_y-0.05), # down out further
        (end_x, end_y), # endpoint
        ]
    
    codes = [Path.MOVETO] + [Path.CURVE3] * (len(verts) - 1)
    
    path = Path(verts, codes)
    
    patch = patches.PathPatch(path, facecolor='none', 
        linestyle=linestyle, linewidth=linewidth)
    
    
    ax.add_patch(patch)
    verts_left = [(-x,y) for x,y in verts]
    path_left = Path(verts_left, codes)
    
    patch_left = patches.PathPatch(path_left, facecolor='none', 
        linestyle=linestyle, linewidth=linewidth)
    
    ax.add_patch(patch_left)