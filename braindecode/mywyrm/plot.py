import matplotlib.pyplot as plt  
import math
import numpy as np
from scipy import interpolate

# the angles here are given in (90 / 4)th degrees - so multiply it with
# (90 / 4) to get the actual angles

""" computed by
sensor_pos_file_content = open('data/sensor-positions/Waveguard2Dpos.csv', 'r').read()

sensor_lines = sensor_pos_file_content.split('\n')
sensor_lines = sensor_lines[:-1]

name_and_coords = [l.split('\t') for l in sensor_lines]

name_and_coords = [(l[0].replace(':', '').strip(), float(l[1]), float(l[2])) for l in name_and_coords]

for name_coord in name_and_coords:
    print "('{:s}', ({:.3f}, {:.3f})),".format(name_coord[0], name_coord[2] * 4 / 90.0, name_coord[1] * 4 / 90.0)
"""

CHANNEL_10_20_exact = (
    ('Fp1', (-1.435, 3.675556)),
    ('Fpz', (0.004, 4.000000)),
    ('Fp2', (1.439, 3.675556)),
    ('F7', (-4.085, 2.382222)),
    ('F3', (-2.133, 1.737778)),
    ('Fz', (0.004, 1.577778)),
    ('F4', (2.187, 1.768889)),
    ('F8', (4.069, 2.391111)),
    ('FC5', (-3.760, 0.715556)),
    ('FC1', (-1.227, 0.440000)),
    ('FC2', (1.259, 0.466667)),
    ('FC6', (3.781, 0.746667)),
    ('M1', (-6.549, -2.248889)),
    ('T7', (-5.280, -0.586667)),
    ('C3', (-2.555, -0.720000)),
    ('Cz', (0.004, -0.746667)),
    ('C4', (2.608, -0.688889)),
    ('T8', (5.269, -0.528889)),
    ('M2', (6.555, -2.275556)),
    ('CP5', (-3.589, -1.911111)),
    ('CP1', (-1.173, -1.746667)),
    ('CP2', (1.237, -1.728889)),
    ('CP6', (3.621, -1.840000)),
    ('P7', (-3.888, -3.417778)),
    ('P3', (-1.989, -2.951111)),
    ('Pz', (0.005, -2.835556)),
    ('P4', (2.032, -2.951111)),
    ('P8', (3.883, -3.408889)),
    ('POz', (0.004, -3.542222)),
    ('O1', (-1.360, -4.648889)),
    ('Oz', (0.004, -4.711111)),
    ('O2', (1.355, -4.648889)),
    ('AF7', (-2.827, 3.244444)),
    ('AF3', (-1.472, 2.831111)),
    ('AF4', (1.525, 2.813333)),
    ('AF8', (2.805, 3.248889)),
    ('F5', (-3.136, 2.008889)),
    ('F1', (-1.061, 1.600000)),
    ('F2', (1.131, 1.617778)),
    ('F6', (3.168, 2.022222)),
    ('FC3', (-2.496, 0.542222)),
    ('FCz', (0.004, 0.417778)),
    ('FC4', (2.533, 0.577778)),
    ('C5', (-3.888, -0.671111)),
    ('C1', (-1.264, -0.733333)),
    ('C2', (1.317, -0.724444)),
    ('C6', (3.893, -0.626667)),
    ('CP3', (-2.363, -1.817778)),
    ('CPz', (0.005, -1.746667)),
    ('CP4', (2.411, -1.786667)),
    ('P5', (-2.955, -3.111111)),
    ('P1', (-0.981, -2.911111)),
    ('P2', (1.035, -2.893333)),
    ('P6', (2.981, -3.133333)),
    ('PO5', (-2.117, -4.182222)),
    ('PO3', (-1.387, -3.880000)),
    ('PO4', (1.391, -3.911111)),
    ('PO6', (2.069, -4.213333)),
    ('FT7', (-5.013, 1.044444)),
    ('FT8', (4.981, 1.093333)),
    ('TP7', (-4.848, -2.035556)),
    ('TP8', (4.837, -2.022222)),
    ('PO7', (-2.683, -3.946667)),
    ('PO8', (2.688, -3.937778)),
    ('FT9', (-6.107, 1.617778)),
    ('FT10', (6.112, 1.626667)),
    ('TPP9h', (-4.987, -3.017778)),
    ('TPP10h', (4.991, -3.017778)),
    ('PO9', (-3.355, -4.622222)),
    ('PO10', (3.344, -4.613333)),
    ('P9', (-4.901, -3.871111)),
    ('P10', (4.859, -3.884444)),
    ('AFF1', (-0.944, 2.146667)),
    ('AFz', (0.004, 2.693333)),
    ('AFF2', (0.971, 2.173333)),
    ('FFC5h', (-2.923, 1.257778)),
    ('FFC3h', (-1.733, 1.106667)),
    ('FFC4h', (1.808, 1.128889)),
    ('FFC6h', (2.981, 1.284444)),
    ('FCC5h', (-3.227, -0.053333)),
    ('FCC3h', (-1.915, -0.137778)),
    ('FCC4h', (1.957, -0.115556)),
    ('FCC6h', (3.264, -0.013333)),
    ('CCP5h', (-3.131, -1.293333)),
    ('CCP3h', (-1.840, -1.275556)),
    ('CCP4h', (1.899, -1.248889)),
    ('CCP6h', (3.173, -1.240000)),
    ('CPP5h', (-2.741, -2.373333)),
    ('CPP3h', (-1.616, -2.284444)),
    ('CPP4h', (1.669, -2.257778)),
    ('CPP6h', (2.795, -2.324444)),
    ('PPO1', (-0.885, -3.368889)),
    ('PPO2', (0.939, -3.351111)),
    ('I1', (-1.701, -5.480000)),
    ('Iz', (0.004, -5.773333)),
    ('I2', (1.680, -5.488889)),
    ('AFp3h', (-0.832, 3.248889)),
    ('AFp4h', (0.859, 3.235556)),
    ('AFF5h', (-2.261, 2.408889)),
    ('AFF6h', (2.288, 2.431111)),
    ('FFT7h', (-4.069, 1.551111)),
    ('FFC1h', (-0.549, 1.000000)),
    ('FFC2h', (0.635, 1.013333)),
    ('FFT8h', (4.069, 1.577778)),
    ('FTT9h', (-5.893, 0.497778)),
    ('FTT7h', (-4.555, 0.115556)),
    ('FCC1h', (-0.635, -0.173333)),
    ('FCC2h', (0.667, -0.168889)),
    ('FTT8h', (4.555, 0.160000)),
    ('FTT10h', (5.893, 0.488889)),
    ('TTP7h', (-4.670, -1.266667)),
    ('CCP1h', (-0.597, -1.266667)),
    ('CCP2h', (0.656, -1.262222)),
    ('TTP8h', (4.675, -1.266667)),
    ('TPP7h', (-3.840, -2.560000)),
    ('CPP1h', (-0.517, -2.222222)),
    ('CPP2h', (0.597, -2.208889)),
    ('TPP8h', (3.861, -2.560000)),
    ('PPO9h', (-3.723, -4.191111)),
    ('PPO5h', (-2.208, -3.497778)),
    ('PPO6h', (2.213, -3.497778)),
    ('PPO10h', (3.691, -4.200000)),
    ('POO9h', (-2.288, -4.866667)),
    ('POO3h', (-0.795, -4.333333)),
    ('POO4h', (0.816, -4.315556)),
    ('POO10h', (2.261, -4.871111)),
    ('OI1h', (-0.768, -5.137778)),
    ('OI2h', (0.752, -5.137778)))

#New version, extracted by:
# channels = ['Fp1','AFp3h','Fpz','AFp4h','Fp2', 
#        'AF7','AFF5h','AF3','AFF1','AFz','AFF2','Af4','AFF6h','AF8', 
#        'F7','F5','F3','F1','Fz','F2','F4','F6','F8', 
#        'FFT7h','FFC5h','FFC3h','FFC1h','FFC2h','FFC4h','FFC6h','FFT8h', 
#        'FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10', 
#        'FTT9h','FTT7h','FCC5h','FCC3h','FCC1h','FCC2h','FCC4h','FCC6h','FTT8h','FTT10h', 
#        'M1','T7','C5','C3','C1','Cz','C2','C4','C6','T8','M2', 
#        'TTP7h','CCP5h','CCP3h','CCP1h','CCP2h','CCP4h','CCP6h','TTP8h', 
#        'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8', 
#        'TPP9h','TPP7h','CPP5h','CPP3h','CPP1h','CPP2h','CPP4h','CPP6h','TPP8h','TPP10h', 
#        'P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10', 
#        'PPO9h','PPO5h','PPO1','PPO2','PPO6h','PPO10h', 
#        'PO9','PO7','PO5','PO3','PO1','POz','PO2','PO4','PO6','PO8','PO10', 
#        'POO9h','POO3h','O1','Oz','O2','POO4h','POO10h', 
#        'I1','OI1h','Iz','OI2h','I2']
# 
# #7th line from below modified from original file, there were too many values!!
# x = """-4 -1.5 0 1.5 4 
#           -4 -3 -2.5 -2 0 2 2.5 3 4 
#           -5 -4 -3 -2 0 2 3 4 5 
#           -3.5 -2.5 -1.5 -0.5 0.5 1.5 2.5 3.5 
#           -5 -4 -3 -2 -1 0 1 2 3 4 5 
#           -4.5 -3.5 -2.5 -1.5 -0.5 0.5 1.5 2.5 3.5 4.5 
#           -5 -4 -3 -2 -1 0 1 2 3 4 5 
#           -3.5 -2.5 -1.5 -0.5 0.5 1.5 2.5 3.5 
#           -5 -4 -3 -2 0 2 3 4 5 
#           -4.5 -3.5 -2.5 -1.5 -0.5 0.5 1.5 2.5 3.5 4.5 
#           -5 -4 -3 -2 -1 0 1 2 3 4 5 
#           -4.5 -3 -0.65 0.65 3 4.5 
#           -5.5 -4 -3 -2 -1 0 1 2 3 4 5.5 
#           -6.5 -4 -1.5 0 1.5 4 6.5 
#           1.5 1 0 -1 -1.5"""
# 
# lines_splitted = [line.split() for line in x.split('\n')]
# 
# names = """
# 'Fp1','AFp3h','Fpz','AFp4h','Fp2', ...
#        'AF7','AFF5h','AF3','AFF1','AFz','AFF2','Af4','AFF6h','AF8', ...
#        'F7','F5','F3','F1','Fz','F2','F4','F6','F8', ...
#        'FFT7h','FFC5h','FFC3h','FFC1h','FFC2h','FFC4h','FFC6h','FFT8h', ...
#        'FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10', ...
#        'FTT9h','FTT7h','FCC5h','FCC3h','FCC1h','FCC2h','FCC4h','FCC6h','FTT8h','FTT10h', ...
#        'M1','T7','C5','C3','C1','Cz','C2','C4','C6','T8','M2', ...
#        'TTP7h','CCP5h','CCP3h','CCP1h','CCP2h','CCP4h','CCP6h','TTP8h', ...
#        'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8', ...
#        'TPP9h','TPP7h','CPP5h','CPP3h','CPP1h','CPP2h','CPP4h','CPP6h','TPP8h','TPP10h', ...
#        'P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10', ...
#        'PPO9h','PPO5h','PPO1','PPO2','PPO6h','PPO10h', ...
#        'PO9','PO7','PO5','PO3','PO1','POz','PO2','PO4','PO6','PO8','PO10', ...
#        'POO9h','POO3h','O1','Oz','O2','POO4h','POO10h', ...
#        'I1','OI1h','Iz','OI2h','I2'
# """
# 
# nameslines = names.split('\n')
# 
# nameslines = nameslines[1:-1]
# sensor_name_rows = [filter(lambda s: s!= '', 
#                            l.replace('...', '').replace('\'', '').strip().split(',')) for l in nameslines]
# 
# for i in range(len(sensor_name_rows)):
#     assert (len(sensor_name_rows[i])) == (len(lines_splitted[i]))
# 
# ys = [ 3.5* np.ones((1,5)) ,
#         3* np.ones((1,9)) ,
#         2* np.ones((1,9)) ,
#       1.5* np.ones((1,8)) ,
#         1* np.ones((1,11)),
#       0.5* np.ones((1,10)),
#         0* np.ones((1,11)),
#      -0.5* np.ones((1,8)) ,
#        -1* np.ones((1,11)),
#      -1.5* np.ones((1,10)),
#        -2* np.ones((1,11)),
#      -2.5* np.ones((1,6)) ,
#        -3* np.ones((1,11)),
#      -3.5* np.ones((1,7)) ,
#        -5* np.ones((1,5))]
# 
# x_flat = [float(x_str) for l in lines_splitted for x_str in l]
# y_flat = [y for yrow in ys for y in yrow[0]] # need the [0] to unpack 2d row
# names_flat = [sensor_name for sensor_name_row in sensor_name_rows for sensor_name in sensor_name_row]
# 
# for i in range(len(names_flat)):
#     print("('{:s}', ({:.3f}, {:.3f})),".format(names_flat[i], x_flat[i], y_flat[i]))





CHANNEL_10_20 = (
    ('Fp1', (-4.000, 3.500)),
    ('AFp3h', (-1.500, 3.500)),
    ('Fpz', (0.000, 3.500)),
    ('AFp4h', (1.500, 3.500)),
    ('Fp2', (4.000, 3.500)),
    ('AF7', (-4.000, 3.000)),
    ('AFF5h', (-3.000, 3.000)),
    ('AF3', (-2.500, 3.000)),
    ('AFF1', (-2.000, 3.000)),
    ('AFz', (0.000, 3.000)),
    ('AFF2', (2.000, 3.000)),
    ('Af4', (2.500, 3.000)),
    ('AFF6h', (3.000, 3.000)),
    ('AF8', (4.000, 3.000)),
    ('F7', (-5.000, 2.000)),
    ('F5', (-4.000, 2.000)),
    ('F3', (-3.000, 2.000)),
    ('F1', (-2.000, 2.000)),
    ('Fz', (0.000, 2.000)),
    ('F2', (2.000, 2.000)),
    ('F4', (3.000, 2.000)),
    ('F6', (4.000, 2.000)),
    ('F8', (5.000, 2.000)),
    ('FFT7h', (-3.500, 1.500)),
    ('FFC5h', (-2.500, 1.500)),
    ('FFC3h', (-1.500, 1.500)),
    ('FFC1h', (-0.500, 1.500)),
    ('FFC2h', (0.500, 1.500)),
    ('FFC4h', (1.500, 1.500)),
    ('FFC6h', (2.500, 1.500)),
    ('FFT8h', (3.500, 1.500)),
    ('FT9', (-5.000, 1.000)),
    ('FT7', (-4.000, 1.000)),
    ('FC5', (-3.000, 1.000)),
    ('FC3', (-2.000, 1.000)),
    ('FC1', (-1.000, 1.000)),
    ('FCz', (0.000, 1.000)),
    ('FC2', (1.000, 1.000)),
    ('FC4', (2.000, 1.000)),
    ('FC6', (3.000, 1.000)),
    ('FT8', (4.000, 1.000)),
    ('FT10', (5.000, 1.000)),
    ('FTT9h', (-4.500, 0.500)),
    ('FTT7h', (-3.500, 0.500)),
    ('FCC5h', (-2.500, 0.500)),
    ('FCC3h', (-1.500, 0.500)),
    ('FCC1h', (-0.500, 0.500)),
    ('FCC2h', (0.500, 0.500)),
    ('FCC4h', (1.500, 0.500)),
    ('FCC6h', (2.500, 0.500)),
    ('FTT8h', (3.500, 0.500)),
    ('FTT10h', (4.500, 0.500)),
    ('M1', (-5.000, 0.000)),
    ('T7', (-4.000, 0.000)),
    ('C5', (-3.000, 0.000)),
    ('C3', (-2.000, 0.000)),
    ('C1', (-1.000, 0.000)),
    ('Cz', (0.000, 0.000)),
    ('C2', (1.000, 0.000)),
    ('C4', (2.000, 0.000)),
    ('C6', (3.000, 0.000)),
    ('T8', (4.000, 0.000)),
    ('M2', (5.000, 0.000)),
    ('TTP7h', (-3.500, -0.500)),
    ('CCP5h', (-2.500, -0.500)),
    ('CCP3h', (-1.500, -0.500)),
    ('CCP1h', (-0.500, -0.500)),
    ('CCP2h', (0.500, -0.500)),
    ('CCP4h', (1.500, -0.500)),
    ('CCP6h', (2.500, -0.500)),
    ('TTP8h', (3.500, -0.500)),
    ('TP7', (-5.000, -1.000)),
    ('CP5', (-4.000, -1.000)),
    ('CP3', (-3.000, -1.000)),
    ('CP1', (-2.000, -1.000)),
    ('CPz', (0.000, -1.000)),
    ('CP2', (2.000, -1.000)),
    ('CP4', (3.000, -1.000)),
    ('CP6', (4.000, -1.000)),
    ('TP8', (5.000, -1.000)),
    ('TPP9h', (-4.500, -1.000)),
    ('TPP7h', (-3.500, -1.000)),
    ('CPP5h', (-2.500, -1.500)),
    ('CPP3h', (-1.500, -1.500)),
    ('CPP1h', (-0.500, -1.500)),
    ('CPP2h', (0.500, -1.500)),
    ('CPP4h', (1.500, -1.500)),
    ('CPP6h', (2.500, -1.500)),
    ('TPP8h', (3.500, -1.500)),
    ('TPP10h', (4.500, -1.500)),
    ('P9', (-5.000, -1.500)),
    ('P7', (-4.000, -1.500)),
    ('P5', (-3.000, -2.000)),
    ('P3', (-2.000, -2.000)),
    ('P1', (-1.000, -2.000)),
    ('Pz', (0.000, -2.000)),
    ('P2', (1.000, -2.000)),
    ('P4', (2.000, -2.000)),
    ('P6', (3.000, -2.000)),
    ('P8', (4.000, -2.000)),
    ('P10', (5.000, -2.000)),
    ('PPO9h', (-4.500, -2.000)),
    ('PPO5h', (-3.000, -2.000)),
    ('PPO1', (-0.650, -2.500)),
    ('PPO2', (0.650, -2.500)),
    ('PPO6h', (3.000, -2.500)),
    ('PPO10h', (4.500, -2.500)),
    ('PO9', (-5.500, -2.500)),
    ('PO7', (-4.000, -2.500)),
    ('PO5', (-3.000, -3.000)),
    ('PO3', (-2.000, -3.000)),
    ('PO1', (-1.000, -3.000)),
    ('POz', (0.000, -3.000)),
    ('PO2', (1.000, -3.000)),
    ('PO4', (2.000, -3.000)),
    ('PO6', (3.000, -3.000)),
    ('PO8', (4.000, -3.000)),
    ('PO10', (5.500, -3.000)),
    ('POO9h', (-6.500, -3.000)),
    ('POO3h', (-4.000, -3.000)),
    ('O1', (-1.500, -3.500)),
    ('Oz', (0.000, -3.500)),
    ('O2', (1.500, -3.500)),
    ('POO4h', (4.000, -3.500)),
    ('POO10h', (6.500, -3.500)),
    ('I1', (1.500, -3.500)),
    ('OI1h', (1.000, -3.500)),
    ('Iz', (0.000, -5.000)),
    ('OI2h', (-1.000, -5.000)),
    ('I2', (-1.500, -5.000))
    )

def get_channelpos(channame, chan_pos_list=CHANNEL_10_20):
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

    >>> plot.get_channelpos('C2')
    (0.1720792096741632, 0.0)
    >>> # the channels are case insensitive
    >>> plot.get_channelpos('c2')
    (0.1720792096741632, 0.0)
    >>> # lookup for an invalid channel
    >>> plot.get_channelpos('foo')
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
    ax=None, annotate=False, vmin=None, vmax=None, colormap=None,
    chan_pos_list=CHANNEL_10_20):
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
    # interplolate the in-between values
    xx = np.linspace(min(x), max(x), 500)
    yy = np.linspace(min(y), max(y), 500)
    xx, yy = np.meshgrid(xx, yy)
    f = interpolate.LinearNDInterpolator(list(zip(x, y)), z)
    zz = f(xx, yy)
    # draw the contour map
    ax.contourf(xx, yy, zz, 20, vmin=vmin, vmax=vmax, cmap=colormap)
    # try also removing contour maybe...
    #ax.contour(xx, yy, zz, 5, colors="k", vmin=vmin, vmax=vmax, linewidths=.1)
    # paint the head
    ax.add_artist(plt.Circle((0, 0), 1, linestyle='solid', linewidth=2, fill=False))
    # add a nose
    ax.plot([-0.1, 0, 0.1], [1, 1.1, 1], 'k-')
    # add markers at channels positions
    #ax.plot(x, y, 'k+')
    # set the axes limits, so the figure is centered on the scalp
    ax.set_ylim([-1.05, 1.15])
    ax.set_xlim([-1.15, 1.15])
    # hide the frame and axes
    # hiding the axes might be too much, as this will also hide the x/y
    # labels :/
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # draw the channel names
    if annotate:
        for i in zip(channels, list(zip(x, y))):
            ax.annotate(" " + i[0], i[1],horizontalalignment="center")
    ax.set_aspect(1)
    #plt.sci(ctr) #TODELAY: find out why this is necessary?
    return ax