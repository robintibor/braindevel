
import matplotlib
from braindecode.online.ring_buffer import RingBuffer
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import sys
import argparse
import numpy as np

# prevent seaborn import
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
    figure, axes = plt.subplots(num_sensors, sharex=True, sharey=sharey,
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
    max_ylim = np.max(np.abs(plt.ylim()))
    plt.ylim(-max_ylim, max_ylim)
    figure.subplots_adjust(hspace=0)
    return figure

class LivePlot:
    def showLivePlots(self, intervalInSec, durationInSec):
        self._plotIntervalInSec = intervalInSec
        self._totalDurationInSec = durationInSec
        self._initPlots(self._sensor_names)
        self._continouslyUpdatePlots()
        
    def _initPlots(self, sensor_names):
        self.n_samples = 0
        self.i_last_plot = 0
        self.i_last_range_update = 0
        plt.ion()
        plt.show()
        n_sensors = len(sensor_names)
        self._sensor_names = sensor_names
        self._data = dict()
        for sensor_name in sensor_names:
            self._data[sensor_name] = RingBuffer(np.sin(np.linspace(0,10,1000)))
        self.fig = plot_sensor_signals(np.outer(np.ones(n_sensors), 
            np.sin(np.linspace(0,10,1000))),
            self._sensor_names, figsize=(12,12))
        self.fig.suptitle("EEG Sensors")

        
    def accept_samples(self, samples):
        """Samples should be in time x chan format"""
        for i_sensor, sensor_name in enumerate(self._sensor_names):
            self._data[sensor_name].extend(samples[:,i_sensor])
        self.n_samples += len(samples)
        if self.n_samples - self.i_last_range_update > 500:
            self._setRangeOfAllPlotsToMaxMin()
            self.i_last_range_update = self.n_samples
        if self.n_samples - self.i_last_plot > 250:
            self._updateAllPlots()
            self.i_last_plot = self.n_samples
    
        
    def _updateAllPlots(self):
        for sensor_name in self._sensor_names:
            self._updatePlot(sensor_name, self._data[sensor_name])
        plt.pause(0.001)

    def _updatePlot(self, sensor_name, new_data):
        ax = self.fig.axes[self._sensor_names.index(sensor_name)]
        line = ax.lines[0]
        line.set_ydata(new_data)
    
    def _continouslyUpdatePlots(self):
        self._lastRangeUpdate = 0
        self._setRangeOfAllPlotsToMaxMin()
        i = 0
        while True:
            self._collectPackets()
            if i % 10 == 0:
                self._updateRangeOfPlots()
                self._updateAllPlots()


    def _setRangeOfAllPlotsToMaxMin(self):
        # find max and min deviation from mean
        totalMax = 0
        totalMin = sys.maxint
        sensorMeans = {}
        for sensor_name in self._sensor_names:
            sensorData = self._data[sensor_name]
            sensorMean = int(sum(sensorData) / len(sensorData))
            sensorMinimum = min(sensorData) - sensorMean
            sensorMaximum = max(sensorData) - sensorMean
            sensorMeans[sensor_name] = sensorMean
            if (sensorMinimum < totalMin):
                totalMin = sensorMinimum
            if (sensorMaximum > totalMax):
                totalMax = sensorMaximum
        for sensor_name in self._sensor_names:
            ax = self.fig.axes[self._sensor_names.index(sensor_name)]
            padding = 0.1
            ax.set_ylim(sensorMeans[sensor_name] + totalMin - padding,
              sensorMeans[sensor_name] + totalMax + padding
                )
        

def parseCommandLineArguments():
        parser = argparse.ArgumentParser(description='Live Plotting all the sensors :)')
        parser.add_argument('--debug', action="store_true", help="take dummy epoc headset")
        parser.add_argument('--duration', type=int, default=sys.maxint, help='how long to run plotting (in sec)')
        parser.add_argument('--interval', type=int, default=5, help='how much data to show in one plot (in sec)')
        return parser.parse_args()

if __name__ == '__main__':
    args = parseCommandLineArguments()
        
    livePlot = LivePlot()
    livePlot.showLivePlots(args.interval, args.duration)
