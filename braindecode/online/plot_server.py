#!/usr/bin/env python
import gevent.server
import signal
import numpy as np
import logging
from pylearn2.config import yaml_parse
from braindecode.online.live_plot import LivePlot
log = logging.getLogger(__name__)

class PlotServer(gevent.server.StreamServer):
    def __init__(self, listener, sensor_names = None,
        handle=None, backlog=None, spawn='default',
        **ssl_args):
        self.sensor_names = sensor_names
        if self.sensor_names is None:
            self.sensor_names = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz',
                 'AF4', 'AF8', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
                 'FC1', 'FCz', 'FC2', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3',
                 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']

        super(PlotServer, self).__init__(listener, handle=handle, spawn=spawn)

    def handle(self, in_socket, address):
        log.info('New connection from {:s}!'.format(str(address)))
        assert np.little_endian, "Should be in little endian"
        n_rows = in_socket.recv(4)
        n_rows = np.fromstring(n_rows, dtype=np.int32)[0]
        log.info("Number of rows:    {:d}".format(n_rows))
        n_cols = in_socket.recv(4)
        n_cols = np.fromstring(n_cols, dtype=np.int32)[0]
        log.info("Number of columns: {:d}".format(n_cols))
        n_numbers = n_rows * n_cols
        n_bytes = n_numbers * 4 # float32
        log.info("Numbers in total:  {:d}".format(n_numbers))
        live_plot = LivePlot()
        live_plot._initPlots(self.sensor_names)

        while True:
            array = ''
            while len(array) < n_bytes:
                array += in_socket.recv(n_bytes - len(array))
            array = np.fromstring(array, dtype=np.float32)
            array = array.reshape(n_rows, n_cols, order='F')
            live_plot.accept_samples(array.T)
                



def main():
    hostname = ''
    port = 1234
    server = PlotServer((hostname, port))
    log.setLevel("DEBUG")
    log.info("Starting server")
    server.start()
    log.info("Started server")
    server.serve_forever()

if __name__ == '__main__':
    logging.basicConfig()
    gevent.signal(signal.SIGQUIT, gevent.kill)
    main()
    
    