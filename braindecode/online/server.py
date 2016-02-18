#!/usr/bin/env python
from braindecode.online.live_plot import LivePlot
import gevent.server
import signal
import numpy as np
import logging
import lasagne
from pylearn2.config import yaml_parse
from braindecode.online.coordinator import OnlineCoordinator
from gevent import socket
from braindecode.online.model import OnlineModel
from braindecode.online.data_processor import StandardizeProcessor
import argparse
import h5py
import datetime
import os.path
import sys
import gevent.select
from braindecode.experiments.experiment_runner import create_experiment
log = logging.getLogger(__name__)

class PredictionServer(gevent.server.StreamServer):
    def __init__(self, listener, coordinator, ui_hostname, ui_port, 
        plot_sensors, save_data,
            handle=None, backlog=None, spawn='default', **ssl_args):
        self.coordinator = coordinator
        self.ui_hostname = ui_hostname
        self.ui_port = ui_port
        self.plot_sensors = plot_sensors
        self.save_data = save_data
        super(PredictionServer, self).__init__(listener, handle=handle, spawn=spawn)


    def handle(self, in_socket, address):
        # Connect to UI Server
        log.info('New connection from {:s}!'.format(str(address)))
        ui_socket = self.connect_to_ui_server()
        log.info("Connected to UI Server")
        
        # Receive Header
        chan_names, n_rows, n_cols = self.receive_header(in_socket)
        n_numbers = n_rows * n_cols
        n_bytes = n_numbers * 4 # float32
        log.info("Numbers in total:  {:d}".format(n_numbers))
        
        # Possibly plot
        if self.plot_sensors:
            self.plot_sensors_until_enter_press(chan_names, in_socket, n_bytes,
            n_rows, n_cols)
        

        self.make_predictions_and_save_data(chan_names, n_rows, n_cols, n_bytes,
        in_socket, ui_socket)
        self.stop()

    def connect_to_ui_server(self):
        ui_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ui_socket.connect((self.ui_hostname, self.ui_port))
        return ui_socket
        
    
    def read_until_bytes_received(self, socket, n_bytes):
        array = ''
        while len(array) < n_bytes:
            array += socket.recv(n_bytes - len(array))
        return array

    def read_until_bytes_received_or_enter_pressed(self, socket, n_bytes):
        enter_pressed = False
        array = ''
        while len(array) < n_bytes and not enter_pressed:
            array += socket.recv(n_bytes - len(array))
            # check if enter is pressed
            i,o,e = gevent.select.select([sys.stdin],[],[],0.0001)
            for s in i:
                if s == sys.stdin:
                    _ = sys.stdin.readline()
                    enter_pressed = True
        if enter_pressed:
            return None
        else:
            return array
    
    def receive_header(self, in_socket):
        chan_names_line = '' + in_socket.recv(1)
        while chan_names_line[-1] != '\n':
            chan_names_line += in_socket.recv(1)
        log.info("Chan names:\n{:s}".format(chan_names_line))
        chan_names = chan_names_line.replace('\n','').split(" ")
            
        assert np.array_equal(chan_names, ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3',
            'AFz', 'AF4', 'AF8', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
            'FC1', 'FCz', 'FC2', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1',
             'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz', 'marker']
            )
        n_rows = self.read_until_bytes_received(in_socket, 4)
        print "0", n_rows[0]
        print "1", n_rows[1]
        print "2", n_rows[2]
        print "3", n_rows[3]
        n_rows = np.fromstring(n_rows, dtype=np.int32)[0]
        log.info("Number of rows:    {:d}".format(n_rows))
        n_cols = self.read_until_bytes_received(in_socket, 4)
        n_cols = np.fromstring(n_cols, dtype=np.int32)[0]
        log.info("Number of columns: {:d}".format(n_cols))
        assert n_rows == len(chan_names)
        return chan_names, n_rows, n_cols

    def plot_sensors_until_enter_press(self, chan_names, in_socket, n_bytes,
            n_rows, n_cols):
        live_plot = LivePlot(plot_freq=150)
        live_plot.initPlots(chan_names)

        log.info("Starting Plot for plot")
        enter_pressed = False
        while not enter_pressed:
            array = ''
            while len(array) < n_bytes:
                array += in_socket.recv(n_bytes - len(array))
            array = np.fromstring(array, dtype=np.float32)
            array = array.reshape(n_rows, n_cols, order='F')
            live_plot.accept_samples(array.T)
            # check if enter is pressed
            i,o,e = gevent.select.select([sys.stdin],[],[],0.001)
            for s in i:
                if s == sys.stdin:
                    _ = sys.stdin.readline()
                    enter_pressed = True
        
        live_plot.close()
        log.info("Plot finished")

    def make_predictions_and_save_data(self, chan_names, n_rows, n_cols, n_bytes,
        in_socket, ui_socket):
        if self.save_data:
            data_saver = DataSaver(chan_names)
        self.coordinator.initialize(n_chans=n_rows - 1) # one is a marker chan(!)
        while True:
            array = self.read_until_bytes_received_or_enter_pressed(in_socket,
                n_bytes)
            if array is None:
                # enter was pressed! quit! :)
                break;
            array = np.fromstring(array, dtype=np.float32)
            array = array.reshape(n_rows, n_cols, order='F')
            sensor_samples = array[:-1,:]
            if self.save_data:
                data_saver.append_samples(array.T)
            # here now also supply y to data processor...
            self.coordinator.receive_samples(array.T)

            if self.coordinator.has_new_prediction():
                pred, i_sample = self.coordinator.pop_last_prediction_and_sample_ind()
                log.info("Prediction for sample {:d}:\n{:s}".format(
                    i_sample, pred))
                # +1 to convert 0-based to 1-based indexing
                ui_socket.sendall("{:d}\n".format(i_sample + 1))
                ui_socket.sendall("{:f} {:f} {:f} {:f}\n".format(*pred[0]))
        if self.save_data:
            data_saver.save()

class DataSaver(object):
    """ Remember and save data streamed during an online session."""
    def __init__(self, chan_names):
        self.chan_names = chan_names
        self.sample_blocks = []

    def append_samples(self, samples):
        """ Expects timexchan"""
        self.sample_blocks.append(samples)
        
    def save(self):
        # save with time as filename
        now = datetime.datetime.now()
        time_string = now.strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join('data/online/', time_string + '.hdf5')
        log.info("Saving to {:s}...".format(filename))
        all_samples = np.concatenate(self.sample_blocks).astype(np.float32)
        with h5py.File(filename, 'w') as out_file:
            dt = h5py.special_dtype(vlen=unicode)
            dset = out_file.create_dataset("chan_names", (len(self.chan_names),), dtype=dt)
            dset[:] = self.chan_names
            out_file.create_dataset("cnt_samples", data=all_samples)
        log.info("Done.")
        
def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Launch server for online decoding.
        Example: online/server.py --host 172.30.2.129 --port 30000"""
    )
    parser.add_argument('--host', action='store',
        default='172.30.2.129', help='Hostname/IP of the UI server')
    parser.add_argument('--port', action='store',
        default=30000, help='Port of the UI server')
    parser.add_argument('--modelfile', action='store',
        default='data/models/raw-net-512/3', 
        help='Basename of the modelfile')
    parser.add_argument('--noplot', action='store_true',
        help="Don't show plots of the sensors first...")
    parser.add_argument('--nosave', action='store_true',
        help="Don't save data...")
    args = parser.parse_args()
    return args

def main(ui_hostname, ui_port, base_name, plot_sensors, save_data):
    assert np.little_endian, "Should be in little endian"
    hostname = ''
    port = 1234
    params = np.load(base_name + '.npy')
    exp = create_experiment(base_name + '.yaml')
    model = exp.final_layer
    lasagne.layers.set_all_param_values(model, params)
    data_processor = StandardizeProcessor(factor_new=1e-3)
    online_model = OnlineModel(model)
    coordinator = OnlineCoordinator(data_processor, online_model, pred_freq=100)
    server = PredictionServer((hostname, port), coordinator=coordinator,
        ui_hostname=ui_hostname, ui_port=ui_port, plot_sensors=plot_sensors,
        save_data=save_data)
    log.setLevel("DEBUG")
    log.info("Starting server")
    server.start()
    log.info("Started server")
    server.serve_forever()

if __name__ == '__main__':
    logging.basicConfig()
    gevent.signal(signal.SIGQUIT, gevent.kill)
    args = parse_command_line_arguments()
    main(args.host, args.port, args.modelfile, not args.noplot, not args.nosave)
    
    