#!/usr/bin/env python
import gevent.server
import signal
import numpy as np
import logging
import lasagne
from pylearn2.config import yaml_parse
from braindecode.online.predictor import OnlinePredictor
from gevent import socket
from braindecode.online.model import OnlineModel
from braindecode.online.data_processor import StandardizeProcessor
import argparse
log = logging.getLogger(__name__)

class PredictionServer(gevent.server.StreamServer):
    def __init__(self, listener, predictor, ui_hostname, ui_port, 
        handle=None, backlog=None, spawn='default',
        **ssl_args):
        self.predictor = predictor
        self.ui_hostname = ui_hostname
        self.ui_port = ui_port
        super(PredictionServer, self).__init__(listener, handle=handle, spawn=spawn)

    def handle(self, in_socket, address):
        log.info('New connection from {:s}!'.format(str(address)))
        ui_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ui_socket.connect((self.ui_hostname, self.ui_port))
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
        self.predictor.initialize(n_chans=n_rows)

        while True:
            array = ''
            while len(array) < n_bytes:
                array += in_socket.recv(n_bytes - len(array))
            array = np.fromstring(array, dtype=np.float32)
            array = array.reshape(n_rows, n_cols, order='F')
            self.predictor.receive_samples(array.T)
            if self.predictor.has_new_prediction():
                pred, i_sample = self.predictor.pop_last_prediction_and_sample_ind()
                log.info("Prediction for sample {:d}:\n{:s}".format(
                    i_sample, pred))
                ui_socket.sendall("{:d}\n".format(i_sample))
                ui_socket.sendall("{:f} {:f} {:f} {:f}\n".format(*pred[0]))
                


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Launch server for online decoding.
        Example: online/server.py --host 172.30.1.145 --port 30000"""
    )
    parser.add_argument('--host', action='store',
        default='172.30.1.145', help='Hostname/IP of the UI server')
    parser.add_argument('--port', action='store',
        default=30000, help='Port of the UI server')
    parser.add_argument('--modelfile', action='store',
        default='data/models/raw-net-500-fs/81', 
        help='Basename of the modelfile')
    args = parser.parse_args()
    return args

def main(ui_hostname, ui_port, base_name):
    hostname = ''
    port = 1234
    params = np.load(base_name + '.npy')
    train_dict = yaml_parse.load(open(base_name + '.yaml', 'r'))
    model = train_dict['layers'][-1]
    lasagne.layers.set_all_param_values(model, params)
    data_processor = StandardizeProcessor(factor_new=1e-4)
    online_model = OnlineModel(model)
    predictor = OnlinePredictor(data_processor, online_model, pred_freq=100)
    server = PredictionServer((hostname, port), predictor=predictor,
        ui_hostname=ui_hostname, ui_port=ui_port)
    log.setLevel("DEBUG")
    log.info("Starting server")
    server.start()
    log.info("Started server")
    server.serve_forever()

if __name__ == '__main__':
    logging.basicConfig()
    gevent.signal(signal.SIGQUIT, gevent.kill)
    args = parse_command_line_arguments()
    main(args.host, args.port, args.modelfile)
    
    