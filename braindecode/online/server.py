import gevent.server
import signal
import numpy as np
import logging
import lasagne
from pylearn2.config import yaml_parse
from braindecode.online.predictor import OnlinePredictor
from gevent import socket
log = logging.getLogger(__name__)

class PredictionServer(gevent.server.StreamServer):
    def __init__(self, listener, predictor, handle=None, backlog=None, spawn='default',
        **ssl_args):
        self.predictor = predictor
        super(PredictionServer, self).__init__(listener, handle=handle, spawn=spawn)

    def handle(self, in_socket, address):
        log.info('new connection from {:s}!'.format(str(address)))
        answer_host = '172.30.1.145'
        answer_port = 30000
        answer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        answer_socket.connect((answer_host, answer_port))
        assert np.little_endian, "Should be in little endian"
        n_rows = in_socket.recv(4)
        n_rows = np.fromstring(n_rows, dtype=np.int32)[0]
        log.info("num rows {:d}".format(n_rows))
        n_cols = in_socket.recv(4)
        n_cols = np.fromstring(n_cols, dtype=np.int32)[0]
        log.info("num cols: {:d}".format(n_cols))
        n_numbers = n_rows * n_cols
        n_bytes = n_numbers * 4 # float32
        log.info("n_numbers: {:d}".format(n_numbers))
        self.predictor.initialize(n_chans=n_rows)

        while True:
            array = ''
            while len(array) < n_bytes:
                array += in_socket.recv(n_bytes - len(array))
            array = np.fromstring(array, dtype=np.float32)
            array = array.reshape(n_rows, n_cols, order='F')
            self.predictor.receive_sample_block(array.T)
            if self.predictor.has_new_prediction():
                pred, i_sample = self.predictor.pop_last_prediction_and_sample_ind()
                log.info("Prediction for sample {:d}:\n{:s}".format(
                    i_sample, pred))

                answer_socket.sendall("{:d}\n".format(i_sample))#, dtype=np.int32).tobytes())
                answer_socket.sendall("{:f} {:f} {:f} {:f}\n".format(*pred[0]))
                
    
if __name__ == '__main__':
    logging.basicConfig()
    gevent.signal(signal.SIGQUIT, gevent.kill)
    hostname = ''
    port = 1234
    base_name = 'data/models/raw-net-500-fs/23'
    params = np.load(base_name + '.npy')
    train_dict = yaml_parse.load(open(base_name + '.yaml', 'r'))
    model = train_dict['layers'][-1]
    lasagne.layers.set_all_param_values(model, params)
    predictor = OnlinePredictor(model, prediction_frequency=100)
    server = PredictionServer((hostname, port), predictor=predictor)
    log.setLevel("DEBUG")
    log.info("Starting server")
    server.start()
    log.info("Started server")
    server.serve_forever()