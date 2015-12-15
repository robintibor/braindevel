import gevent.server
import signal
import numpy as np
import logging
log = logging.getLogger(__name__)

class PredictionServer(gevent.server.StreamServer):
    def __init__(self, listener, predictor, handle=None, backlog=None, spawn='default',
        **ssl_args):
        super(PredictionServer, self).__init__(listener, handle=handle, spawn=spawn)

    def handle(self, socket, address):
        log.info('new connection!')
        assert np.little_endian, "Should be in little endian"
        num_rows = socket.recv(4)
        num_rows = np.fromstring(num_rows, dtype=np.int32)[0]
        log.info("num rows {:d}".format(num_rows))
        num_cols = socket.recv(4)
        num_cols = np.fromstring(num_cols, dtype=np.int32)[0]
        log.info("num cols: {:d}".format(num_cols))
        n_numbers = num_rows * num_cols
        n_bytes = n_numbers * 4 # float32
        log.info("n_numbers: {:d}".format(n_numbers))
            
        while True:
            array = ''
            while len(array) < n_bytes:
                array += socket.recv(n_bytes - len(array))
                log.info("Received bytes: {:d}".format(len(array)))
            array = np.fromstring(array, dtype=np.float32)
            array = array.reshape(num_rows, num_cols, order='F')
            log.info("array")
            print(array)
    

logging.basicConfig()
gevent.signal(signal.SIGQUIT, gevent.kill)
hostname = ''
port = 1234
server = PredictionServer((hostname, port), None)
log.setLevel("DEBUG")
log.info("Starting server")
server.start()
log.info("Started server")
server.serve_forever()