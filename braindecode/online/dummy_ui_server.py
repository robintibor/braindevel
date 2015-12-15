import gevent.server
import signal
import numpy as np

def handle(socket, address):
    print ("new connection")
    # using a makefile because we want to use readline()
    while True:
        i_sample = socket.recv(8)
        i_sample = np.fromstring(i_sample, dtype=np.int64)
        preds = socket.recv(4 * 4)
        preds = np.fromstring(preds, dtype=np.float32)
        print i_sample
        print preds
    
gevent.signal(signal.SIGQUIT, gevent.kill)
hostname = ''
server = gevent.server.StreamServer((hostname, 30000), handle)
print("Starting server")
server.start()
print("Started server")
server.serve_forever()