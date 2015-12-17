
from gevent import socket
import gevent
import signal
import numpy as np
from numpy.random import RandomState

gevent.signal(signal.SIGQUIT, gevent.kill)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", 1234))

n_chans = 32
n_samples = 50
s.send(np.array([n_chans], dtype=np.int32).tobytes())
s.send(np.array([n_samples], dtype=np.int32).tobytes())
rng = RandomState(874363)
while True:
    arr = rng.rand(n_chans,n_samples).astype(np.float32)
    s.send(arr.tobytes(order='F'))
    # make sleep time with realistic sampling rate
    gevent.sleep((1 / 500.0) * 50)
