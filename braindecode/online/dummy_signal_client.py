
from gevent import socket
import gevent
import signal
import numpy as np
from numpy.random import RandomState

gevent.signal(signal.SIGQUIT, gevent.kill)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", 1234))

s.send(np.array([32], dtype=np.int32).tobytes())
s.send(np.array([50], dtype=np.int32).tobytes())
rng = RandomState(874363)
while True:
    arr = rng.rand(2,50).astype(np.float32)
    s.send(arr.tobytes(order='F'))
    gevent.sleep(0.002)