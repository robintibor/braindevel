
from gevent import socket
import gevent
import signal
import numpy as np
gevent.signal(signal.SIGQUIT, gevent.kill)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", 1234))

s.send(np.array([2], dtype=np.int32).tobytes())
s.send(np.array([3], dtype=np.int32).tobytes())
while True:
    s.send(np.array([[0.0, 1.0, 2.0], [3.0,4.0,5.0]], dtype=np.float32).tobytes(order='F'))
    gevent.sleep(1)