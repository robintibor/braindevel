
from gevent import socket
import gevent
import signal
import numpy as np
gevent.signal(signal.SIGQUIT, gevent.kill)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("172.30.1.145", 30000))
while True:
    s.sendall(np.array(5, dtype=np.int32).tobytes(order='F'))
    s.sendall(np.array([0.0,1.0,0.1,0.2]).astype(np.float32).tobytes(order='F'))
    #s.flush()
    gevent.sleep(0.1)