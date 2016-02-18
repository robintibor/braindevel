
from gevent import socket
import gevent
import signal
import numpy as np
from numpy.random import RandomState

gevent.signal(signal.SIGQUIT, gevent.kill)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", 1234))

n_chans = 33 # 32 + 1 marker :)
n_samples = 50
s.send("Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 F5 F3 F1 Fz F2 F4 F6 FC1 FCz FC2 "
    "C3 C1 Cz C2 C4 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz marker\n")
s.send(np.array([n_chans], dtype=np.int32).tobytes())
s.send(np.array([n_samples], dtype=np.int32).tobytes())
rng = RandomState(874363)
while True:
    arr = rng.rand(n_chans,n_samples).astype(np.float32)
    s.send(arr.tobytes(order='F'))
    # make sleep time with realistic sampling rate
    gevent.sleep((1 / 500.0) * 50)
