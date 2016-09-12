
from gevent import socket
import gevent
import signal
import numpy as np
from numpy.random import RandomState
import sys


if len(sys.argv) > 1:
    port = int(sys.argv[1])
else:
    port = 7986
    
print("Port", port)

gevent.signal(signal.SIGQUIT, gevent.kill)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", port))

n_chans = 65 # 64 + 1 marker :)
n_samples = 50
s.send("Fp1 Fpz Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 M1 T7 C3 Cz C4 T8 M2 CP5 "
    "CP1 CP2 CP6 P7 P3 Pz P4 P8 POz O1 Oz O2 AF7 AF3 AF4 AF8 F5 F1 F2 F6 FC3 " 
    "FCz FC4 C5 C1 C2 C6 CP3 CPz CP4 P5 P1 P2 P6 PO5 PO3 PO4 PO6 FT7 FT8 TP7 "
    "TP8 PO7 PO8 marker\n")
s.send(np.array([n_chans], dtype=np.int32).tobytes())
s.send(np.array([n_samples], dtype=np.int32).tobytes())
rng = RandomState(874363)
while True:
    arr = rng.rand(n_chans,n_samples).astype(np.float32)
    # transfor form 0..1 to 0,1,2,3,4
    arr[-1] = np.round(arr[-1] * 4)
    s.send(arr.tobytes(order='F'))
    # make sleep time with realistic sampling rate
    gevent.sleep((1 / 500.0) * 50)
