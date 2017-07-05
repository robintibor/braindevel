
from gevent import socket
import gevent
import signal
import numpy as np
from numpy.random import RandomState
import sys


if len(sys.argv) > 1:
    port = int(sys.argv[1])
else:
    port = 7987
    
print("Port", port)

gevent.signal(signal.SIGQUIT, gevent.kill)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", port))

n_chans = 65 # 64 + 1 marker :)
n_samples = 50
s.send("Fp1 Fpz Fp2 AF7 AF3 AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7 FC5 "
    "FC3 FC1 FCz FC2 FC4 FC6 FT8 M1 T7 C5 C3 C1 Cz C2 C4 C6 T8 M2 TP7 "
    "CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO5 "
    "PO3 POz PO4 PO6 PO8 O1 Oz O2 marker\n")
s.send(np.array([n_chans], dtype=np.int32).tobytes())
s.send(np.array([n_samples], dtype=np.int32).tobytes())
rng = RandomState(874363)
cur_marker = 0
i_sample_in_break_or_trial = 0
while True:
    arr = rng.rand(n_chans,n_samples).astype(np.float32)
    # Set marker channel
    arr[-1] = cur_marker
    s.send(arr.tobytes(order='F'))
    # change marker roughly every 500 samples
    if i_sample_in_break_or_trial > 500:
        if cur_marker == 0:
            cur_marker = rng.randint(1,6)
        else:
            cur_marker = 0
        i_sample_in_break_or_trial = 0
    else:
        i_sample_in_break_or_trial += n_samples
    # make sleep time with realistic sampling rate
    gevent.sleep((1 / 500.0) * n_samples)
    
