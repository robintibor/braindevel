
from gevent import socket
import gevent
import signal
gevent.signal(signal.SIGQUIT, gevent.kill)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", 1234))

while True:
    s.send("hi\n")
    socket_file = s.makefile('rb', 1024)
    print socket_file.readline()
    gevent.sleep(1)