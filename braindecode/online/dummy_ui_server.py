import gevent.server
import signal

def handle(socket, address):
    print ("new connection")
    # using a makefile because we want to use readline()
    socket_file = socket.makefile()
    while True:
        i_sample = socket_file.readline()
        preds = socket_file.readline()
        print i_sample
        print preds
    
gevent.signal(signal.SIGQUIT, gevent.kill)
hostname = ''
port = 30000
server = gevent.server.StreamServer((hostname, port), handle)
print("Starting server on port {:d}".format(port))
server.start()
print("Started server")
server.serve_forever()