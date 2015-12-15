import gevent
import gevent.server
import signal
gevent.signal(signal.SIGQUIT, gevent.kill)

def handle(socket, address):
    
    # using a makefile because we want to use readline()
    rfileobj = socket.makefile(mode='rb')
    while True:
        line = rfileobj.readline()
        if not line:
            print("client disconnected")
            break
        if line.strip().lower() == b'quit':
            print("client quit")
            break
        socket.sendall(line)
        print("echoed %r" % line)
    rfileobj.close()
    
hostname = ''
server = gevent.server.StreamServer((hostname, 1234), handle)
print("Starting server")
server.start()
print("Started server")
server.serve_forever()