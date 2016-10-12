import socket
import os
import errno
import abc
#from logger import log
#import logging
import struct
import threading

from fibte import CFG
tmp_files = CFG.get("DEFAULT","tmp_files")

RECV_BUFFER_SIZE = 1024
MAX_CONN = 20

def send_msg(sock, msg):
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    """
    Read message length and unpack it into an integer

    :param sock: socket to read from
    :return: integer
    """
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]

    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    """
    Helper function to recv n bytes or return None if EOF is hit

    :param sock: socket to read from
    :param n: n bytes
    :return: return received bytes or None if EOF is hit
    """
    data = ''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# Base classes for unix sockets

class UnixClient(object):
    def __init__(self, server_address_base=tmp_files + "flowServer_{0}"):
        self.server_address_base = server_address_base
        self.sock = self.createSocket()

    @abc.abstractmethod
    def createSocket(self):
        """"""

    @abc.abstractmethod
    def send(self, msg, server):
        """"""

    def close(self):
        """Close socket"""
        self.sock.close()

class UnixServer(object):
    def __init__(self, address):
        self.server_address = address
        try:
            os.unlink(self.server_address)
        except OSError:
            if os.path.exists(self.server_address):
                raise

        # Create new socket
        self.sock = self.createSocket()

        # Bind the socket to address. The socket must not already be bound.
        self.sock.bind(self.server_address)

    @abc.abstractmethod
    def createSocket(self):
        """"""

    def receive(self):
        return self.sock.recv(RECV_BUFFER_SIZE)

    def close(self):
        self.sock.close()
        os.remove(self.server_address)

# TCP sockets

class UnixClientTCP(UnixClient):
    def __init__(self, *args, **kwargs):
        super(UnixClientTCP, self).__init__(*args, **kwargs)

    def createSocket(self):
        """Create socket"""
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    def send(self, msg, server):
        """
        Send msg to UnixServerTCP
        """
        try:
            self.createSocket()
            self.sock.connect(self.server_address_base.format(server))
            send_msg(self.sock,msg)
            self.close()

        except socket.error as serr:
            if serr.errno != errno.ECONNREFUSED:
                raise serr
            else:
                print serr
                print "Server {0} could not be reached".format(server)

class UnixClientUDP(UnixClient):
    def __init__(self, *args, **kwargs):
        super(UnixClientUDP, self).__init__(*args, **kwargs)

    def createSocket(self):
        """Creates a new socket"""
        return socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    def send(self, m, server):
        try:
            self.sock.sendto(m, self.server_address_base.format(server))
        except socket.error as serr:
            print serr
            print "Server {0} could not be reached".format(self.server_address_base.format(server))

# UDP sockets

class UnixServerTCP(UnixServer):
    def __init__(self, queue, *args, **kwargs):
        super(UnixServerTCP, self).__init__(*args, **kwargs)
        self.queue = queue

        # Listen for connectoins made to socket (max 5)
        self.sock.listen(MAX_CONN)

    def createSocket(self):
        return socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    def handleConnection(self, conn, queue):
        # Receives whole message, closes the connection
        # and puts it in a queue
        message = recv_msg(conn)
        conn.close()
        queue.put(message)

    def run(self):
        while True:
            # Accept a connection.
            #   conn is a new socket object usable to send and receive data.
            #   addr is the address bound to the socket on the other end.
            conn, addr = self.sock.accept()
            p = threading.Thread(target = self.handleConnection, args = (conn, self.queue))
            p.setDaemon(True)
            p.start()

class UnixServerUDP(UnixServer):

    def __init__(self, *args, **kwargs):
        super(UnixServerUDP, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def createSocket(self):
        return socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)


