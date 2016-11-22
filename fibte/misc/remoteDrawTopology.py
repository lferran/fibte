import pickle
import socket
from networkx import nx
import subprocess
import struct
import time
import matplotlib

matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

import threading
import Queue
import os


def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = ''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


class RemoteDrawTopology(object):
    def __init__(self, listeningPort=5010, k=4, remote=False):
        self.k = k
        self.queue = Queue.Queue(0)
        self.sock = ""
        self.connections = []

        # We always listen on local machine at specified port
        self.listeningPort = listeningPort
        self.listeningIp = '127.0.0.1'

        # Is remotePlot in local machine?
        self.remote = remote
        if self.remote:
            # Parameters for remote ssh tunner to eth server
            self.userName = 'lferran'
            self.serverName = 'pc-10327.ethz.ch'
            self.openPort = 5010  # Should be changed for what Ahmed tells me

        else:
            # Parameters for ssh tunnel within local VM
            self.userName = 'root'
            self.serverName = '127.0.0.1'
            self.openPort = 2222

        print("RemoteDrawTopology(host:{0}, port:{1})".format(self.serverName, self.listeningPort))

    def run(self):
        try:
            # Start thread that checks connectivity to the server
            p = threading.Thread(target=self.handlesTunnelConnection, args=())
            p.setDaemon(True)
            p.start()

            # Start a thread that handles TCP connections
            self.serverThread()

            # Start to plot the graph
            self.plotsGraph(self.k)

        except KeyboardInterrupt:
            print("KeyboardInterrupt: plot stopped")
            pass

    def handlesTunnelConnection(self):
        # First thing to do is to clear the server
        self.clearReverseSSH()

        # Setup reverse SSH tunnel
        self.reverseSSH()

        connected = True
        while True:
            time.sleep(2)
            status = self.checkInternetOn()
            #print status, "current port:", self.listeningPort
            if status:
                if not (connected):
                    connected = True
                    self.clearReverseSSH()
                    time.sleep(5)
                    self.reverseSSH()
            else:
                print("Internet connection not available")
                if connected:
                    print "Disconnecting"
                    connected = False

    def checkInternetOn(self):
        response = os.system("ping -c 1 " + self.serverName + " > /dev/null")
        if response == 0:
            return True
        else:
            return False

    def reverseSSH(self):
        """
        - Since we have set a port forwarding to access the VM, -p is needed
        - Parameter -f is so that it runs on the background
        - Parameter -R is because it is a remote port forwarding
        - Parameter -N
        - SSH will go into the VM (ferran@pc-17321.ethz.ch or root@vagrant)
        - Will setup a port forwarding such that all that you send at listeningPort
          inside the VM, will be received at the local machine at the same port
        """
        command = "ssh -p {3} -f -N -R {0}:localhost:{0} {1}@{2}".format(self.listeningPort, self.userName,
                                                                         self.serverName, self.openPort)
        print("Trying to setup SSH tunnel: {0}".format(command))
        subprocess.call(command, shell=True)

    def clearReverseSSH(self):
        """"""
        # We kill the previous ssh tunnel command locally
        command = "kill -9 $(ps aux | grep 'ssh -p {0} -f -N -R {1}' ".format(self.openPort,self.listeningPort)
        command +=  "| awk '{print $2}')"
        subprocess.call(command, shell=True)

        # Then we will clear the port number
        command = "ssh -p {3} {1}@{2} 'fuser -k -n tcp {0}'".format(self.listeningPort,self.userName,
                                                                    self.serverName,self.openPort)
        subprocess.call(command, shell=True)
        for connection in self.connections:
            connection.close()
        self.connections = []

    def serverThread(self):
        # Start a thread that handles TCP connections
        self.serverProcess = threading.Thread(target=self.serverStart, args=())
        self.serverProcess.setDaemon(True)
        self.serverProcess.start()

    def serverStart(self):
        """"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind((self.listeningIp, self.listeningPort))
            self.sock.listen(5)
        except Exception as e:
            print("Couldn't bind socket: {0}:{1}".format(self.listeningIp, self.listeningPort))
            print("Error: {0}".format(e))
        else:
            print("Server lisitening for connections on: {0}:{1}".format(self.listeningIp, self.listeningPort))

        while True:
            self.conn, addr = self.sock.accept()
            self.connections.append(self.conn)
            print("A new connection received")

            p = threading.Thread(target=self.handleConnection, args=(self.conn, self.queue,))
            p.setDaemon(True)
            p.start()

    def handleConnection(self, conn, queue):
        try:
            # Get all the elements to start
            self.receiveTopology()
            self.receivePosition()
            while True:
                msg = pickle.loads(recv_msg(conn))
                queue.put(msg)
        except:
            print("Error on handling connection {0}".format(conn))
            conn.close()

    def receiveTopology(self):
        """"""
        # waits for the topology to arrive
        self.topology = pickle.loads(recv_msg(self.conn))
        self.routers = [x for x in self.topology.node if self.topology.node[x]['type'] == "router"]
        self.hosts = [x for x in self.topology.node if self.topology.node[x]['type'] == "host"]

    def receivePosition(self):
        """Receive the position of the nodes in the graph"""
        self.pos = pickle.loads(recv_msg(self.conn))

    def plotsGraph(self, k):
        plt.ion()
        fig, ax = plt.subplots()

        # nx.draw(g,arrows = False,width=1.5,pos=pos, node_shape = 'o', node_color = 'b')
        plt.tight_layout()
        plt.show(block=False)

        plt.xlim([0, k*10])

        ONEYEAR = 365 * 24 * 3600
        while True:
            # Read new link_loads
            link_loads = self.queue.get(timeout=ONEYEAR)
            while not self.queue.empty():
                link_loads = self.queue.get(timeout=ONEYEAR)

            # weights = {x:y for x,y in link_loads.items() if all("sw" not in e for e in x)}
            weights = link_loads
            tt = time.time()

#            import ipdb; ipdb.set_trace()

            # Draw nodes and edges
            nx.draw_networkx_nodes(self.topology, ax=None, nodelist=self.hosts, pos=self.pos, node_shape='o',
                                   node_color='r')
            nx.draw_networkx_nodes(self.topology, ax=None, nodelist=self.routers, pos=self.pos, node_shape='s',
                                   node_color='b')
            nx.draw_networkx_edges(self.topology, ax=None, width=1.5, pos=self.pos)

            # Draw edges labels - colored depending on current load
            red_labels = {x: y for x, y in link_loads.items() if y > 0.75}
            orange_lables = {x: y for x, y in link_loads.items() if y >= 0.5 and y < 0.75}
            green_labels = {x: y for x, y in link_loads.items() if y >= 0.25 and y < 0.5}
            blue_labels = {x: y for x, y in link_loads.items() if y < 0.25}

            nx.draw_networkx_edge_labels(self.topology, self.pos, ax=None, edge_labels=red_labels, label_pos=0.15,
                                         font_size=10, font_color="r", font_weight='bold')
            nx.draw_networkx_edge_labels(self.topology, self.pos, ax=None, edge_labels=orange_lables, label_pos=0.15,
                                         font_size=10, font_color="orange", font_weight='bold')
            nx.draw_networkx_edge_labels(self.topology, self.pos, ax=None, edge_labels=green_labels, label_pos=0.15,
                                         font_size=10, font_color="g", font_weight='bold')
            nx.draw_networkx_edge_labels(self.topology, self.pos, ax=None, edge_labels=blue_labels, label_pos=0.15,
                                         font_size=10, font_color="b", font_weight='bold')
            print time.time() - tt

            # plt.show()
            # fig.canvas.update()
            # fig.canvas.flush_events()
            tt = time.time()

            fig.canvas.draw()
            fig.canvas.flush_events()
            ax.cla()
            plt.show(block=False)
            plt.tight_layout()
            #plt.xlim([-5, 117])
            plt.xlim([-5, k*10])

            print "drawing", time.time() - tt
            time.sleep(1e-6)  # unnecessary, but useful

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k',
                        help='Fat-Tree parameter',
                        type=int,
                        default=4)

    parser.add_argument('-p', '--port',
                        help='Port at which the remoteDraw is listening',
                        type=int,
                        default=5010)

    parser.add_argument('-r', '--remote',
                        help='Is the network running in the remote server? (ETH)',
                        type=bool,
                        action='store_true',
                        default=False)

    args = parser.parse_args()
    RemoteDrawTopology(listeningPort=args.port, k=args.k, remote=args.remote).run()
