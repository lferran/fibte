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
    def __init__(self, listeningIp="192.168.33.1", listeningPort=5010, k=4):
        self.listeningIp = listeningIp
        self.listeningPort = listeningPort

        self.k = k
        self.queue = Queue.Queue(0)
        self.sock = ""
        self.connections = []

    def serverThread(self):
        # start a thread that handles TCP connections
        self.serverProcess = threading.Thread(target=self.serverStart, args=())
        self.serverProcess.setDaemon(True)
        self.serverProcess.start()

    def run(self):
        try:
            # Start a thread that handles TCP connections
            self.serverThread()

            # Start to plot the graph
            self.plotsGraph(self.k)

        except KeyboardInterrupt:
            print("KeyboardInterrupt: plot stopped")
            pass

    def serverStart(self):
        """"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.listeningIp, self.listeningPort))
        self.sock.listen(5)
        print("Server lisitening for connections on: {0}:{1}".format(self.listeningIp, self.listeningPort))
        while True:
            self.conn, addr = self.sock.accept()
            self.connections.append(self.conn)
            print("A new connection received")

            self.handleConnection(self.conn, self.queue)
            #p = threading.Thread(target=self.handleConnection, args=(self.conn, self.queue,))
            #p.setDaemon(True)
            #p.start()

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

        # g = self.topology

        # routers = [x for x in g.node if g.node[x]['type']=="router"]
        # switches = [x for x in g.node if g.node[x]['type']=="switch"]

        # print routers

        # nx.draw(g,arrows = False,width=1.5,pos=pos, node_shape = 'o', node_color = 'b')
        plt.tight_layout()
        plt.show(block=False)

        plt.xlim([0, 120])

        ONEYEAR = 365 * 24 * 3600
        while True:
            # Read new link_loads
            link_loads = self.queue.get(timeout=ONEYEAR)
            while not self.queue.empty():
                link_loads = self.queue.get(timeout=ONEYEAR)

            import ipdb; ipdb.set_trace()

            # weights = {x:y for x,y in link_loads.items() if all("sw" not in e for e in x)}
            weights = link_loads
            tt = time.time()

            nx.draw_networkx_nodes(self.topology, ax=None, nodelist=self.hosts, pos=self.pos, node_shape='o',
                                   node_color='r')
            nx.draw_networkx_nodes(self.topology, ax=None, nodelist=self.routers, pos=self.pos, node_shape='s',
                                   node_color='b')
            nx.draw_networkx_edges(self.topology, ax=None, width=1.5, pos=self.pos)

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
            plt.xlim([-5, 117])

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
                        help='Port at which the remoteDrow is listening',
                        type=int,
                        default=5010)

    parser.add_argument('--ip',
                        help='Ip of the host',
                        type=str,
                        default="192.168.33.1")

    args = parser.parse_args()

    RemoteDrawTopology(listeningIp=args.ip, listeningPort=args.port, k=args.k).run()