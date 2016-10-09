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
            # start thread that checks connectivity to the server
            #p = threading.Thread(target=self.handlesTunnelConnection, args=())
            #p.setDaemon(True)
            #p.start()

            # start a thread that handles TCP connections
            self.serverThread()

            # self.clearReverseSSH()
            # self.reverseSSH()
            # self.serverStart()
            # self.receiveTopology()
            print "STARTED SERVER now i start GRAPH"
            self.plotsGraph(self.k)

        except KeyboardInterrupt:
            pass
            #self.clearReverseSSH()

    def serverStart(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.listeningIp, self.listeningPort))
        self.sock.listen(5)

        while True:
            self.conn, addr = self.sock.accept()
            self.connections.append(self.conn)
            print "received a connection"
            self.handleConnection(self.conn, self.queue)
#            p = threading.Thread(target=self.handleConnection, args=(self.conn, self.queue,))
#            p.setDaemon(True)
#            p.start()

    def handleConnection(self, conn, queue):
        try:
            # get all the elements to start
            self.receiveTopology()
            self.receivePosition()
            while True:
                msg = pickle.loads(recv_msg(conn))
                queue.put(msg)
        except:
            conn.close()

    def receiveTopology(self):

        # waits for the topology to arrive
        self.topology = pickle.loads(recv_msg(self.conn))
        import ipdb; ipdb.set_trace()

        self.routers = [x for x in self.topology.node if self.topology.node[x]['type'] == "router"]
        self.switches = [x for x in self.topology.node if self.topology.node[x]['type'] == "switch"]

    def receivePosition(self):
        self.pos = pickle.loads(recv_msg(self.conn))

    def reverseSSH(self):
        subprocess.call("ssh -p 3006 -f -N -R {0}:localhost:{0} edgar@pisco.ethz.ch".format(self.listeningPort),
                        shell=True)
        # subprocess.call("ssh -p 2222 -f -N -R {0}:localhost:{0} edgar@pc-10326.ethz.ch".format(self.listeningPort),shell=True)

    def clearReverseSSH(self):
        subprocess.call("kill -9 $(ps aux | grep 'ssh -p 3006 -f -N -R " + "{0}' ".format(
            self.listeningPort) + "| awk '{print $2}')", shell=True)
        # then we will clear the port number
        subprocess.call("ssh -p 3006 edgar@pisco.ethz.ch  'sudo fuser -k -n tcp {0}'".format(self.listeningPort),
                        shell=True)

        for connection in self.connections:
            connection.close()
        self.connections = []
        # subprocess.call("kill -9 $(ps aux | grep 'ssh -p 2222 -f -N' | awk '{print $2}')",shell=True)

    def handlesTunnelConnection(self):

        # first think to do is to clear the server
        #self.clearReverseSSH()
        #self.reverseSSH()
        connected = True

        while True:
            time.sleep(2)
            status = self.checkInternetOn("pisco.ethz.ch")
            print status, "current port:", self.listeningPort
            if status:
                if not (connected):
                    connected = True
                    self.clearReverseSSH()
                    time.sleep(5)
                    self.reverseSSH()

            else:
                if connected:
                    print "disconnecting"
                    connected = False

                    #

    def checkInternetOn(self, serverName="pisco.ethz.ch"):
        response = os.system("ping -c 1 " + serverName + " > /dev/null")
        if response == 0:
            return True
        else:
            return False

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
            # read new link_loads

            link_loads = self.queue.get(timeout=ONEYEAR)
            while not self.queue.empty():
                link_loads = self.queue.get(timeout=ONEYEAR)

            # weights = {x:y for x,y in link_loads.items() if all("sw" not in e for e in x)}
            weights = link_loads
            tt = time.time()

            nx.draw_networkx_nodes(self.topology, ax=None, nodelist=self.switches, pos=self.pos, node_shape='o',
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

    def plotsGraph2(self, k):

        plt.ion()
        fig = plt.figure()
        g = self.topology
        pos = pickle.loads(recv_msg(self.conn))

        nx.draw(g, arrows=False, width=1.5, pos=pos, node_shape='o', node_color='b')

        plt.show(block=False)

        while True:
            # read new link_loads
            link_loads = pickle.loads(recv_msg(self.conn))

            weights = {x: y for x, y in link_loads.items() if all("sw" not in e for e in x)}
            tt = time.time()
            nx.draw_networkx_edge_labels(g, pos, edge_labels=weights, label_pos=0.15, font_size=8, font_color="k",
                                         font_weight='bold')
            print time.time() - tt

            # plt.show()
            # fig.canvas.update()
            # fig.canvas.flush_events()
            plt.pause(0.0001)
            print time.time() - tt

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
                        default=8083)

    args = parser.parse_args()

    RemoteDrawTopology(args.port, args.k).run()