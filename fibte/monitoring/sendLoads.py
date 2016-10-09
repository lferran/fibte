from fibte.misc.topology_graph import TopologyGraph
import json
import time
import Queue
from threading import Thread
import socket

try:
    import cPickle as pickle
except:
    import pickle
import struct

from fibte import tmp_files, db_topo
import os

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

class SendLoads(object):
    def __init__(self, remotePort=5010, k=4, time_interval=1):

        # Load topology
        self.topology = TopologyGraph(getIfindexes=True, db=os.path.join(tmp_files, db_topo))
        self.routers = self.topology.getRouters()
        self.edgeRouters = self.topology.getEdgeRouters()

        self.link_loads = self.topology.getEdgesUsageDictionary()

        self.queue = Queue.Queue(maxsize=0)
        self.remotePort = remotePort
        self.k = k

        self.time_interval = time_interval

    def startGrahpThread(self):

        graphPlotterThread = Thread(target=self.topology.networkGraph.plotGraphAnimated, args=(self.k, self.queue))
        graphPlotterThread.setDaemon(True)
        graphPlotterThread.start()

    def sendLoadToThread(self):

        self.queue.put(self.link_loads)

    def connectToRemoteServer(self):
        import ipdb; ipdb.set_trace()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(("127.0.0.1", self.remotePort))

    def readLoads(self):

        loads = {}

        for router in self.routers:

            try:
                with open("{1}load_{0}".format(router, tmp_files)) as f:
                    loads[router] = json.load(f)

            except IOError, e:
                print e

        return loads

    def readBridgesLoads(self):

        loads = {}
        try:
            with open("{1}load_{0}".format("root", tmp_files)) as f:
                loads["bridgeSwitches"] = json.load(f)

        except IOError, e:
            print e
        return loads

    def readLoadsEdges(self):

        loads = {}
        for router in self.edgeRouters:
            try:
                with open("{1}load_{0}".format(router, tmp_files)) as f:
                    loads[router] = json.load(f)
            except IOError, e:
                print e

        return loads

    def getInOutTraffic(self):

        in_traffic = sum({x: y for x, y in self.link_loads.items() if "sw" in x[0]}.values())
        out_traffic = sum({x: y for x, y in self.link_loads.items() if "sw" in x[1]}.values())

        return in_traffic, out_traffic

    def getParsedLoads(self):
        self.link_loads = {}
        d = self.readLoads()
        self.topology.routerUsageToLinksLoad(d, self.link_loads)
        return self.link_loads

    def getParsedLoads_with_bridges(self):
        self.link_loads = {}
        routers_interfaces = self.readLoads()
        bridges_interfaces = self.readBridgesLoads()
        routers_interfaces.update(bridges_interfaces)

        self.topology.routerUsageToLinksLoad(routers_interfaces, self.link_loads)

        return self.link_loads

    def handShake(self):
        self.connectToRemoteServer()

        # send toplogy
        send_msg(self.s, pickle.dumps(self.topology.networkGraph.keepRoutersAndNormalSwitches()))

        # send positions
        send_msg(self.s, pickle.dumps(self.topology.networkGraph.getFatTreePositions(self.k, True)))

    def run(self):

        #
        # graphPlotterThread = Thread(target=self.topology.networkGraph.plotGraphAnimated,args=(4,self.queue))
        # graphPlotterThread.setDaemon(True)
        # graphPlotterThread.start()

        # connect to the remote server that will plot data
        try:
            self.handShake()

        except Exception, e:

            print e

        start_time = time.time()
        i = 1
        interval = self.time_interval

        # file for the in/out traffic
        in_out_file = open("in_out_file_{0}.txt".format(self.k), "w")

        while True:

            time.sleep(max(0, start_time + i * interval - time.time()))
            i += 1
            now = time.time()

            # loads self.link_loads
            try:
                self.getParsedLoads_with_bridges()
            except:
                continue
            # print {x: y for x, y in self.link_loads.items() if any("sw" in e for e in x)}

            # send link_load
            try:
                send_msg(self.s, pickle.dumps(self.link_loads))

                in_traffic, out_traffic = self.getInOutTraffic()
                print "in traffic:", in_traffic, "out traffic: ", out_traffic

                # save in_out traffic
                in_out_file.write("{0},{1}\n".format(in_traffic, out_traffic))
                in_out_file.flush()
                # save link loads

                # with open("getLoads_linkLoad_{0}".format(self.k),"w") as f:
                #     pickle.dump(self.link_loads,f)

            except socket.error:
                # try to reconnect again
                try:
                    print "Trying to connect with the server again"
                    self.connectToRemoteServer()
                    # send toplogy
                    send_msg(self.s, pickle.dumps(self.topology.networkGraph.keepRoutersAndNormalSwitches()))
                    # send positions
                    send_msg(self.s, pickle.dumps(self.topology.networkGraph.getFatTreePositions(self.k, True)))
                except:
                    pass
            except Exception, e:
                print e
                break

            # queue new loads
            # self.sendLoadToThread()

            print time.time() - now


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-k', '--k',
                        help='Fat-Tree parameter',
                        type=int,
                        default=4)

    parser.add_argument('-p', '--remote_port',
                        help='Port at which the remoteDrow is listening',
                        type=int,
                        default=5010 )

    parser.add_argument('-i', '--time_interval',
                        help='Polling interval',
                        type=float,
                        default=1)

    args = parser.parse_args()

    sendLoads = SendLoads(args.remote_port, args.k, time_interval=args.time_interval).run()