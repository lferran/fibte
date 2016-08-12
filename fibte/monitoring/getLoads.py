#!/usr/bin/python

from fibte.misc.topologyGraph import TopologyGraph
import json
import time

try:
    import cPickle as pickle
except:
    import pickle

from fibte import CFG, LINK_BANDWIDTH

tmp_files = CFG.get("DEFAULT", "tmp_files")
db_topo = CFG.get("DEFAULT", "db_topo")

import logging
from fibte.logger import log

import os

results_folder = os.path.join(os.path.dirname(__file__), 'results/')

class GetLoads(object):
    def __init__(self, k=4, time_interval=1):

        # Config logging to dedicated file for this thread
        handler = logging.FileHandler(filename='{0}getLoads_thread.log'.format(tmp_files))
        fmt = logging.Formatter('[%(levelname)20s] %(asctime)s %(funcName)s: %(message)s ')

        handler.setFormatter(fmt)
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)

        # Load the topology object
        self.topology = TopologyGraph(getIfindexes=True, db=os.path.join(tmp_files, db_topo))

        # Get all routers
        self.routers = self.topology.getRouters()

        # Get all edge routers
        self.edgeRouters = self.topology.getEdgeRouters()

        # Dictionary of edge between router -> load
        self.link_loads = self.topology.getEdgesUsageDictionary()

        # Fat-Tree parameter
        self.k = k

        # Load read-outs interval
        self.time_interval = time_interval

    def readLoads(self):
        """
        Reads all loads from the routers and returns a dictionary {}: router -> load
        """
        loads = {}
        for router in self.routers:
            try:
                with open("{1}load_{0}".format(router, tmp_files)) as f:
                    loads[router] = json.load(f)
            except IOError, e:
                log.error("Error in readLoads(): {0}".format(e))

        return loads

    def readLoadsEdges(self):
        """
        Same as readLoadas bur only from edge routers
        """
        loads = {}
        for router in self.edgeRouters:
            try:
                with open("{1}load_{0}".format(router, tmp_files)) as f:
                    loads[router] = json.load(f)
            except IOError, e:
                log.error("Error in readLoadsEdges(): {0}".format(e))

        return loads

    def getInOutTraffic(self):
        """
        Reads Incoming and Outgoing traffic to/from hosts
        """
        # Get all hosts incoming traffic
        in_traffic = sum({x: y for x, y in self.link_loads.items() if "ovs" in x[0]}.values())

        # Get all hosts outgoing traffic
        out_traffic = sum({x: y for x, y in self.link_loads.items() if "ovs" in x[1]}.values())

        return in_traffic, out_traffic

    def getBisectionTraffic(self):
        """
        Calculates average bisection bandwidth
        """
        # Get aggregation routers
        aggregationRouters = set(self.topology.getAgreggationRouters())

        # Get core routers
        coreRouters = set(self.topology.getCoreRouters())

        # Get only bisection edges -- only upwards
        bisectionEdges = [(a, b, load) for (a, b), load in self.link_loads.iteritems()
                          if (a in aggregationRouters and b in coreRouters)]

        # Compute bisection bandwdith
        bisectionBandwidth = sum([load for (_,_,load) in bisectionEdges])

        return bisectionBandwidth

    def run(self):
        # Log a bit
        log.info("GetLoads thread is active")

        start_time = time.time()
        i = 1
        interval = self.time_interval

        # File for the in/out traffic
        in_out_file = open("{1}in_out_file_{0}.txt".format(self.k, results_folder), "w")

        # File for bisection bandwidth
        bb_file = open("{1}bisection_bw_file_{0}.txt".format(self.k, results_folder), "w")

        while True:
            # Sleep first the exact remaining time
            time.sleep(max(0, start_time + i * interval - time.time()))
            now = time.time()

            # Fill loads of the router edges into link_loads
            self.topology.routerUsageToLinksLoad(self.readLoads(), self.link_loads)
            # print {x:y for x,y in self.link_loads.items() if any("sw" in e for e in x)}

            # Print
            try:
                # Get In&Out traffic
                in_traffic, out_traffic = self.getInOutTraffic()

                # Get bisection BW
                bisecBW = self.getBisectionTraffic()

                max_bisecBW = ((self.k**3)/4.0)#*LINK_BANDWIDTH
                bisecBw_ratio = round((bisecBW / max_bisecBW)*100.0, 3)

                # Save in_out traffic in a file
                in_out_file.write("{0},{1}\n".format(in_traffic, out_traffic))
                in_out_file.flush()

                # Save bisection bw in a file
                bb_file.write("{0},{1}\n".format(bisecBW, bisecBw_ratio))
                bb_file.flush()

                # Save all link loads
                with open("{1}getLoads_linkLoad_{0}".format(self.k, results_folder), "w") as f:
                    pickle.dump(self.link_loads, f)

            except Exception, e:
                log.error("Error in run(): {0}".format(e))
                break

            i += 1
            #print time.time() - now


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group()

    parser.add_argument('-k', '--k', help='Fat-Tree parameter', type=int, default=4)

    parser.add_argument('-i', '--time_interval', help='Polling interval', type=float, default=1.5)

    args = parser.parse_args()

    gl = GetLoads(k=args.k, time_interval=args.time_interval)
    gl.run()