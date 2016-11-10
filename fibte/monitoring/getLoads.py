#!/usr/bin/python

from fibte.misc.topology_graph import TopologyGraph
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
import signal
import sys

throughput_folder = os.path.join(os.path.dirname(__file__), 'results/throughput/')


class GetLoads(object):
    def __init__(self, k=4, time_interval=1, lb_algorithm=None):

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

        # Algorithm that the LB is using
        self.lb_algorithm = lb_algorithm

        # Load read-outs intervalt
        self.time_interval = time_interval

        # Create the dictionary to store temporary aggregation loads
        self.aggregation_loads = self.createAggregationLoads()

        signal.signal(signal.SIGTERM, self.signal_term_handler)

    def signal_term_handler(self, signal, frame):
        log.info('got SIGTERM - getLoads thread is exiting!')
        self.agg_file.flush()
        self.in_out_file.flush()
        sys.exit(0)

    def createAggregationLoads(self):
        d = {}
        edgeRouters = self.topology.getEdgeRouters()
        coreRouters = self.topology.getCoreRouters()
        others = edgeRouters + coreRouters
        for u in self.topology.getAggregationRouters():
            u_i = self.topology.getRouterIndex(u)
            u_pod = self.topology.getRouterPod(u)
            d[u] = {}
            for v in others:
                if v in edgeRouters:
                    v_pod = self.topology.getRouterPod(v)
                    if v_pod == u_pod:
                        d[u][v] = {'in': 0, 'out': 0}

                elif v in coreRouters:
                    v_i = self.topology.getRouterIndex(v)
                    if v_i in range((self.k/2)*u_i, (u_i+1)*(self.k/2)):
                        d[u][v] = {'in': 0, 'out': 0}
        return d

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
        in_traffic = sum({x: y for x, y in self.link_loads.items() if "h" in x[0]}.values())

        # Get all hosts outgoing traffic
        out_traffic = sum({x: y for x, y in self.link_loads.items() if "h" in x[1]}.values())

        return in_traffic, out_traffic

    def getAggregationTraffic(self):
        # Make a copy
        aggregationTraffic = self.aggregation_loads.copy()

        # Get aggregation routers
        aggregationRouters = set(self.topology.getAggregationRouters())

        # Take time
        readout_time = time.time()
        for ((a, b), load) in self.link_loads.iteritems():
            if a in aggregationRouters:
                self.aggregation_loads[a][b]['out'] = load
            elif b in aggregationRouters:
                self.aggregation_loads[b][a]['in'] = load

        # Append time
        aggregationTraffic['time'] = readout_time
        return aggregationTraffic

    def run(self):
        # Log a bit
        log.info("GetLoads thread is active")

        start_time = time.time()
        i = 1
        interval = self.time_interval

        # File for aggregation traffic
        self.agg_file = open("{1}bb_{0}_{2}.txt".format(self.k, throughput_folder, self.lb_algorithm), "w")
        self.in_out_file = open("{1}in_out_{0}_{2}.txt".format(self.k, throughput_folder, self.lb_algorithm), "w")

        while True:
            try:
                # Sleep first the exact remaining time
                time.sleep(max(0, start_time + i * interval - time.time()))
                now = time.time()

                reading_time = time.time()

                # Fill loads of the router edges into link_loads
                self.topology.routerUsageToLinksLoad(self.readLoads(), self.link_loads)
            except KeyboardInterrupt as e:
                log.exception("Exception catched in run(): {0}".format(e))
                break

            # Print
            try:
                # Save all link loads
                with open("{1}getLoads_linkLoad_{0}".format(self.k, throughput_folder), "w") as f:
                    pickle.dump(self.link_loads, f)

                # Get aggreagation traffic
                aggTraffic = self.getAggregationTraffic()
                intf, outtf = self.getInOutTraffic()

                # Save aggregation traffic in a file
                self.agg_file.write("{0}\n".format(json.dumps(aggTraffic)))
                self.agg_file.flush()

                self.in_out_file.write("{0}\t{1}\n".format(intf, outtf))
                self.in_out_file.flush()

            except Exception as e:
                log.exception("Exception catched in run(): {0}".format(e))
                self.agg_file.flush()
                self.in_out_file.flush()
                break

            else:
                # Incement counter
                i += 1

        # Leave
        return

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group()

    parser.add_argument('-k', '--k', help='Fat-Tree parameter', type=int, default=4)

    parser.add_argument('-a', '--algorithm', help='Algorithm of the loadbalancer - used to save results file by algorithm name', type=str, default=None)

    parser.add_argument('-i', '--time_interval', help='Polling interval', type=float, default=1.1)

    args = parser.parse_args()

    gl = GetLoads(k=args.k, time_interval=args.time_interval, lb_algorithm=args.algorithm)
    gl.run()
    sys.exit()
