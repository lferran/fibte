#!/usr/bin/python

from fibbingnode.algorithms.southbound_interface import SouthboundManager
from fibbingnode import CFG as CFG_fib
from fibte.misc.unixSockets import UnixServer
import threading
import subprocess
import os
import time
import argparse
import json

from fibte.misc.topologyGraph import TopologyGraph

from fibte.monitoring.getLoads import GetLoads

from fibte import tmp_files, db_topo, LINK_BANDWIDTH, CFG

# Threading event to signal that the initial topo graph
# has been received from the Fibbing controller
HAS_INITIAL_GRAPH = threading.Event()

UDS_server_name = CFG.get("DEFAULT","controller_UDS_name")
C1_cfg = CFG.get("DEFAULT", "C1_cfg")

import inspect
import fibte.monitoring.getLoads
getLoads_path = inspect.getsourcefile(fibte.monitoring.getLoads)

import logging
from fibte.logger import log

class MyGraphProvider(SouthboundManager):
    """This class overrwides the received_initial_graph abstract method of
    the SouthboundManager class. It is used to receive the initial
    graph from the Fibbing controller.
    The HAS_INITIAL_GRAPH is set when the method is called.
    """
    def __init__(self):
        super(MyGraphProvider, self).__init__()

    def received_initial_graph(self):
        super(MyGraphProvider, self).received_initial_graph()
        HAS_INITIAL_GRAPH.set()

class LBController(object):
    def __init__(self, doBalance = True, k=4):

        # Config logging to dedicated file for this thread
        handler = logging.FileHandler(filename='{0}loadbalancer.log'.format(tmp_files))
        fmt = logging.Formatter('[%(levelname)20s] %(asctime)s %(funcName)s: %(message)s ')

        handler.setFormatter(fmt)
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)

        # Set fat-tree parameter
        self.k = k

        # Load the topology
        self.topology = TopologyGraph(getIfindexes=True, db=os.path.join(tmp_files, db_topo))

        # Either we load balance or not
        self.doBalance = doBalance

        # Lock for accessing link loads
        self.link_loads_lock = threading.Lock()

        # Get dictionary where loads are stored
        self.link_loads = self.topology.getEdgesUsageDictionary()

        # Unix domain server to make things faster and possibility to communicate with hosts
        self.server = UnixServer(os.path.join(tmp_files, UDS_server_name))

        # Connects to the southbound controller. Must be called before
        # creating the instance of SouthboundManager
        CFG_fib.read(os.path.join(tmp_files, C1_cfg))

        # Start the Southbound manager in a different thread
        self.sbmanager = MyGraphProvider()
        t = threading.Thread(target=self.sbmanager.run, name="Southbound Manager")
        t.start()

        # Blocks until initial graph received from SouthBound Manager
        HAS_INITIAL_GRAPH.wait()
        log.info("Initial graph received from SouthBound Controller")
        
        # Receive network graph
        self.networw_graph = self.sbmanager.igp_graph

        # Start getLoads thread that reads from counters
        process = subprocess.call([getLoads_path, '-k', str(self.k)], shell=False)

        # Start getLoads thread reads from link usage
        thread = threading.Thread(target=self._getLoads, args=([1]))
        thread.setDaemon(True)
        thread.start()

    def reset(self):
        pass

    def handleFlow(self, event):
        log.info("Event to handle received: {0}".format(event["type"]))
        if event["type"] == 'startingFlow':
            flow = event['flow']
            log.debug("New flow STARTED: {0}".format(flow))

        elif event["type"] == 'stoppingFlow':
            flow = event['flow']
            log.debug("Flow FINISHED: {0}".format(flow))            

        else:
            log.error("epali")

    def _getLoads(self, t):
        """
        This function runs in a separate thread and periodically reads
         updates the loads of the network links

        :param t: period at which we update the LB link_loads
        :return:
        """
        getLoads = GetLoads(k=self.k)
        while True:
            # Take time
            now = time.time()

            # Locking to update link_loads
            with self.link_loads_lock:
                getLoads.topology.routerUsageToLinksLoad(getLoads.readLoads(), self.link_loads)

            # Sleep remaining interval time
            time.sleep(max(0, t - (time.time() - now)))

    def _getStats(self, t):
        """
        Function that periodically updates the statistics gathered
        from the edge of the network.

        :param t:
        :return:
        """
        # TODO
        pass

    def run(self):
        # Receive events and handle them
        #import ipdb; ipdb.set_trace()
        while True:
            try:
                if not(self.doBalance):
                    while True:
                        event = self.server.receive()
                        log.info("LB not active - event received: {0}".format(json.loads(event)))
                else:
                    event = json.loads(self.server.receive())
                    if event["type"] == "reset":
                        self.reset()
                        continue
                    else:
                        self.handleFlow(event)
                        
            except KeyboardInterrupt:
                break

if __name__ == '__main__':
    from fibte.logger import log
    import logging

    parser = argparse.ArgumentParser()
    #group = parser.add_mutually_exclusive_group()

    parser.add_argument('--doBalance',
                        help='If set to False, ignores all events and just prints them',
                        action='store_true',
                        default = True)

    parser.add_argument('-k', '--k', help='Fat-Tree parameter', type=int, default=4)

    args = parser.parse_args()

    log.setLevel(logging.DEBUG)
    log.info("Starting Controller - k = {0} , doBalance = {1}".format(args.k, args.doBalance))

    lb = LBController(doBalance = args.doBalance, k=args.k)
    lb.run()

    #import ipdb; ipdb.set_trace()
