#!/usr/bin/python

from fibbingnode.algorithms.southbound_interface import SouthboundManager
from fibbingnode import CFG as CFG_fib
from fibte.misc.unixSockets import UnixServer
import threading
import os
import time
import copy
import argparse
import json
import networkx as nx
from fibte.misc.dc_graph import DCGraph, DCDag
import ipaddress as ip

import subprocess

from fibte.misc.topology_graph import TopologyGraph


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
    def __init__(self, doBalance = True, k=4, load_variables=True):

        # Config logging to dedicated file for this thread
        handler = logging.FileHandler(filename='{0}loadbalancer.log'.format(tmp_files))
        fmt = logging.Formatter('[%(levelname)20s] %(asctime)s %(funcName)s: %(message)s ')

        handler.setFormatter(fmt)
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)

        # Set fat-tree parameter
        self.k = k

        # Either we load balance or not
        self.doBalance = doBalance

        # Lock for accessing link loads
        self.link_loads_lock = threading.Lock()

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

        # Load the topology
        self.topology = TopologyGraph(getIfindexes=True, db=os.path.join(tmp_files, db_topo))

        # Get dictionary where loads are stored
        self.link_loads = self.topology.getEdgesUsageDictionary()

        # Receive network graph
        self.network_graph = self.sbmanager.igp_graph

        # Create my modified version of the graph
        self.dc_graph = DCGraph(k=self.k)

        # Here we store the current dags for each destiantion
        self.dags = self._createInitialDags()
        self.initial_dags = copy.deepcopy(self.dags)

        # Fill ospf_prefixes dict
        self.ospf_prefixes = self._fillInitialOSPFPrefixes()

        # Start getLoads thread that reads from counters
        #os.system(getLoads_path + ' -k {0} &'.format(self.k))
        self.p_getLoads = subprocess.Popen([getLoads_path, '-k', str(self.k)], shell=False)

        # Start getLoads thread reads from link usage
        thread = threading.Thread(target=self._getLoads, args=([1]))
        thread.setDaemon(True)
        thread.start()

        # This is for debugging purposes only --should be removed
        if load_variables == True and self.k == 4:
            self.r_0_e0 = self.topology.getRouterId('r_0_e0')
            self.r_0_e1 = self.topology.getRouterId('r_0_e1')

            self.r_1_e0 = self.topology.getRouterId('r_1_e0')
            self.r_1_e1 = self.topology.getRouterId('r_1_e1')

            self.r_2_e0 = self.topology.getRouterId('r_2_e0')
            self.r_2_e1 = self.topology.getRouterId('r_2_e1')

            self.r_3_e0 = self.topology.getRouterId('r_3_e0')
            self.r_3_e1 = self.topology.getRouterId('r_3_e1')

            self.r_0_a0 = self.topology.getRouterId('r_0_a0')
            self.r_0_a1 = self.topology.getRouterId('r_0_a1')

            self.r_1_a0 = self.topology.getRouterId('r_1_a0')
            self.r_1_a1 = self.topology.getRouterId('r_1_a1')

            self.r_2_a0 = self.topology.getRouterId('r_2_a0')
            self.r_2_a1 = self.topology.getRouterId('r_2_a1')

            self.r_3_a0 = self.topology.getRouterId('r_3_a0')
            self.r_3_a1 = self.topology.getRouterId('r_3_a1')

            self.r_c0 = self.topology.getRouterId('r_c0')
            self.r_c1 = self.topology.getRouterId('r_c1')
            self.r_c2 = self.topology.getRouterId('r_c2')
            self.r_c3 = self.topology.getRouterId('r_c3')

    def _getGatewayRouter(self, prefix):
        """
        Given a prefix, returns the connected edge router
        :param prefix:
        :return:
        """
        if self.network_graph.is_prefix(prefix):
            pred = self.network_graph.predecessors(prefix)
            if len(pred) == 1 and not self.network_graph.is_fake_route(pred[0], prefix):
                return pred[0]
            else:
                real_gw = [p for p in pred if not self.network_graph.is_fake_route(p, prefix)]
                if len(real_gw) == 1:
                    return real_gw[0]
                else:
                    import ipdb; ipdb.set_trace()
                    log.error("This prefix has several predecessors: {0}".format(prefix))

                    #raise ValueError("This prefix has several predecessors: {0}".format(prefix))
        else:
            raise ValueError("{0} is not a prefix!".format(prefix))

    def getGatewayRouter(self, prefix):
        """
        Checks if is already stored. Otherwise calls _getGatewayRouter()
        :param prefix:
        :return:
        """
        if prefix in self.dags.keys() and self.dags[prefix].has_key('gateway'):
            return self.dags[prefix]['gateway']
        else:
            # Maybe it's a newly created prefix -- so check in the network graph
            return self._getGatewayRouter(prefix)

    def getMatchingPrefix(self, hostip):
        """
        Given a ip address of a host in the mininet network, returns
        the longest prefix currently being advertised by the OSPF routers.

        :param hostip: string representing a host's ip of an IPv4Address object
                             address. E.g: '192.168.233.254/30'
        Returns: an ipaddress.IPv4Network object
        """
        if not isinstance(hostip, ip.IPv4Address) and (isinstance(hostip, str) or isinstance(hostip, unicode)):
            # Convert it to ipv4address type
            hostip = ip.ip_address(hostip)

        longest_match = (None, 0)
        for prefix in self.ospf_prefixes:
            prefix_len = prefix.prefixlen
            if hostip in prefix and prefix_len > longest_match[1]:
                longest_match = (prefix, prefix_len)
        return longest_match[0].compressed

    def getConnectedPrefixes(self, edgeRouter):
        """
        Given an edge router, returns the connected prefixes
        :param edgeRouter:
        :return:
        """
        if self.dc_graph.is_edge(edgeRouter):
            return [r for r in self.network_graph.successors(edgeRouter) if self.network_graph.is_prefix(r)]
        else:
            raise ValueError("{0} is not an edge router".format(edgeRouter))

    def _createInitialDags(self):
        """
        Populates the self.dags dictionary for each existing prefix in the network
        """
        # Result is stored here
        log.info("Creating initial DAGs (default OSPF)")
        dags = {}

        for prefix in self.network_graph.prefixes:

            # TODO: change this so that IP is read dynamically
            if prefix != '192.168.255.0/24': #IP of the fibbing controller prefix...

                # Get Edge router connected to prefix
                gatewayRouter = self._getGatewayRouter(prefix)

                # Create a new dag
                dc_dag = self.dc_graph.get_default_ospf_dag(sinkid=gatewayRouter)

                # Add dag
                dags[prefix] = {'gateway': gatewayRouter, 'dag': dc_dag}

                try:
                    # Instruct fibbing controller
                    self.sbmanager.add_dag_requirement(prefix, dc_dag)
                except Exception as e:
                    import ipdb; ipdb.set_trace()

        return dags

    def _fillInitialOSPFPrefixes(self):
        """
        Fills up the data structure
        """
        prefixes = []
        fill = [prefixes.append(ip.ip_network(prefix)) for prefix in self.network_graph.prefixes]
        return prefixes

    def reset(self):
        """
        Sets the load balancer to its initial state
        :return:
        """
        # Start crono
        reset_time = time.time()

        # Set all dags to original ospf dag
        self.dags = copy.deepcopy(self.initial_dags)

        # Reset all dags to initial state
        action = [self.sbmanager.add_dag_requirement(prefix, self.dags[prefix]['dag']) for prefix in self.dags.keys()]

        log.debug("Time to perform the reset to the load balancer: {0}s".format(time.time() - reset_time))

    def handleFlow(self, event):
        """
         Default handle flow skeleton
         :param event:
         :return:
         """
        log.info("Event to handle received: {0}".format(event["type"]))
        if event["type"] == 'startingFlow':
            flow = event['flow']
            log.debug("New flow STARTED: {0}".format(flow))

        elif event["type"] == 'stoppingFlow':
            flow = event['flow']
            log.debug("Flow FINISHED: {0}".format(flow))

        else:
            log.error("Unknown event type: {0}".format(event['type']))

    def _runGetLoads(self):
        getLoads = GetLoads(k=self.k)
        getLoads.run()

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

    def exitGracefully(self):
        """
        Exit load balancer gracefully
        :return:
        """
        log.info("Keyboad Interrupt catched!")

        log.info("Cleaning up the network from fake LSAs ...")
        # Remove all lies before leaving
        for prefix in self.network_graph.prefixes:
            self.sbmanager.remove_dag_requirement(prefix)

        # self.p_getLoads.terminate()

        # Finally exit
        os._exit(0)

    def run(self):
        # Receive events and handle them
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
                # Exit load balancer
                self.exitGracefully()

class ECMPController(LBController):
    def __init__(self, doBalance=True, k=4):
        super(ECMPController, self).__init__(doBalance, k)

class RandomUplinksController(LBController):
    def __init__(self, doBalance=True, k=4):
        super(RandomUplinksController, self).__init__(doBalance, k)
        # Keeps track of elephant flows from each pod to every destination
        self.flows_per_pod = {px: {pod: [] for pod in range(0, self.k)} for px in self.network_graph.prefixes}

    def handleFlow(self, event):
        """
         Default handle flow skeleton
         :param event:
         :return:
         """
        log.info("Event to handle received: {0}".format(event["type"]))
        if event["type"] == 'startingFlow':
            flow = event['flow']
            self.chooseRandomUplinks(flow)

        elif event["type"] == 'stoppingFlow':
            flow = event['flow']
            self.resetOSPFDag(flow)

        else:
            log.error("Unknown event type: {0}".format(event['type']))

    def areOngoingFlowsInPod(self, dst_prefix, src_pod=None):
        """Returns the list of ongoing elephant flows to
        destination prefix from specific pod.

        If pod==None, all flows to dst_prefix are returned

        :param src_pod:
        :param dst_prefix:
        :return:
        """
        if src_pod != None:
            if self.dc_graph._valid_pod_number(src_pod):
                return self.flows_per_pod[dst_prefix][src_pod] != []
            else:
                raise ValueError("Wrong pod number: {0}".format(src_pod))
        else:
            return any([True for _, flowlist in self.flows_per_pod[dst_prefix].iteritems() if flowlist != []])

    def getOngoingFlowsInPod(self, dst_prefix, src_pod=None):
        """Returns the list of ongoing elephant flows to
        destination prefix from specific pod.

        If pod==None, all flows to dst_prefix are returned

        :param src_pod:
        :param dst_prefix:
        :return:
        """
        if src_pod != None:
            if self.dc_graph._valid_pod_number(src_pod):
                return self.flows_per_pod[dst_prefix][src_pod]
            else:
                raise ValueError("Wrong pod number: {0}".format(src_pod))
        else:
            final_list = []
            action = [final_list.append(flowlist) for _, flowlist in self.flows_per_pod[dst_prefix].iteritems()]
            return final_list

    def chooseRandomUplinks(self, flow):
        """
        Given a new elephant flow from a certain source, it forces a random
        uplink from that pod to the core layer.

        :param flow:
        :return:
        """
        # Get prefix from host ip
        src_ip = flow['src']
        dst_ip = flow['dst']
        src_prefix = self.getMatchingPrefix(src_ip)
        dst_prefix = self.getMatchingPrefix(dst_ip)

        # Get gateway router
        src_gw = self.getGatewayRouter(src_prefix)

        # Get source gateway pod
        src_pod = self.dc_graph.get_router_pod(routerid=src_gw)

        # Check if already ongoing flows
        if not self.areOngoingFlowsInPod(dst_prefix=dst_prefix, src_pod=src_pod):
            log.debug("There are no ongoing flows to {0} from pod {1}".format(dst_prefix, src_pod))

            # Retrieve current DAG for destination prefix
            current_dag = self.dags[dst_prefix]['dag']

            # Modify DAG
            current_dag.modify_random_uplinks(src_pod=src_pod)

            # Apply DAG
            self.sbmanager.add_dag_requirement(prefix=dst_prefix, dag=current_dag)
            log.info("A new random uplink DAG from pod {0} was forced to prefix {1}".format(src_pod, dst_prefix))

        # There are already ongoing flows!
        else:
            log.debug("There are ongoing flows from pod {0} to prefix {1}".format(src_pod, dst_prefix))

        # Add flow to flows_per_pod
        self.flows_per_pod[dst_prefix][src_pod].append(flow)

    def resetOSPFDag(self, flow):
        """
        When an elephant flow finishes, this function tries to reset
        the random uplink from the pod to the core to its original
        OSPF dag.

        It will only do it if no more elephant flows to the destination
        are ongoing from that pod.

        :param flow:
        :return:
        """
        reset_ospf_time = time.time()

        # Get prefix from host ip
        src_ip = flow['src']
        dst_ip = flow['dst']
        src_prefix = self.getMatchingPrefix(src_ip)
        dst_prefix = self.getMatchingPrefix(dst_ip)

        # Get gateway router for source
        src_gw = self.getGatewayRouter(src_prefix)

        # Get source gateway pod
        src_pod = self.dc_graph.get_router_pod(routerid=src_gw)

        # Remove flow from flows_per_pod
        self.flows_per_pod[dst_prefix][src_pod].remove(flow)

        # Check if already ongoing flows
        if not self.areOngoingFlowsInPod(dst_prefix=dst_prefix, src_pod=src_pod):
            # Retrieve current DAG for destination prefix
            current_dag = self.dags[dst_prefix]['dag']

            # Modify DAG -- set uplinks from source pod to default ECMP
            current_dag.set_ecmp_uplinks(src_pod=src_pod)

            # Apply DAG
            self.sbmanager.add_dag_requirement(prefix=dst_prefix, dag=current_dag)
            log.info("A new DAG was forced -> {0}".format(dst_prefix))

        # Still ongoing flows from that source pod
        else:
            log.debug("There are ongoing flows from pod{0} -> {1}".format(src_pod, dst_prefix))

        elapsed_time = round(time.time()-reset_ospf_time, 3)
        log.debug("It took {0}s to reset the default OSPF uplinks: pod{1} -> {2}".format(elapsed_time, src_pod, dst_prefix))

    def reset(self):
        super(RandomUplinksController, self).reset()
        # Reset the flows_per_pod too
        self.flows_per_pod = {px: {pod: [] for pod in range(0, self.k)} for px in self.network_graph.prefixes}

if __name__ == '__main__':
    from fibte.logger import log
    import logging

    parser = argparse.ArgumentParser()    #group = parser.add_mutually_exclusive_group()

    parser.add_argument('--doBalance',
                        help='If set to False, ignores all events and just prints them',
                        action='store_true',
                        default = True)

    parser.add_argument('--algorithm',
                        help='Choose loadbalancing strategy: ecmp|random',
                        default = 'ecmp')

    parser.add_argument('-k', '--k', help='Fat-Tree parameter', type=int, default=4)

    args = parser.parse_args()

    log.setLevel(logging.DEBUG)
    log.info("Starting Controller - k = {0} , algorithm = {1}".format(args.k, args.algorithm))

    if args.algorithm == 'ecmp':
        lb = ECMPController(doBalance = args.doBalance, k=args.k)

    elif args.algorithm == 'random':
        lb = RandomUplinksController(doBalance= args.doBalance, k=args.k)

    lb.run()
    #import ipdb; ipdb.set_trace()
