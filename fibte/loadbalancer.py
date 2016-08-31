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
from fibte.trafficgen.flow import Base
import random
import subprocess
import abc

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
    def __init__(self, doBalance = True, k=4, algorithm=None, load_variables=False):

        # Config logging to dedicated file for this thread
        handler = logging.FileHandler(filename='{0}loadbalancer_{1}.log'.format(tmp_files, algorithm))
        fmt = logging.Formatter('[%(levelname)20s] %(asctime)s %(funcName)s: %(message)s ')

        handler.setFormatter(fmt)
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)

        # Set fat-tree parameter
        self.k = k

        # Either we load balance or not
        self.doBalance = doBalance

        # Loadbalancing strategy/algorithm
        self.algorithm = algorithm

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

        # Lock for accessing link loads
        self.link_loads_lock = threading.Lock()

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
        self.p_getLoads = subprocess.Popen([getLoads_path, '-k', str(self.k), '-a', self.algorithm], shell=False)

        # Start getLoads thread reads from link usage
        thread = threading.Thread(target=self._getLoads, args=([1]))
        thread.setDaemon(True)
        thread.start()

        # Object useful to make some unit conversions
        self.base = Base()

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
        if '/' in prefix:
            if prefix in self.dags.keys() and self.dags[prefix].has_key('gateway'):
                return self.dags[prefix]['gateway']
            else:
                # Maybe it's a newly created prefix -- so check in the network graph
                return self._getGatewayRouter(prefix)
        else:
            # Must be an ip then
            ipe = prefix
            ipe_prefix = self.getMatchingPrefix(ipe)
            return self.getGatewayRouter(ipe_prefix)

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

    def getPrefixesFromFlow(self, flow):
        return (self.getSourcePrefixFromFlow(flow), self.getDestinationPrefixFromFlow(flow))

    def getDestinationPrefixFromFlow(self, flow):
        dst_ip = flow['dst']
        dst_prefix = self.getMatchingPrefix(dst_ip)
        return dst_prefix

    def getSourcePrefixFromFlow(self, flow):
        src_ip = flow['src']
        src_prefix = self.getMatchingPrefix(src_ip)
        return src_prefix

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

        # Remove all attraction points and lsas
        # Set all dags to original ospf dag
        self.dags = copy.deepcopy(self.initial_dags)

        # Reset all dags to initial state
        action = [self.sbmanager.add_dag_requirement(prefix, self.dags[prefix]['dag']) for prefix in self.dags.keys()]

        return reset_time

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
        log.info("Looping for new events!")
        while True:
            try:
                if not(self.doBalance):
                    while True:
                        event = self.server.receive()
                        log.info("LB not active - event received: {0}".format(json.loads(event)))
                else:
                    event = json.loads(self.server.receive())
                    if event["type"] == "reset":
                        log.info("RESET event received")
                        self.reset()
                        continue
                    else:
                        self.handleFlow(event)
            except KeyboardInterrupt:
                # Exit load balancer
                self.exitGracefully()

class ECMPController(LBController):
    def __init__(self, doBalance=True, k=4):
        super(ECMPController, self).__init__(doBalance, k, algorithm='ecmp')

class RandomUplinksController(LBController):
    def __init__(self, doBalance=True, k=4):
        super(RandomUplinksController, self).__init__(doBalance, k, algorithm='random-dags')
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
        (src_prefix, dst_prefix) = self.getPrefixesFromFlow(flow)

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
            current_dag.set_ecmp_uplinks_from_pod(src_pod=src_pod)

            # Apply DAG
            self.sbmanager.add_dag_requirement(prefix=dst_prefix, dag=current_dag)
            log.info("A new DAG was forced -> {0}".format(dst_prefix))

        # Still ongoing flows from that source pod
        else:
            log.debug("There are ongoing flows from pod{0} -> {1}".format(src_pod, dst_prefix))

        elapsed_time = round(time.time()-reset_ospf_time, 3)
        log.debug("It took {0}s to reset the default OSPF uplinks: pod{1} -> {2}".format(elapsed_time, src_pod, dst_prefix))

    def reset(self):
        reset_time = super(RandomUplinksController, self).reset()

        # Reset the flows_per_pod too
        self.flows_per_pod = {px: {pod: [] for pod in range(0, self.k)} for px in self.network_graph.prefixes}

        log.debug("Time to perform the reset to the load balancer: {0}s".format(time.time() - reset_time))

class CoreChooserController(LBController):
    def __init__(self, doBalance=True, k=4, threshold=0.9, algorithm=None):
        super(CoreChooserController, self).__init__(doBalance, k, algorithm)

        # Create structure where we store the ongoing elephant  flows in the graph
        self.elephants_in_paths = self._createElephantInPathsDict()

        # We consider congested a link more than threshold % of its capacity
        self.capacity_threshold = threshold

        # Store all paths --for performance reasons
        self.edge_to_core_paths = self._generateEdgeToCorePaths()

        # Keeps track to which core is each flow directed to
        self.flow_to_core = self._generateFlowToCoreDict()

    def _generateFlowToCoreDict(self):
        flow_to_core = {}
        for c in self.dc_graph.core_routers_iter():
            flow_to_core[c] = {}
            for p in self.network_graph.prefixes:
                flow_to_core[c][p] = []
        return flow_to_core

    def _generateEdgeToCorePaths(self):
        d = {}
        for edge in self.dc_graph.edge_routers_iter():
            d[edge] = {}
            for core in self.dc_graph.core_routers_iter():
                d[edge][core] = self._getPathFromEdgeToCore(edge, core)
        return d

    def _createElephantInPathsDict(self):
        elephant_in_paths = self.dc_graph.copy()
        for (u, v, data) in elephant_in_paths.edges_iter(data=True):
            data['flows'] = []
            data['capacity'] = LINK_BANDWIDTH
        return elephant_in_paths

    def getEdgesFromPath(self, path):
        return zip(path[:-1], path[1:])

    def getSourcePodFromFlow(self, flow):
        """Returns the pod of the source address for the
        givn flow"""
        srcpx = self.getSourcePrefixFromFlow(flow)
        srcgw = self.getGatewayRouter(srcpx)
        src_pod = self.dc_graph.get_router_pod(srcgw)
        return src_pod

    def getDstPodFromFlow(self, flow):
        """Returns the pod of the source address for the
        givn flow"""
        dstpx = self.getSourcePrefixFromFlow(flow)
        dstgw = self.getGatewayRouter(dstpx)
        dst_pod = self.dc_graph.get_router_pod(dstgw)
        return dst_pod

    def getSourceGatewayFromFlow(self, flow):
        """Returns the source entry point in the network
        """
        src_prefix = self.getSourcePrefixFromFlow(flow)
        return self.getGatewayRouter(src_prefix)

    def getPathFromEdgeToCore(self, edge, core):
        """
        Return single path from edge router to core router
        :param edge:
        :param core:
        :return:
        """
        return self.edge_to_core_paths[edge][core]

    def _getPathFromEdgeToCore(self, edge, core):
        """Used to construct the edge_to_core data structure
        """
        return nx.dijkstra_path(self.dc_graph, edge, core)

    def addFlowToPath(self, flow, path):
        edges = self.getEdgesFromPath(path)
        core = path[-1]
        for (u, v) in edges:
            self.elephants_in_paths[u][v]['flows'].append(flow)
            self.elephants_in_paths[u][v]['capacity'] -= flow['size']

        # Add it to flow_to_core datastructure too
        dst_prefix = self.getDestinationPrefixFromFlow(flow)
        if not self.flow_to_core[core].has_key(dst_prefix):
            self.flow_to_core[core][dst_prefix] = [flow]
        else:
            self.flow_to_core[core][dst_prefix].append(flow)

    def removeFlowFromPath(self, flow, path):
        edges = self.getEdgesFromPath(path)
        core = path[-1]
        for (u, v) in edges:
            if flow in self.elephants_in_paths[u][v]['flows']: self.elephants_in_paths[u][v]['flows'].remove(flow)
            self.elephants_in_paths[u][v]['capacity'] += flow['size']

        # Remove it from flow_to_core
        dst_prefix = self.getDestinationPrefixFromFlow(flow)
        if flow in self.flow_to_core[core][dst_prefix]: self.flow_to_core[core][dst_prefix].remove(flow)

    def flowFitsInPath(self, flow, path):
        """
        Returns a bool indicating if flow fits in path
        :param flow:
        :param path:
        :return: bool
        """
        path_edges = self.getEdgesFromPath(path)
        return all([True if flow['size'] <= self.elephants_in_paths[u][v]['capacity'] else False for (u,v) in path_edges])

    def getPathMinCapacity(self, path):
        """Returns the minimum available capacity observed along the
        edges of the given path"""
        path_edges = self.getEdgesFromPath(path)
        return min([self.elephants_in_paths[u][v]['capacity'] for (u,v) in path_edges])

    def getAvailableCorePaths(self, src_gw, flow):
        """
        Returns the list of available core router together with
        their path from source gw.

        It checks if flow fits in path and also
        :param src_gw:
        :param flow:
        :return:
        """
        #import ipdb; ipdb.set_trace()

        core_paths = []
        for core in self.dc_graph.core_routers():
            path = self.getPathFromEdgeToCore(src_gw, core)
            if self.flowFitsInPath(flow, path):
                if not self.collidesWithPreviousFlows(src_gw, core, flow):
                    capacity = self.getPathMinCapacity(path)
                    core_paths.append({'core': core, 'path': path, 'capacity': capacity})

        # No available core was found... so rank them!
        if core_paths == []:
            # Take one at random --for the moment
            #TODO: think what to do with this
            log.error("No available core paths found!! Returning the non-colliding paths!")
            core_paths = []
            for core in self.dc_graph.core_routers():
                path = self.getPathFromEdgeToCore(src_gw, core)
                if not self.collidesWithPreviousFlows(src_gw, core, flow):
                    capacity = self.getPathMinCapacity(path)
                    core_paths.append({'core': core, 'path': path, 'capacity': capacity})

        return core_paths

    def getongoingFlowsFromGateway(self, src_gw, dst_prefix):
        # Get ongoing flows to same prefix
        ongoing_flows = [f for c in self.flow_to_core.keys() for f in self.flow_to_core[c][dst_prefix]]

        # Filters only those with the same source pod
        ongoing_flows_same_gw = [flow for flow in ongoing_flows if self.getSourceGatewayFromFlow(flow) == src_gw]

        return ongoing_flows_same_gw

    def getOngoingFlowsFromPod(self, src_pod, dst_prefix):
        """
        Checks if
        :param src_pod:
        :param core:
        :param dst_prefix:
        :return:
        """
        # Get ongoing flows to same prefix
        ongoing_flows = [f for c in self.flow_to_core.keys() for f in self.flow_to_core[c][dst_prefix]]

        # Filters only those with the same source pod
        ongoing_flows_same_pod = [flow for flow in ongoing_flows if self.dc_graph.get_router_pod(self.getSourceGatewayFromFlow(flow)) == src_pod]

        return ongoing_flows_same_pod

    def getCurrentCoreForFlow(self, flow):
        """Get the current core used for flow"""
        core = [c for (c, data) in self.flow_to_core.iteritems() for (prefix, flowlist) in data.iteritems() if flow in flowlist]
        if len(core) == 1:
            return core[0]
        else:
            raise ValueError("Flow could not be found for any core!!!")

    def collidesWithPreviousFlows(self, src_gw, core, flow):
        """
        Checks if already ongoing flows to that destination
        :param src_gw:
        :param core:
        :return:
        """
        dst_prefix = self.getDestinationPrefixFromFlow(flow)

        src_pod = self.dc_graph.get_router_pod(src_gw)

        # Get all flows from that pod going to dst_prefix
        all_flows_same_pod = [self.flow_to_core[c][dst_prefix] for c in self.dc_graph.core_routers_iter()]
        all_flows_same_pod = [f for fl in all_flows_same_pod for f in fl if self.getSourcePodFromFlow(f) == src_pod]

        #Check if flow from same gateway router
        flows_from_same_gw = [f for f in all_flows_same_pod if self.getSourcePrefixFromFlow(f) == src_gw]
        if flows_from_same_gw:
            f = flows_from_same_gw[0]
            current_core_f = self.getCurrentCoreForFlow(f)
            if current_core_f == core:
                return False
            else:
                return True

        # Check if flow from gateways in same pod
        if all_flows_same_pod:
            # Get own path to core
            own_path_to_core = self.getPathFromEdgeToCore(src_gw, core)

            for f in all_flows_same_pod:
                # Get their current core
                f_core = self.getCurrentCoreForFlow(f)
                f_px = self.getSourcePrefixFromFlow(f)
                f_gw = self.getGatewayRouter(f_px)
                f_path_to_core = self.getPathFromEdgeToCore(f_gw, f_core)

                # Check if paths collide: same aggregation router but different core!
                if f_path_to_core[1] == own_path_to_core[1] and f_core != core:
                    return True

            return False

        return False

    def getCoreFromFlow(self, flow):
        """
        Given a flow that was already in the network, returns
        the core router that was assigned to
        :param flow:
        :return:
        """
        flow_dst_px = self.getDestinationPrefixFromFlow(flow)
        cores = [c for c, data in self.flow_to_core.iteritems() if data.has_key(flow_dst_px) and flow in data[flow_dst_px]]
        if len(cores) == 1:
            return cores[0]
        else:
            raise ValueError("Flow is not assigned to any core - that's weird")

    @abc.abstractmethod
    def chooseCore(self, available_cores):
        pass

    def allocateFlow(self, flow):
        # Check for which cores I can send (must remove those
        # ones that would collide with already ongoing flows

        # Get prefix from host ip
        src_gw = self.getGatewayRouter(flow['src'])
        src_gw_name = self.dc_graph.get_router_name(src_gw)
        dst_prefix = self.getMatchingPrefix(flow['dst'])

        # Get ongonig flows to dst_prefix from the same gateway
        ongoingFlowsSameGateway = self.getongoingFlowsFromGateway(src_gw, dst_prefix)
        if ongoingFlowsSameGateway:
            src_gw_name = self.dc_graph.get_router_name(src_gw)
            log.info("There are already ongoing flows to {0} starting at gateway {1}. We CAN NOT modify DAG".format(dst_prefix, src_gw_name))
            chosen_core = self.getCurrentCoreForFlow(ongoingFlowsSameGateway[0])
            chosen_path = self.getPathFromEdgeToCore(src_gw, chosen_core)

        else:
            # Compute the available cores
            available_cores = self.getAvailableCorePaths(src_gw, flow)

            # If no available cores found, we leave the flow to
            if available_cores == []:
                log.error("No available & no non-colliding-only cores found for flow. Doing nothing in this case...")
                raise EnvironmentError

            # Log a bit
            ac = [self.dc_graph.get_router_name(c['core']) for c in available_cores]
            log.info("Available cores: {0}".format(ac))

            # Choose core
            chosen = self.chooseCore(available_cores)

            # Extract data
            chosen_core = chosen['core']
            chosen_core_name = self.dc_graph.get_router_name(chosen_core)
            chosen_path = chosen['path']
            chosen_capacity = chosen['capacity']

            # Log a bit
            log.info("{0} was chosen with an available capacity of {1}".format(chosen_core_name, self.base.setSizeToStr(chosen_capacity)))

            # Retrieve current DAG for destination prefix
            current_dag = self.dags[dst_prefix]['dag']

            # Apply path
            current_dag.apply_path_to_core(src_gw, chosen_core)

            # Apply DAG
            self.sbmanager.add_dag_requirement(prefix=dst_prefix, dag=current_dag)
            src_gw_name = self.dc_graph.get_router_name(src_gw)
            chosen_core_name = self.dc_graph.get_router_name(chosen_core)
            log.info("A new path was forced from {0} -> {1} for prefix {2}".format(src_gw_name, chosen_core_name, dst_prefix))

        # Append flow to state variables
        self.addFlowToPath(flow, chosen_path)

        #import ipdb; ipdb.set_trace()

    def deallocateFlow(self, flow):
        #import ipdb; ipdb.set_trace()

        # Get prefix from flow
        (src_prefix, dst_prefix) = self.getPrefixesFromFlow(flow)

        # Get gateway router and pod
        src_gw = self.getGatewayRouter(src_prefix)
        src_pod = self.dc_graph.get_router_pod(src_gw)

        # Get to which core the flow was directed to
        core = self.getCoreFromFlow(flow)

        # Compute its current path
        current_path = self.getPathFromEdgeToCore(src_gw, core)

        # Remove it from the path!!!!!
        self.removeFlowFromPath(flow, current_path)

        # Get ongoing flows from the same pod to the dst_prefix
        ongoing_flows_same_pod = self.getOngoingFlowsFromPod(src_pod, dst_prefix)

        if not ongoing_flows_same_pod:
            # Log a bit
            log.debug("No ongoing flows from the same pod {0} we found to prefix {1}".format(src_pod, dst_prefix))

            # Retrieve current DAG for destination prefix
            current_dag = self.dags[dst_prefix]['dag']

            # Restore the DAG
            current_dag.set_ecmp_uplinks_from_source(src_gw, current_path, all_layers=True)

            # Apply DAG
            self.sbmanager.add_dag_requirement(prefix=dst_prefix, dag=current_dag)

            # Log a bit
            src_gw_name = self.dc_graph.get_router_name(src_gw)
            log.info("A new DAG was forced restoring the default ECMP DAG from source {0} -> {1}".format(src_gw_name, dst_prefix))

        # There are ongoing flows to same destination from same pod
        else:
            # Log a bit
            log.debug("There are colliding flows from the same pod {0} we found to prefix {1}".format(src_pod, dst_prefix))

            # Check if any of them comes from the same source as the terminating flow
            collide_same_aggregation = any([True for f in ongoing_flows_same_pod if self.getSourceGatewayFromFlow(f) == src_gw])
            if not collide_same_aggregation:
                # Log a bit
                log.debug("There are NO colliding flows with same edge router gateway: {0}".format(src_gw))

                # Retrieve current DAG for destination prefix
                current_dag = self.dags[dst_prefix]['dag']

                # Restore the DAG
                current_dag.set_ecmp_uplinks_from_source(src_gw, current_path, all_layers=False)

                # Apply DAG
                self.sbmanager.add_dag_requirement(prefix=dst_prefix, dag=current_dag)

                # Log a bit
                src_gw_name = self.dc_graph.get_router_name(src_gw)
                log.info("A new DAG was forced restoring PART OF the ECMP DAG from source {0} -> {1}".format(src_gw_name, dst_prefix))

            # There are colliding flows from the same source edge router
            else:
                # Log a bit
                log.debug("There are colliding flows with same edge gateway router {0}. We CAN NOT restore original DAG yet".format(src_gw))

    def handleFlow(self, event):
        log.info("Event to handle received: {0}".format(event["type"]))
        if event["type"] == 'startingFlow':
            flow = event['flow']
            self.allocateFlow(flow)
        elif event["type"] == 'stoppingFlow':
            flow = event['flow']
            self.deallocateFlow(flow)
        else:
            log.error("Unknown event type: {0}".format(event['type']))

    def reset(self):
        # Reset parent class first
        reset_time = super(CoreChooserController, self).reset()

        # Create structure where we store the ongoing elephant  flows in the graph
        self.elephants_in_paths = self._createElephantInPathsDict()

        # Store all paths --for performance reasons
        self.edge_to_core_paths = self._generateEdgeToCorePaths()

        # Keeps track to which core is each flow directed to
        self.flow_to_core = self._generateFlowToCoreDict()

        log.debug("Time to perform the reset to the load balancer: {0}s".format(time.time() - reset_time))

class BestRankedCoreChooser(CoreChooserController):
    def __init__(self, doBalance=True, k=4, threshold=0.9):
        super(BestRankedCoreChooser, self).__init__(doBalance, k, threshold, algorithm="best-ranked-core")

    def chooseCore(self, available_cores):
        log.info("Choosing available core: HIGHEST AVAILABLE CAPACITY")

        # Sort available cores from more to less available capacity
        sorted_available_cores = sorted(available_cores, key=lambda x: x['capacity'], reverse=True)

        # Take the first one
        chosen = sorted_available_cores[0]

        return chosen

class RandomCoreChooser(CoreChooserController):
    def __init__(self, doBalance=True, k=4, threshold=0.9):
        super(RandomCoreChooser, self).__init__(doBalance, k, threshold, algorithm="random-core")

    def chooseCore(self, available_cores):
        log.info("Choosing available core: AT RANDOM")

        # Shuffle list of av.cores
        random.shuffle(available_cores)

        #Take the first one
        chosen = available_cores[0]

        return chosen

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

    elif args.algorithm == 'random_dags':
        lb = RandomUplinksController(doBalance=args.doBalance, k=args.k)

    elif args.algorithm == 'random_core':
        lb = RandomCoreChooser(doBalance=args.doBalance, k=args.k)

    elif args.algorithm == 'best_ranked_core':
        lb = BestRankedCoreChooser(doBalance=args.doBalance, k=args.k)

    # Run the controller
    lb.run()


    #import ipdb; ipdb.set_trace()
