#!/usr/bin/python

import threading
import os
import time
import copy
import argparse
import json
import networkx as nx
from threading import Thread
import ipaddress as ip
import random
import subprocess
import abc
import Queue
import numpy as np
import sys
import itertools as it

from fibbingnode.algorithms.southbound_interface import SouthboundManager
from fibbingnode import CFG as CFG_fib

from fibte.misc.unixSockets import UnixServerTCP, UnixServer, UnixClient, UnixClientTCP
from fibte.trafficgen.flow import Base
from fibte.misc.dc_graph import DCGraph
from fibte.loadbalancer.mice_estimator import MiceEstimatorThread
from fibte.misc.topology_graph import TopologyGraph
from fibte.monitoring.getLoads import GetLoads
from fibte import tmp_files, db_topo, LINK_BANDWIDTH, UDS_server_name, UDS_server_traceroute, C1_cfg, getLoads_path

# Threading event to signal that the initial topo graph
# has been received from the Fibbing controller
HAS_INITIAL_GRAPH = threading.Event()

import logging
from fibte.logger import log
import fibte.misc.ipalias as ipalias
from fibte import time_func

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
    def __init__(self, doBalance = True, k=4, algorithm=None, load_variables=True):
        # Set fat-tree parameter
        self.k = k

        # Either we load balance or not
        self.doBalance = doBalance

        # Loadbalancing strategy/algorithm
        self.algorithm = algorithm

        # Configure logging
        self._do_logging_stuff()

        # Unix domain server to make things faster and possibility to communicate with hosts
        self._address_server = os.path.join(tmp_files, UDS_server_name)
        self.q_server = Queue.Queue(0)
        self.server = UnixServerTCP(self._address_server, self.q_server)

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

        # Remove old DAG requirements
        self.sbmanager.remove_all_dag_requirements()

        # Load the topology
        self.topology = TopologyGraph(getIfindexes=True, interfaceToRouterName=True, db=os.path.join(tmp_files, db_topo))

        # Get dictionary where loads are stored
        self.link_loads = self.topology.getEdgesUsageDictionary()
        self.link_loads_lock = threading.Lock()

        # Receive network graph
        self.fibbing_network_graph = self.sbmanager.igp_graph

        # Create my modified version of the graph
        self.dc_graph_elep = DCGraph(k=self.k, prefix_type='primary')
        self.dc_graph_mice = DCGraph(k=self.k, prefix_type='secondary')

        # Here we store the current DAGs for each destiantion
        self.current_elephant_dags = self._createInitialElephantDags()
        self.current_mice_dags = self._createInitialMiceDags()

        # We keep a copy of the initial ones
        self.initial_elep_dags = copy.deepcopy(self.current_elephant_dags)
        self.initial_mice_dags = copy.deepcopy(self.current_mice_dags)

        # Fill ospf_prefixes dict
        self.ospf_prefixes = self._fillInitialOSPFPrefixes()

        # Flow to path allocations
        self.flows_to_paths = {}
        self.edges_to_flows = {(a, b): [] for (a, b) in self.dc_graph_elep.edges()}
        # Lock used to modify these two variables
        self.add_del_flow_lock = threading.Lock()

        # Start getLoads thread that reads from counters
        self.p_getLoads = subprocess.Popen([getLoads_path, '-k', str(self.k), '-a', self.algorithm], shell=False)
        #os.system(getLoads_path + ' -k {0} &'.format(self.k))

        # Start getLoads thread reads from link usage
        thread = threading.Thread(target=self._getLoads, args=([1]))
        thread.setDaemon(True)
        thread.start()

        # Object useful to make some unit conversions
        self.base = Base()

        # Start mice estimator thread
        self._startMiceEstimatorThread()

        # UDS server where we listen for the traceroute data
        self.traceroute_server = UnixServer(os.path.join(tmp_files, UDS_server_traceroute))

        # UDS client used to instruct host to start traceroute path discovery
        self.traceroute_client = UnixClient(os.path.join(tmp_files, "/tmp/tracerouteServer_{0}"))

        # Start thread that listends for results
        thread = threading.Thread(target=self.traceroute_listener, args=())
        thread.setDaemon(True)
        thread.start()

        # Send initial data to flowServers
        self.traceroute_sendInitialData()

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

    @staticmethod
    def get_links_from_path(path):
        return zip(path[:-1], path[1:])

    @staticmethod
    def flowToKey(flow):
        """Fastest way to create a dictionary key out of a dictionary
        """
        return tuple(sorted(flow.items()))

    @staticmethod
    def keyToFlow(key):
        """Given a flow key, returns the flow as a dictionary"""
        return dict(key)

    def _startMiceEstimatorThread(self):
        # Here we store the estimated mice levels
        self.hosts_notified = []
        self.total_hosts = ((self.k/2)**2)*self.k
        self.mice_dbs = {}
        self.mice_dbs_lock = threading.Lock()
        self.mice_caps_graph = self._createElephantsCapsGraph()
        self.mice_caps_lock = threading.Lock()
        self.mice_orders_queue= Queue.Queue()
        self.mice_result_queue = Queue.Queue()

        # Create the mice estimator thread
        self.miceEstimatorThread = MiceEstimatorThread(sbmanager= self.sbmanager,
                                                       orders_queue = self.mice_orders_queue,
                                                       results_queue = self.mice_result_queue,
                                                       mice_distributions = self.mice_dbs,
                                                       mice_distributions_lock = self.mice_dbs_lock,
                                                       capacities_graph = self.mice_caps_graph,
                                                       capacities_lock = self.mice_caps_lock,
                                                       dags = self.current_mice_dags,
                                                       samples = 50)
        # Start the thread
        self.miceEstimatorThread.start()

    def traceroute_sendInitialData(self):
        """
        :return:
        """
        log.info("Sending initial data to traceroute servers")
        hosts_couldnt_inform = []
        for host in self.topology.getHosts().keys():
            # Get hosts in the same pod
            gw_name = self.topology.hostToGatewayMapping['hostToGateway'][host]
            gw_pod = self.topology.getRouterPod(gw_name)
            own_pod_ers = [r for r in self.topology.getRouters().keys() if self.topology.isEdgeRouter(r) and self.topology.getRouterPod(r) == gw_pod]
            own_pod_hosts = [self.topology.getHostIp(h) for r in own_pod_ers for h in self.topology.hostToGatewayMapping['gatewayToHosts'][r]]

            # Send list to traceroute client
            try:
                # Send command
                command = {'type': 'own_pod_hosts', 'data': own_pod_hosts}
                self.traceroute_client.send(json.dumps(command), host)
            except Exception as e:
                hosts_couldnt_inform.append(host)

        if hosts_couldnt_inform:
            log.error("The following flowServers could not be contacted: {0}".format(hosts_couldnt_inform))

    def traceroute_listener(self):
        """
        This function is meant to be executed in a separate thread.
        It iteratively reads for new traceroute results, that are
        sent by flowServers, and saves the results in a data
        structure
        """
        # Unix Client to communicate to controller
        self.client_to_self = UnixClientTCP("/tmp/controllerServer")

        while True:
            # Get a new route
            try:
                # Reads from the
                traceroute_data = json.loads(self.traceroute_server.sock.recv(65536))

                # Extract flow
                flow = traceroute_data['flow']

                # Extracts the router traceroute data
                route = traceroute_data['route']

                # Traceroute fast is used
                if route:
                    if len(route) == 1:
                        router_name = self.get_router_name(route[0])

                        # Convert it to ip
                        router_ip = self.topology.getRouterId(router_name)

                        # Compute whole path
                        path = self.computeCompletePath(flow=flow, router=router_ip)

                    # Complete traceroute is used
                    else:
                        # Extracts route from traceroute data
                        route_names = self.ipPath_to_namePath(traceroute_data)
                        path = [self.topology.getRouterId(rname) for rname in route_names]

                    # Add flow->path allocation
                    if not self.isOngoingFlow(flow):
                        # Add flow to path
                        self.addFlowToPath(flow, path)
                    else:
                        # Log a bit
                        src_n = self.topology.getHostName(flow['src'])
                        dst_n = self.topology.getHostName(flow['dst'])
                        size_n = self.base.setSizeToStr(flow['size'])
                        log.warning("Path couldn't be found for Flow({0} -> {1}: {2}). Traceroute again!".format(src_n, dst_n, size_n))

                        # Update its path
                        self.updateFlowPath(flow, path)

                else:
                    # Couldn't find route for flow yet...
                    self.tracerouteFlow(flow)

            except Exception:
                import traceback
                print traceback.print_exc()

    def computeCompletePath(self, flow, router):
        """
        Given a flow and a router, which can be either a core or aggregation router, returns the complete path of
        that flow in case that router was traversed.
        """
        if self.isInterPodFlow(flow):
            if not self.dc_graph_elep.is_core(routerid=router):
                raise ValueError("Can't compute complete path from inter-pod flow if given router is not core: {0}".format(router))

        else:
            if not self.dc_graph_elep.is_aggregation(routerid=router):
                raise ValueError("Can't compute complete path from intra-pod flow if given router is not aggregation: {0}".format(router))

        # Compute first gateway routers
        src_px, dst_px = self.getPrefixesFromFlow(flow)
        src_gw = self.getGatewayRouter(src_px)
        dst_gw = self.getGatewayRouter(dst_px)

        # Compute path
        src_to_core = nx.shortest_path(self.dc_graph_elep, src_gw, router)
        core_to_dst = nx.shortest_path(self.dc_graph_elep, router, dst_gw)
        complete_path = src_to_core + core_to_dst[1:]
        return complete_path

    def isInterPodFlow(self, flow):
        """Returns true if flow is an inter-pod router,
         and False if it is intra pod router
        """

        # Compute first gateway routers
        src_px, dst_px = self.getPrefixesFromFlow(flow)
        src_gw = self.getGatewayRouter(src_px)
        dst_gw = self.getGatewayRouter(dst_px)

        return self.dc_graph_elep.get_router_pod(src_gw) != self.dc_graph_elep.get_router_pod(dst_gw)

    def isFlowToSameNetwork(self, flow):
        src_px, dst_px = self.getPrefixesFromFlow(flow)
        src_gw = self.getGatewayRouter(src_px)
        dst_gw = self.getGatewayRouter(dst_px)
        return src_gw == dst_gw

    def get_router_name(self, addr):
        """Addr can either the router id, the interface ip or the private ip"""
        for fun in [self.topology.getRouterName, self.topology.getRouterFromPrivateIp, self.topology.getRouterFromInterfaceIp]:
            try:
                return fun(addr)
            except KeyError:
                continue
        return ValueError("{0} is neither a private ip, router id or interface ip".format(addr))

    def ipPath_to_namePath(self, traceroute_data):
        """
        Converts traceroute information from ip to router names so we get the path
        """
        route = traceroute_data['route']
        elephant_flow = traceroute_data["flow"]

        path = []

        notComplete=False
        if route:
            srcRouter = self.get_router_name(route[0])
            dstRouter = self.get_router_name(route[-1])

            # Check if the first and last routers are the correct gateways:
            src_gw_ok = self.topology.getGatewayRouter(self.topology.getHostName(elephant_flow["src"])) == srcRouter
            dst_gw_ok = self.topology.getGatewayRouter(self.topology.getHostName(elephant_flow["dst"])) == dstRouter

            if src_gw_ok and dst_gw_ok:
                for i, router_ip in enumerate(route):
                    # Get router name
                    router_name = self.get_router_name(router_ip)

                    # If we are not the first or last router
                    if not(router_name == srcRouter):
                        # Check connectivity
                        if self.topology.areNeighbors(router_name, path[-1]):
                            path.append(router_name)
                        else:
                            notComplete = True
                            break

                    # Source router
                    else:
                        path.append(router_name)

                # Return path if complete
                if not(notComplete):
                    return path
        return None

    def _createElephantsCapsGraph(self):
        graph = DCGraph(k=self.k, prefix_type='secondary')
        for (u, v, data) in graph.edges_iter(data=True):
            data['elephants_capacity'] = 0
        return graph

    def _printPath(self, path):
        return [self.dc_graph_elep.get_router_name(n) for n in path]

    def _do_logging_stuff(self):
        # Config logging to dedicated file for this thread
        handler = logging.FileHandler(filename='{0}loadbalancer_{1}.log'.format(tmp_files, self.algorithm))
        fmt = logging.Formatter('[%(levelname)20s] %(asctime)s %(funcName)s: %(message)s ')
        handler.setFormatter(fmt)
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)

    def _getGatewayRouter(self, prefix):
        """
        Given a prefix, returns the connected edge router
        :param prefix:
        :return:
        """
        # Convert it to primary ip prefix if needed
        if ipalias.is_secondary_ip_prefix(prefix):
            prefix = ipalias.get_primary_ip_prefix(prefix)

        if self.fibbing_network_graph.is_prefix(prefix):
            pred = self.fibbing_network_graph.predecessors(prefix)
            if len(pred) == 1 and not self.fibbing_network_graph.is_fake_route(pred[0], prefix):
                return pred[0]
            else:
                real_gw = [p for p in pred if not self.fibbing_network_graph.is_fake_route(p, prefix)]
                if len(real_gw) == 1:
                    return real_gw[0]
                else:
                    log.error("This prefix has several predecessors: {0}".format(prefix))
                    import ipdb; ipdb.set_trace()
        else:
            raise ValueError("{0} is not a prefix!".format(prefix))

    def getGatewayRouter(self, prefix):
        """
        Checks if is already stored. Otherwise calls _getGatewayRouter()
        :param prefix:
        :return:
        """
        if '/' in prefix:
            # If elephant ip prefix
            if not ipalias.is_secondary_ip_prefix(prefix):
                if prefix in self.current_elephant_dags.keys() and self.current_elephant_dags[prefix].has_key(
                        'gateway'):
                    return self.current_elephant_dags[prefix]['gateway']
                else:
                    # Maybe it's a newly created prefix -- so check in the network graph
                    return self._getGatewayRouter(prefix)

            # If mise ip prefix
            else:
                if prefix in self.current_mice_dags.keys() and self.current_mice_dags[prefix].has_key('gateway'):
                    return self.current_mice_dags[prefix]['gateway']
                else:
                    # Maybe it's a newly created prefix -- so check in the network graph
                    return self._getGatewayRouter(prefix)

        else:
            # Must be an ip then
            ipe = prefix
            ipe_prefix = self.getMatchingPrefix(ipe)
            return self.getGatewayRouter(ipe_prefix)

    def getCurrentDag(self, prefix):
        """"""
        return self.current_elephant_dags[prefix]['dag']

    def saveCurrentDag(self, prefix, dag, fib=False):
        """"""
        self.current_elephant_dags[prefix]['dag'] = dag
        if fib:
            self.sbmanager.add_dag_requirement(prefix, dag)

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

    def getSourcePodFromFlow(self, flow):
        """Returns the pod of the originator of the flow"""
        flow_gw = self.getGatewayRouter(self.getSourcePrefixFromFlow(flow))
        return self.dc_graph_elep.get_router_pod(flow_gw)

    def getConnectedPrefixes(self, edgeRouter):
        """
        Given an edge router, returns the connected prefixes
        :param edgeRouter:
        :return:
        """
        if self.dc_graph_elep.is_edge(edgeRouter):
            return [r for r in self.fibbing_network_graph.successors(edgeRouter) if self.fibbing_network_graph.is_prefix(r)]
        else:
            raise ValueError("{0} is not an edge router".format(edgeRouter))

    def _createInitialElephantDags(self):
        """There is one dag for each default network prefix in the network"""
        # Result is stored here
        log.info("Creating initial DAGs for the elephant prefixes in the network")
        dags = {}

        # Need to refresh lsas?
        need_to_refresh_lsas = False

        for prefix in self.fibbing_network_graph.prefixes:
            if not ipalias.is_secondary_ip_prefix(prefix):
                # Get prefix gateway router
                gatewayRouter = self._getGatewayRouter(prefix)

                # Compute initial dag (default OSPF)
                dc_dag = self.dc_graph_elep.get_default_ospf_dag(prefix)

                # Add dag
                dags[prefix] = {'gateway': gatewayRouter, 'dag': dc_dag}

                # If there are old requirements for prefix
                old_requirements = prefix in self.sbmanager.fwd_dags.keys()
                if old_requirements:
                    self.sbmanager.fwd_dags.pop(prefix)
                    need_to_refresh_lsas = True

        # Refresh LSA
        if need_to_refresh_lsas:
            self.sbmanager.refresh_lsas()

        return dags

    def _createInitialMiceDags(self):
        """
        Populates the self.current_mice_dags dictionary
        for each existing prefix in the network
        """
        # Result is stored here
        log.info("Creating initial DAGs for mice prefixes in the network")
        dags = {}

        # Need to refresh lsas?
        need_to_refresh_lsas = False

        for prefix in self.fibbing_network_graph.prefixes:
            if not ipalias.is_secondary_ip_prefix(prefix):
                gatewayRouter = self._getGatewayRouter(prefix)
                prefix = ipalias.get_secondary_ip_prefix(prefix)
                dc_dag = self.dc_graph_mice.get_default_ospf_dag(prefix)

            else:
                original_prefix = ipalias.get_primary_ip_prefix(prefix)
                gatewayRouter = self._getGatewayRouter(original_prefix)
                dc_dag = self.dc_graph_mice.get_default_ospf_dag(prefix)

            # Add dag
            if prefix not in dags:
                dags[prefix] = {'gateway': gatewayRouter, 'dag': dc_dag}

            # If there are old requirements for prefix
            old_requirements = prefix in self.sbmanager.fwd_dags.keys()
            if old_requirements:
                self.sbmanager.fwd_dags.pop(prefix)
                need_to_refresh_lsas = True

        # Refresh LSA
        if need_to_refresh_lsas:
            self.sbmanager.refresh_lsas()

        return dags

    def _fillInitialOSPFPrefixes(self):
        """
        Fills up the data structure
        """
        prefixes = []
        fill = [prefixes.append(ip.ip_network(prefix)) for prefix in self.fibbing_network_graph.prefixes]
        return prefixes

    def reset(self):
        """
        Sets the load balancer to its initial state
        :return:
        """
        # Start crono
        reset_start_time = time.time()

        # Remove all attraction points and lsas
        # Set all dags to original ospf dag
        self.current_elephant_dags = copy.deepcopy(self.initial_elep_dags)
        self.current_mice_dags = copy.deepcopy(self.initial_mice_dags)

        # Reset all dags to initial state
        self.sbmanager.remove_all_dag_requirements()

        # Terminate pevious mice estimator thread
        self.miceEstimatorThread.orders_queue.put({'type':'terminate'})
        self.miceEstimatorThread.join()

        # Restart Mice Estimator Thread
        self._startMiceEstimatorThread()

        with self.add_del_flow_lock:
            self.flows_to_paths = {}
            self.edges_to_flows = {(a, b): [] for (a, b) in self.dc_graph_elep.edges()}

        return reset_start_time

    def getLinkCapacity(self, link):
        """Returns the difference between link bandwidth and
        the sum of all flows going through the link"""
        return LINK_BANDWIDTH - sum([f['size'] for f in self.edges_to_flows[link]])

    def getLinkLoad(self, link, exclude=[]):
        """
        Returns the current load of the link, assuming flows
        are limited by sender and receiver's NIC

        Counts only the sizes of the flows that are not present
        in the exclude list.
        """
        if self.edges_to_flows[link]:
            return sum([self.keyToFlow(f)['size'] for f in self.edges_to_flows[link] if f not in exclude])
        else:
            return 0.0

    def isOngoingFlow(self, flow):
        """Returns True if flow is in the data structures,
         meaning that the flow is still ongoing"""
        fkey = self.flowToKey(flow)

        with self.add_del_flow_lock:
            return fkey in self.flows_to_paths.keys()

    def addFlowToPath(self, flow, path):
        """Adds a new flow to path datastructures"""
        # Get key from flow
        fkey = self.flowToKey(flow)

        # Get lock
        with self.add_del_flow_lock:
            # Update flows_to_paths
            if fkey not in self.flows_to_paths.keys():
                self.flows_to_paths[fkey] = path

            else:
                log.error("Weird: flow was already in the data structure {0}... updating it".format(flow))
                self.client_to_self.send(json.dumps({"type": "sleep", 'data': 2000}), "")
                import ipdb; ipdb.set_trace()
                self.flows_to_paths[fkey] = path

            # Upadte links too
            for link in self.get_links_from_path(path):
                if link in self.edges_to_flows.keys():
                    self.edges_to_flows[link].append(fkey)
                else:
                    self.edges_to_flows[link] = [fkey]
                    log.warning("Weird: edge {0} didn't exist...".format(link))
                    self.client_to_self.send(json.dumps({"type": "sleep", 'data': 2000}), "")
                    import ipdb; ipdb.set_trace()

        # Get lock
        with self.mice_caps_lock:
            for (u, v) in self.get_links_from_path(path):
                self.mice_caps_graph[u][v]['elephants_capacity'] += flow['size']

        # Log a bit
        src_n = self.topology.getHostName(flow['src'])
        dst_n = self.topology.getHostName(flow['dst'])
        size_n = self.base.setSizeToStr(flow['size'])
        path_n = [self.topology.getRouterName(r) for r in path]
        log.info("Flow({0} -> {1}: {2}) added to path ({3})".format(src_n, dst_n, size_n, ' -> '.join(path_n)))

    def updateFlowPath(self, flow, new_path):
        """Updates path from an already existing flow"""
        # Get key from flow
        fkey = self.flowToKey(flow)

        with self.add_del_flow_lock:
            # Remove flow from old path first
            if fkey in self.flows_to_paths.keys():
                old_path = self.flows_to_paths.pop(fkey)
            else:
                old_path = []

            if old_path:
                for link in self.get_links_from_path(old_path):
                    if link in self.edges_to_flows.keys():
                        if fkey in self.edges_to_flows[link]:
                            self.edges_to_flows[link].remove(fkey)
                        else:
                            log.warning("Weird: flow wasn't in the data structure {0}".format(flow))
                            self.client_to_self.send(json.dumps({"type": "sleep", 'data': 2000}), "")
                            import ipdb; ipdb.set_trace()
                            pass
                    else:
                        log.warning("Weird: link {0} not in the data structure".format(link))
                        self.client_to_self.send(json.dumps({"type": "sleep", 'data': 2000}), "")
                        import ipdb;ipdb.set_trace()
                        self.edges_to_flows[link] = []
            else:
                log.warning("Weird: flow wasn't in the data structure {0}".format(flow))
                self.client_to_self.send(json.dumps({"type": "sleep", 'data': 2000}), "")
                import ipdb;ipdb.set_trace()

            # Add it to the new path
            if fkey not in self.flows_to_paths.keys():
                self.flows_to_paths[fkey] = new_path

            # Upadte links too
            for link in self.get_links_from_path(new_path):
                if link in self.edges_to_flows.keys():
                    self.edges_to_flows[link].append(fkey)
                else:
                    self.edges_to_flows[link] = [fkey]
                    log.warning("Weird: edge {0} didn't exist...".format(link))
                    self.client_to_self.send(json.dumps({"type": "sleep", 'data': 2000}), "")
                    import ipdb;ipdb.set_trace()

        # Update mice estimator structure
        with self.mice_caps_lock:
            # Remove flow from edges of old path
            for (u, v) in self.get_links_from_path(old_path):
                self.mice_caps_graph[u][v]['elephants_capacity'] -= flow['size']

            # Add it to new ones
            for (u, v) in self.get_links_from_path(new_path):
                self.mice_caps_graph[u][v]['elephants_capacity'] += flow['size']

        # Log a bit
        src_n = self.topology.getHostName(flow['src'])
        dst_n = self.topology.getHostName(flow['dst'])
        size_n = self.base.setSizeToStr(flow['size'])
        new_path_n = [self.topology.getRouterName(r) for r in new_path]
        old_path_n = [self.topology.getRouterName(r) for r in old_path]
        log.info("Flow({0} -> {1}: {2}) changed path from ({3}) to ({4})".format(src_n, dst_n, size_n, ' -> '.join(old_path_n), ' -> '.join(new_path_n)))

    def delFlowFromPath(self, flow):
        """Removes an ongoing flow from path"""

        # Get key from flow
        fkey = self.flowToKey(flow)

        with self.add_del_flow_lock:
            # Update data structures
            if fkey in self.flows_to_paths.keys():
                path = self.flows_to_paths.pop(fkey)
            else:
                path = []

            if path:
                for link in self.get_links_from_path(path):
                    if link in self.edges_to_flows.keys():
                        if fkey in self.edges_to_flows[link]:
                            self.edges_to_flows[link].remove(fkey)
                        else:
                            log.error("Weird: flow wasn't in the data structure {0}".format(flow))
                            self.client_to_self.send(json.dumps({"type": "sleep", 'data': 2000}), "")
                            import ipdb; ipdb.set_trace()
                    else:
                        log.error("Weird: link {0} not in the data structure".format(link))
                        self.client_to_self.send(json.dumps({"type": "sleep", 'data': 2000}), "")
                        import ipdb;ipdb.set_trace()
            else:
                log.error("Weird: flow wasn't in the data structure {0}".format(flow))
                self.client_to_self.send(json.dumps({"type": "sleep", 'data': 2000}), "")
                import ipdb;ipdb.set_trace()

        # Update mice estimator structure
        with self.mice_caps_lock:
            for (u, v) in self.get_links_from_path(path):
                self.mice_caps_graph[u][v]['elephants_capacity'] -= flow['size']

        # Log a bit
        src_n = self.topology.getHostName(flow['src'])
        dst_n = self.topology.getHostName(flow['dst'])
        size_n = self.base.setSizeToStr(flow['size'])
        path_n = [self.topology.getRouterName(r) for r in path]
        log.info("Flow({0} -> {1}: {2}) deleted from path ({3})".format(src_n, dst_n, size_n, ' -> '.join(path_n)))

        return path

    def allocateFlow(self, flow):
        """
        Subclass this method
        """
        src = self.topology.getHostName(flow['src'])
        dst = self.topology.getHostName(flow['dst'])
        size = self.base.setSizeToStr(flow['size'])
        log.debug("New flow STARTED: Flow({0} -> {1}: {2})".format(src, dst, size))

    def deallocateFlow(self, flow):
        """
        Subclass this method
        """
        src = self.topology.getHostName(flow['src'])
        dst = self.topology.getHostName(flow['dst'])
        size = self.base.setSizeToStr(flow['size'])
        log.debug("Flow FINISHED: Flow({0} -> {1}: {2})".format(src, dst, size))

    def tracerouteFlow(self, flow):
        """Starts a traceroute"""
        # Get prefix from host ip
        src_name = self.topology.getHostName(flow['src'])
        try:
            # Send instruction to start traceroute to specific flowServer
            command = {'type':'flow', 'data': flow}
            self.traceroute_client.send(json.dumps(command), src_name)

        except Exception as e:
            log.error("{0} could not be reached. Exception: {1}".format(src_name, e))

    def handleFlow(self, event):
        """
         Default handle flow skeleton
         :param event:
         :return:
         """
        if event["type"] == 'startingFlow':
            flow = event['flow']
            self.allocateFlow(flow)

        elif event["type"] == 'stoppingFlow':
            flow = event['flow']
            self.deallocateFlow(flow)

        else:
            log.error("Unknown event type: {0}".format(event['type']))
            import ipdb; ipdb.set_trace()

    def handleMiceEstimation(self, estimation_data):
        """"""
        src_ip = self.topology.hostsIpMapping['nameToIp'][estimation_data['src']]
        src_px = self.getSourcePrefixFromFlow({'src': src_ip})
        src_mice_px = ipalias.get_secondary_ip_prefix(src_px)
        if src_mice_px in self.hosts_notified:
            #log.error("Notification catch up!")
            self.notified_flows = []

        self.hosts_notified.append(src_mice_px)

        # Get the lock first
        with self.mice_dbs_lock:
            # Get per-destination samples
            dst_samples = estimation_data['samples']
            for (dst_ip, samples) in dst_samples.iteritems():
                dst_px = self.getDestinationPrefixFromFlow({'dst': dst_ip})
                dst_mice_px = ipalias.get_secondary_ip_prefix(dst_px)
                if dst_mice_px not in self.mice_dbs.keys():
                    self.mice_dbs[dst_mice_px] = {}

                samples = np.asarray(samples)
                avg, std = samples.mean(), samples.std()

                # Add new average and std values
                if src_mice_px not in self.mice_dbs[dst_mice_px].keys():
                    self.mice_dbs[dst_mice_px][src_mice_px] = {'avg': [avg], 'std': [std]}
                else:
                    self.mice_dbs[dst_mice_px][src_mice_px]['avg'].append(avg)
                    self.mice_dbs[dst_mice_px][src_mice_px]['std'].append(std)

        # Check if all hosts have notified this round
        if len(self.hosts_notified) == self.total_hosts:
            # Reset list of hosts that have notified
            self.hosts_notified = []

            # Order mice estimator thread to update its distributoins
            order = {'type': 'propagate_new_distributions'}
            self.miceEstimatorThread.orders_queue.put(order)

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

    def exitGracefully(self):
        """
        Exit load balancer gracefully
        :return:
        """
        log.info("Keyboad Interrupt catched!")

        # Remove all lies before leaving
        log.info("Cleaning up the network from fake LSAs ...")
        self.sbmanager.remove_all_dag_requirements()

        # self.p_getLoads.terminate()

        # Finally exit
        os._exit(0)

    def run(self):
        # Receive events and handle them
        log.info("Looping for new events!")

        # Start process that handles connections to the TCP server
        tcp_server_thread = Thread(target = self.server.run)
        tcp_server_thread.setDaemon(True)
        tcp_server_thread.start()

        # Loop for new events
        while True:
            try:
                if not(self.doBalance):
                    while True:
                        # Read from TCP server's queue
                        event = json.loads(self.q_server.get())
                        self.q_server.task_done()
                        log.info("LB not active - event received: {0}".format(json.loads(event)))
                else:
                    # Read from TCP server's queue
                    event = json.loads(self.q_server.get())
                    self.q_server.task_done()

                    if event["type"] == "reset":
                        log.info("RESET event received")
                        self.reset()
                        continue

                    elif event["type"] in ["startingFlow", "stoppingFlow"]:
                        log.info("{0} event received".format(event['type']))
                        self.handleFlow(event)

                    elif event["type"] == "miceEstimation":
                        estimation_data = event['data']
                        self.handleMiceEstimation(estimation_data)

                    elif event['type'] == 'sleep':
                        log.info("Sleep order received. Sleeping for {0} seconds...".format(event['data']))
                        time.sleep(event['data'])
                        log.info("WOKE UP!")
                    else:
                        log.error("Unknown event: {0}".format(event))

            except KeyboardInterrupt:
                # Exit load balancer
                self.exitGracefully()

class ECMPController(LBController):
    def __init__(self, doBalance=True, k=4):
        super(ECMPController, self).__init__(doBalance=doBalance, k=k, algorithm='ecmp')

    def allocateFlow(self, flow):
        """Just checks the default route for this path,
        and uptades a flow->path data structure"""
        #super(ECMPController, self).allocateFlow(flow)

        # Check if flow is within same subnetwork
        if self.isFlowToSameNetwork(flow):
            # No need to loadbalance anything
            # For TCP: we need to update the matrix
            return

        # Get default path of flow
        self.tracerouteFlow(flow)

    def deallocateFlow(self, flow):
        """Removes flow from flow->path data structure"""
        #super(ECMPController, self).deallocateFlow(flow)

        # Check if flow is within same subnetwork
        if self.isFlowToSameNetwork(flow):
            # No need to loadbalance anything
            # For TCP: we need to update the matrix
            return

        # Remove flow from path
        path = self.delFlowFromPath(flow)

class DAGShifterController(LBController):
    def __init__(self, doBalance=True, k=4, capacity_threshold=1, congProb_threshold=0.20):
        super(DAGShifterController, self).__init__(doBalance=doBalance, k=k, algorithm='dag-shifter')

        # We consider congested a link more than threshold % of its capacity
        self.capacity_threshold = capacity_threshold
        self.congestion_threshold = self.capacity_threshold*LINK_BANDWIDTH
        self.congProb_threshold = congProb_threshold
        self.dag_samples = 81

    @time_func
    def allocateFlow(self, flow):
        super(DAGShifterController, self).allocateFlow(flow)

        # Check if flow is within same subnetwork
        if self.isFlowToSameNetwork(flow):
            # No need to loadbalance anything
            # For TCP: we need to update the matrix
            return

        # Get matching destination
        dst_px = self.getMatchingPrefix(flow['dst'])

        # Get current DAG
        cdag = self.getCurrentDag(dst_px)

        # Compute congestion probability of current DAG with new flow
        congProb = self.flowCongestionProbability(flow, dag=cdag)
        log.info("Flow congestion probability: {0}%".format(congProb*100))

        # If above the threshold
        if congProb > self.congProb_threshold:
            # Get source pod
            src_pod = self.getSourcePodFromFlow(flow)

            # Get the flows originating at same pod towards same destination
            dst_ongoing_flows = self.getOngoingFlowsToDst(dst=dst_px, from_pod=src_pod)

            # Get dag with spare capacities
            idag = self.getInitialDagWithoutFlows(src_pod=src_pod, dst_px=dst_px, ongoing_flows=dst_ongoing_flows)

            # Add current flow to computations too
            dst_ongoing_flows = [self.keyToFlow(fkey) for fkey in dst_ongoing_flows]
            dst_ongoing_flows.append(flow)

            # Choose new DAG minimizing probability
            (best_dag, best_dag_cost) = self.findBestSourcePodDag(src_pod, idag, dst_ongoing_flows) or idag

            # Plot it
            src_name = self.topology.getHostName(flow['src'])
            dst_name = self.topology.getHostName(flow['dst'])
            size = self.base.setSizeToStr(flow['size'])

            img_name = './images/bestDag{0}_{1}_{2}.png'.format(src_name, dst_name, size)
            #best_dag.plot(img_name)

            # Log a bit
            log.info("Best DAG was found with a cost of: {0}".format(best_dag_cost))

            # Force it with Fibbing
            self.sbmanager.add_dag_requirement(dst_px, best_dag)


            time.sleep(0.02)

            # Finnally find new paths taken by flows in DAG
            for f in dst_ongoing_flows:
                self.tracerouteFlow(f)

        else:
            # Traceroute path taken by flow
            self.tracerouteFlow(flow)

    def deallocateFlow(self, flow):
        super(DAGShifterController, self).deallocateFlow(flow)

        # Check if flow is within same subnetwork
        if self.isFlowToSameNetwork(flow):
            # No need to loadbalance anything
            # For TCP: we need to update the matrix
            return

        # Deallocate if from the network
        self.delFlowFromPath(flow)

    @time_func
    def findBestSourcePodDag(self, src_pod, complete_dag, ongoing_flows):
        """"""
        # Start the generator of all possible random edge choices
        all_edge_choices = complete_dag.all_random_uplinks_iter(src_pod=src_pod)

        # Accumulate best result here
        best_dag = None
        best_eFlowCost = 0

        # Get egress router
        dst_gw = self.getGatewayRouter(self.getDestinationPrefixFromFlow(ongoing_flows[0]))

        # for i in range(20):
        #     ndag = complete_dag.copy()
        #
        #     # Modify corresponding uplinks
        #     ndag.modify_random_uplinks(src_pod=src_pod)
        #     ndag.plot('./images/random/epa_{0}.png'.format(i))

        # For each dag sample

        for i, edges_choice in enumerate(all_edge_choices):
            # Create a new dag of the complete dag with these edges only
            ndag = complete_dag.copy()

            # Modify corresponding uplinks
            ndag.modify_uplinks_from(edges_choice)

            # Compute
            eFlowCost = self.computeExpectedFlowsCost(ndag, dst_gw, ongoing_flows)

            if not best_dag:
                best_dag = ndag
                best_eFlowCost = eFlowCost

            else:
                if eFlowCost < best_eFlowCost:
                    best_dag = ndag
                    best_eFlowCost = eFlowCost

        return (best_dag, best_eFlowCost)

    #@time_func
    def computeExpectedFlowsCost(self, ndag, dst_gw, ongoing_flows):
        """
        Given a dag, and a set of ongoing flows, computes the expected sum of flow rates
        of these flows going through the dag

        :param ndag:
        :param dst_gw:
        :param ongoing_flows:
        :return:
        """
        # Compute all path combinations
        flows_src_gws = [self.getGatewayRouter(self.getSourcePrefixFromFlow(flow)) for flow in ongoing_flows]
        all_paths_flow = [nx.all_simple_paths(ndag, flows_src_gws[i], dst_gw) for i, flow in enumerate(ongoing_flows)]
        all_path_combinations = it.product(*all_paths_flow)

        # FlowsCost
        flows_cost = []

        # For each path combination
        for pcomb in all_path_combinations:

            # Compute flow congestion
            flowsCost = self.computeFlowsCost(ndag, ongoing_flows, pcomb)

            # Append it
            flows_cost.append(flowsCost)

        # Convert it to a numpy array
        flows_cost = np.asarray(flows_cost)

        # Return expected value
        return flows_cost.mean()

    def computeFlowsCost(self, ndag, ongoing_flows, path_combination):
        """Given a list of flows and a list repersentinc a path taken by each flow, returns
        the total overflow observed by the flows"""

        # Add the flow sizes to their paths first
        for i, flow in enumerate(ongoing_flows):
            fpath = path_combination[i]
            for (u, v) in self.get_links_from_path(fpath):
                # Remove previous one
                if i == 0 and ndag[u][v].has_key('virtual'):
                    ndag[u][v]['virtual'] = 0

                elif not ndag[u][v].has_key('virtual'):
                    # Add new one
                    ndag[u][v]['virtual'] = flow['size']

                else:
                    ndag[u][v]['virtual'] += flow['size']

        totalCost = 0
        # Compute overload on the flow paths
        for fpath in path_combination:
            for (u,v) in self.get_links_from_path(fpath):
                totalLoad = ndag[u][v]['load'] + ndag[u][v]['virtual']
                overLoad = abs(min(0, LINK_BANDWIDTH - totalLoad))
                totalCost += overLoad

        return totalCost

    @time_func
    def getInitialDagWithoutFlows(self, src_pod, dst_px, ongoing_flows):
        """
        Returns the dag that we need in order to compute the
        best dag search from a certain pod to a certain destination

        :param src_pod:
        :param dag:
        :param ongoing_flows:
        :return:
        """
        # Get initial DAG
        idag = self.initial_elep_dags[dst_px]['dag']
        idag_c = idag.copy()

        # Insert current loads into the edges of the graph
        with self.add_del_flow_lock:
            # Iterkeys
            for edge in self.edges_to_flows.iterkeys():
                # Check src and dst
                (u, v) = edge

                # Check if edge is from initial dag
                if edge in idag.edges():

                    # Get all the loads except those of ongoing flows
                    edge_load = self.getLinkLoad(edge, exclude=ongoing_flows)

                    # Insert it in the new dag
                    idag_c[u][v]['load'] = edge_load

        # Return the dag
        return idag_c

    def getOngoingFlowsToDst(self, dst, from_pod=None):
        """
        Return a list of flow keys that are going towards dst.
        If from_pod is none, it returs all the flows in the network.
        Otherwise, it returns only those flows towards dst starting
        at specified source pod
        """
        # Get destination gateway
        dst_gw = self.getGatewayRouter(dst)

        # Take gateway incoming edges
        incoming_edges = [(other, dst_gw) for other in self.dc_graph_elep.predecessors(dst_gw)]

        # Return result here
        dst_ongoing_flows = []

        # Return all the flows in these links that have source pod
        with self.add_del_flow_lock:
            # no pod filter
            if not from_pod:
                # All flows
                fws = [fkey for edge in incoming_edges for fkey in self.edges_to_flows[edge]
                       if self.getDestinationPrefixFromFlow(self.keyToFlow(fkey)) == dst]

            else:
                # Only flows from source pod
                fws = [fkey for edge in incoming_edges for fkey in self.edges_to_flows[edge]
                       if self.getSourcePodFromFlow(self.keyToFlow(fkey)) == from_pod
                       and self.getDestinationPrefixFromFlow(self.keyToFlow(fkey)) == dst]

            return fws

    @time_func
    def flowCongestionProbability(self, flow, dag):
        """"""
        paths_to_loads = self.pathLoadsAfterFlowInDag(flow, dag)

        total_paths = float(len(paths_to_loads.keys()))
        congested_paths = 0

        for path, capacities in paths_to_loads.iteritems():
            if any([c for c in capacities if c > self.congestion_threshold]):
                congested_paths += 1

        return congested_paths/total_paths

    def pathLoadsAfterFlowInDag(self, flow, dag):
        """
        Given a dag and a flow, computes the costs of all the paths
        possibly used by the flow, after allocating the flow

        :param flow:
        :param dag:
        :return:
        """
        # Get matching destination
        dst_px = self.getMatchingPrefix(flow['dst'])
        dst_gw = self.getGatewayRouter(dst_px)

        # Compute source gateway
        src_px = self.getSourcePrefixFromFlow(flow)
        src_gw = self.getGatewayRouter(src_px)

        # Get all paths
        paths = self.getAllPathsInDag(dag, src_gw, dst_gw, k=0)

        # Compute path capacities
        paths_to_loads = {tuple(path): self.getPathLoads(path) for path in paths}

        # Iterate paths anad add flow size
        for path in paths_to_loads.iterkeys():
            capacities = paths_to_loads[path]
            paths_to_loads[path] = [c + flow['size'] for c in capacities]

        return paths_to_loads

    def getPathLoads(self, path):
        """Returns current capacity used by the elephants in the path"""
        # Get links in path
        path_links = self.get_links_from_path(path)

        # Get capacities of links
        with self.add_del_flow_lock:
            link_capacities = map(self.getLinkLoad, path_links)

        return link_capacities

    def getAllPathsInDag(self, dag, start, end, k, path=[]):
        """Recursive function that finds all paths from start node to end node
        with maximum length of k.

        If the function is called with k=0, returns all existing
        loopless paths between start and end nodes.

        :param dag: nx.DiGraph representing the current paths towards
        a certain destination.

        :param start, end: string representing the ip address of the
        star and end routers (or nodes) (i.e:
        10.0.0.3).

        :param k: specified maximum path length (here means hops,
        since the dags do not have weights).
        """
        # Accumulate nodes in path
        path = path + [start]

        if start == end:
            # Arrived to the end. Go back returning everything
            return [path]

        if not start in dag:
            return []

        paths = []
        for node in dag[start]:
            if node not in path:  # Ommiting loops here
                if k == 0:
                    # If we do not want any length limit
                    newpaths = self.getAllPathsInDag(dag, node, end, k, path=path)
                    for newpath in newpaths:
                        paths.append(newpath)
                elif len(path) < k + 1:
                    newpaths = self.getAllPathsInDag(dag, node, end, k, path=path)
                    for newpath in newpaths:
                        paths.append(newpath)
        return paths

class TestController(object):
    def __init__(self):
        log.setLevel(logging.DEBUG)

        # Connects to the southbound controller. Must be called before
        # creating the instance of SouthboundManager
        CFG_fib.read(os.path.join(tmp_files, C1_cfg))

        # Start the Southbound manager in a different thread
        self.sbmanager = MyGraphProvider()
        t = threading.Thread(target=self.sbmanager.run, name="Southbound Manager")
        t.start()

        # Unix domain server to make things faster and possibility to communicate with hosts
        self._address_server = os.path.join(tmp_files, UDS_server_name)
        self.q_server = Queue.Queue(0)
        self.server = UnixServerTCP(self._address_server, self.q_server)

        # Blocks until initial graph received from SouthBound Manager
        HAS_INITIAL_GRAPH.wait()
        log.info("Initial graph received from SouthBound Controller")

        self.topology = TopologyGraph(db=os.path.join(tmp_files, db_topo))

        self.print_all_ips()

        upper = map(self.topology.getRouterId, ['r1', 'r3', 'r5'])
        middle = map(self.topology.getRouterId, ['r1', 'r5'])
        lower = map(self.topology.getRouterId, ['r1', 'r2', 'r4', 'r5'])

        d1_ip = self.topology.getHostIp('d1')
        d1_px = self.topology.hostsToNetworksMapping['hostToNetwork']['d1']['d1-eth0']

        dag = nx.DiGraph()
        dag.add_edges_from(self.get_links_from_path(path=upper))
        dag.add_edges_from(self.get_links_from_path(path=middle))
        self.sbmanager.add_dag_requirement(d1_px, dag)

        import ipdb; ipdb.set_trace()

        # Here we store the mice levels from each host to all other hosts
        self.mice_dbs = {}

    @staticmethod
    def get_links_from_path(path):
        return zip(path[:-1], path[1:])

    def print_all_ips(self):
        regex = "{0}\t->\t{1}"
        log.info("*** Routers")
        for name, rid in self.topology.routersIdMapping['nameToId'].iteritems():
            log.info(regex.format(name, rid))

        regex = "{0}\t->\t{1}, {2}"
        log.info("*** Hosts")
        for name, hip in self.topology.hostsIpMapping['nameToIp'].iteritems():
            log.info(regex.format(name, hip, ipalias.get_secondary_ip(hip)))

    def reset(self):
        pass

    def run(self):
        # Start process that handles connections to the TCP server
        tcp_server_thread = Thread(target = self.server.run)
        tcp_server_thread.setDaemon(True)
        tcp_server_thread.start()

        log.info("Looping for new events!")
        while True:
            try:
                while True:
                    # Read from TCP server queue
                    event = json.loads(self.q_server.get())
                    self.q_server.task_done()

                    log.info("Event received: {0}".format(event))
                    if event["type"] == "reset":
                        log.info("RESET event received")
                        self.reset()
                        continue

                    elif event["type"] in ["startingFlow", "stoppingFlow"]:
                        self.handleFlow(event)

                    elif event["type"] == "miceEstimation":
                        estimation_data = event['data']
                        self.handleMiceEstimation(estimation_data)

                    else:
                        log.error("Unknown event: {0}".format(event))
                        import ipdb; ipdb.set_trace()

            except KeyboardInterrupt:
                # Exit load balancer
                self.exitGracefully()

    @time_func
    def handleMiceEstimation(self, estimation_data):
        # Get mice source
        src = estimation_data['src']
        log.info("Received estimation data from {0}".format(src))

        # Get per-destination samples
        dst_samples = estimation_data['samples']
        for (dst, samples) in dst_samples.iteritems():
            if dst not in self.mice_dbs.keys():
                self.mice_dbs[dst] = {}

            samples = np.asarray(samples)
            avg, std = samples.mean(), samples.std()

            # Add new average and std values
            if src not in self.mice_dbs[dst].keys():
                self.mice_dbs[dst][src] = {'avg': [avg], 'std': [std]}
            else:
                self.mice_dbs[dst][src]['avg'].append(avg)
                self.mice_dbs[dst][src]['std'].append(std)

    def handleFlow(self, flow):
        src_ip = flow['src']
        dst_e_ip = flow['dst']

        dst_mice_px = [p for p in self.sbmanager.igp_graph.prefixes if ip.ip_address(dst_e_ip) in ip.ip_network(p)][0]
        dst_elep_px = ipalias.get_secondary_ip_prefix(dst_mice_px)

        long_path = [self.topology.routerid(r) for r in ('r1', 'r2', 'r4', 'r5')]
        short_path = [self.topology.routerid(r) for r in ('r1', 'r3', 'r5')]

        import ipdb; ipdb.set_trace()

        log.info("BEFORE mice -> long path")
        self.sbmanager.simple_path_requirement(dst_mice_px, long_path)
        log.info("AFTER mice -> long path")

        import ipdb; ipdb.set_trace()

        log.info("BEFORE eleph -> short path")
        self.sbmanager.simple_path_requirement(dst_elep_px, short_path)
        log.info("AFTER eleph -> short path")

        import ipdb; ipdb.set_trace()


        #self.sbmanager.simple_path_requirement(d1_elephant_px, short_path)
        #self.sbmanager.simple_path_requirement(d1_mice_px, long_path)

    def exitGracefully(self):
        """
        Exit load balancer gracefully
        :return:
        """
        log.info("Keyboad Interrupt catched!")

        self.sbmanager.remove_all_dag_requirements()

        # Finally exit
        os._exit(0)


if __name__ == '__main__':
    from fibte.logger import log
    import logging

    parser = argparse.ArgumentParser()

    parser.add_argument('--doBalance',
                        help='If set to False, ignores all events and just prints them',
                        action='store_true',
                        default = True)

    parser.add_argument('--algorithm',
                        help='Choose loadbalancing strategy',
                        choices=["ecmp", "dag-shifter"],
                        default = 'ecmp')

    parser.add_argument('-k', '--k', help='Fat-Tree parameter', type=int, default=4)

    parser.add_argument('-t', '--test', help='Test controller', action="store_true", default=False)

    parser.add_argument('--cong_prob', help='Threshold of congestion probability', type=float, default=0.1)

    args = parser.parse_args()

    log.setLevel(logging.DEBUG)
    log.info("Starting Controller - k = {0} , algorithm = {1}".format(args.k, args.algorithm))

    if args.test == True:
        lb = TestController()

    elif args.algorithm == 'ecmp':
        lb = ECMPController(doBalance = args.doBalance, k=args.k)

    elif args.algorithm == 'dag-shifter':
        lb = DAGShifterController(doBalance= args.doBalance, k=args.k, congProb_threshold=args.cong_prob)


    # Run the controller
    lb.run()


    #import ipdb; ipdb.set_trace()
