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

from fibbingnode.algorithms.southbound_interface import SouthboundManager
from fibbingnode import CFG as CFG_fib

from fibte.misc.unixSockets import UnixServerTCP, UnixServer, UnixClient
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
        while True:
            # Get a new route
            try:
                # Reads from the
                traceroute_data = json.loads(self.traceroute_server.sock.recv(65536))

                # Extract flow
                flow = traceroute_data['flow']

                # Extracts the router traceroute data
                route = traceroute_data['route']

                # traceroute fast is used
                if len(route) == 1:
                    router_name = self.get_router_name(route[0])

                    # Convert it to ip
                    router_ip = self.topology.getRouterId(router_name)

                    # Compute whole path
                    path = self.computeCompletePath(flow=flow, router=router_ip)

                # complete traceroute is used
                else:
                    # Extracts route from traceroute data
                    route_names = self.ipPath_to_namePath(traceroute_data)
                    path = [self.topology.getRouterId(rname) for rname in route_names]

                # Add flow->path allocation
                if not self.isOngoingFlow(flow):
                    # Add flow to path
                    self.addFlowToPath(flow, path)
                else:
                    # Update its path
                    self.updateFlowPath(flow, path)

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

        return reset_start_time

    def getLinkCapacity(self, link):
        """Returns the difference between link bandwidth and
        the sum of all flows going through the link"""
        return LINK_BANDWIDTH - sum([f['size'] for f in self.edges_to_flows[link]])

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
                self.flows_to_paths[fkey] = path
                log.error("Weird: flow was already in the data structure {0}... updating it".format(flow))

            # Upadte links too
            for link in self.get_links_from_path(path):
                if link in self.edges_to_flows.keys():
                    self.edges_to_flows[link].append(fkey)
                else:
                    self.edges_to_flows[link] = [fkey]
                    log.warning("Weird: edge {0} didn't exist...".format(link))

        # Get lock
        with self.mice_caps_lock:
            for (u, v) in self.get_links_from_path(path):
                self.mice_caps_graph[u][v]['elephants_capacity'] += flow['size']

        # Log a bit
        src_n = self.topology.getHostName(flow['src'])
        dst_n = self.topology.getHostName(flow['dst'])
        size_n = self.base.setSizeToStr(flow['size'])
        path_n = [self.topology.getRouterName(r) for r in path]
        log.info("Flow({0}->{1}: {2}) added to path ({3})".format(src_n, dst_n, size_n, ' -> '.join(path_n)))

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
                            pass
                    else:
                        log.warning("Weird: link {0} not in the data structure".format(link))
                        self.edges_to_flows[link] = []
            else:
                log.warning("Weird: flow wasn't in the data structure {0}".format(flow))

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
        log.info("Flow({0}->{1}: {2}) changed path from ({3}) to ({4})".format(src_n, dst_n, size_n, ' -> '.join(old_path_n), ' -> '.join(new_path_n)))

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
                    else:
                        log.error("Weird: link {0} not in the data structure".format(link))
            else:
                log.error("Weird: flow wasn't in the data structure {0}".format(flow))


        # Update mice estimator structure
        with self.mice_caps_lock:
            for (u, v) in self.get_links_from_path(path):
                self.mice_caps_graph[u][v]['elephants_capacity'] -= flow['size']

        # Log a bit
        src_n = self.topology.getHostName(flow['src'])
        dst_n = self.topology.getHostName(flow['dst'])
        size_n = self.base.setSizeToStr(flow['size'])
        path_n = [self.topology.getRouterId(r) for r in path]
        log.info("Flow({0}->{1}: {2}) deleted from path ({3})".format(src_n, dst_n, size_n, ' -> '.join(path_n)))

        return path

    def allocateFlow(self, flow):
        """
        Subclass this method
        """
        log.debug("New flow STARTED: {0}".format(flow))

    def deallocateFlow(self, flow):
        """
        Subclass this method
        """
        log.debug("Flow FINISHED: {0}".format(flow))

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
            log.error("Notification catch up!")
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
        super(ECMPController, self).allocateFlow(flow)

        # Get default path of flow
        self.tracerouteFlow(flow)

    def deallocateFlow(self, flow):
        """Removes flow from flow->path data structure"""
        super(ECMPController, self).deallocateFlow(flow)

        # Remove flow from path
        path = self.delFlowFromPath(flow)

        # Log a bit
        path_names = [self.topology.getRouterId(r) for r in path]
        [src, dst] = [self.topology.getHostName(a) for a in [flow['src'], flow['dst']]]
        log.info("Flow {0} -> {1} ({2}) allocated to {3}".format(src, dst, flow['size'], path_names))
        log.info("Flow {0} removed from {1}".format(flow, path))

class DAGShifterController(LBController):
    def __init__(self, doBalance=True, k=4, capacity_threshold=1, congProb_threshold=0.5):
        super(DAGShifterController, self).__init__(doBalance=doBalance, k=k, algorithm='dag-shifter')

        # We consider congested a link more than threshold % of its capacity
        self.capacity_threshold = capacity_threshold
        self.congProb_threshold = congProb_threshold

    def allocateFlow(self, flow):
        super(DAGShifterController, self).allocateFlow(flow)
        # Get matching destination
        dst_px = self.getMatchingPrefix(flow['dst'])

        # Compute congestion probability of current DAG with new flow
        congProb = self.flowCongestionProbability(flow)

        # If above the threshold
        if congProb > self.congProb_threshold:
            # Choose new DAG minimizing probability
            #self.findBestSourcePodDag(dst_px)
            pass
        else:
            # Leave current DAG
            pass

    def deallocateFlow(self, flow):
        super(DAGShifterController, self).deallocateFlow(flow)
        pass

    def flowCongestionProbability(self, flow):
        """"""
        # Get matching destination
        dst_px = self.getMatchingPrefix(flow['dst'])
        dst_gw = self.getGatewayRouter(dst_px)

        # Compute source gateway
        src_px = self.getSourcePrefixFromFlow(flow)
        src_gw = self.getGatewayRouter(src_px)

        # Get current dag to destination
        cDag = self.getCurrentDag(dst_px)

        # Get all paths
        paths = self.getAllPathsInDag(cDag, src_gw, dst_gw, k=0)

        # Compute path capacities
        paths_capacities = {tuple(path): self.getPathCapacities(path) for path in paths}


        #for path, capacities in paths_capacities.iteritems():


        #congProbability

    def getPathCapacities(self, path):
        """Returns current capacity used by the elephants in the path"""
        # Get links in path
        path_links = self.get_links_from_path(path)

        # Get capacities of links
        link_capacities = map(self.getLinkCapacity, path_links)

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


## Past controllers ######################################################################################

class RandomUplinksController(LBController):
    def __init__(self, doBalance=True, k=4):
        super(RandomUplinksController, self).__init__(doBalance=doBalance, k=k, algorithm='random-dags')
        # Keeps track of elephant flows from each pod to every destination
        self.flows_per_pod = {px: {pod: [] for pod in range(0, self.k)} for px in self.fibbing_network_graph.prefixes}

    def allocateFlow(self, flow):
        """"""
        self.chooseRandomUplinks(flow)

    def deallocateFlow(self, flow):
        """"""
        self.resetOSPFDag(flow)

    def areOngoingFlowsInPod(self, dst_prefix, src_pod=None):
        """Returns the list of ongoing elephant flows to
        destination prefix from specific pod.

        If pod==None, all flows to dst_prefix are returned

        :param src_pod:
        :param dst_prefix:
        :return:
        """
        if src_pod != None:
            if self.dc_graph_elep._valid_pod_number(src_pod):
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
            if self.dc_graph_elep._valid_pod_number(src_pod):
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
        src_pod = self.dc_graph_elep.get_router_pod(routerid=src_gw)

        # Check if already ongoing flows
        if not self.areOngoingFlowsInPod(dst_prefix=dst_prefix, src_pod=src_pod):
            log.debug("There are no ongoing flows to {0} from pod {1}".format(dst_prefix, src_pod))

            # Retrieve current DAG for destination prefix
            current_dag = self.current_elephant_dags[dst_prefix]['dag']

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
        src_pod = self.dc_graph_elep.get_router_pod(routerid=src_gw)

        # Remove flow from flows_per_pod
        self.flows_per_pod[dst_prefix][src_pod].remove(flow)

        # Check if already ongoing flows
        if not self.areOngoingFlowsInPod(dst_prefix=dst_prefix, src_pod=src_pod):
            # Retrieve current DAG for destination prefix
            current_dag = self.current_elephant_dags[dst_prefix]['dag']

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
        self.flows_per_pod = {px: {pod: [] for pod in range(0, self.k)} for px in self.fibbing_network_graph.prefixes}

        log.debug("Time to perform the reset to the load balancer: {0}s".format(time.time() - reset_time))

class CoreChooserController(LBController):
    def __init__(self, doBalance=True, k=4, threshold=0.9, algorithm=None):
        super(CoreChooserController, self).__init__(doBalance=doBalance, k=k, algorithm=algorithm)

        # Create structure where we store the ongoing elephant  flows in the graph
        self.elephants_in_paths = self._createElephantInPathsDict()

        # We consider congested a link more than threshold % of its capacity
        self.capacity_threshold = threshold

        # Store all paths --for performance reasons
        self.edge_to_core_paths = self._generateEdgeToCorePaths()
        self.core_to_edge_paths = self._generateCoreToEdgePaths()

        # Keeps track to which core is each flow directed to
        self.flow_to_core = self._generateFlowToCoreDict()

    def _generateFlowToCoreDict(self):
        flow_to_core = {}
        for c in self.dc_graph_elep.core_routers_iter():
            flow_to_core[c] = {}
            for p in self.fibbing_network_graph.prefixes:
                flow_to_core[c][p] = []
        return flow_to_core

    def _generateEdgeToCorePaths(self):
        d = {}
        for edge in self.dc_graph_elep.edge_routers_iter():
            d[edge] = {}
            for core in self.dc_graph_elep.core_routers_iter():
                d[edge][core] = self._getPathBetweenNodes(edge, core)
        return d

    def _generateCoreToEdgePaths(self):
        d = {}
        for core in self.dc_graph_elep.core_routers_iter():
            d[core] = {}
            for edge in self.dc_graph_elep.edge_routers_iter():
                d[core][edge] = self._getPathBetweenNodes(core, edge)
        return d

    def _createElephantInPathsDict(self):
        elephant_in_paths = self.dc_graph_elep.copy()
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
        src_pod = self.dc_graph_elep.get_router_pod(srcgw)
        return src_pod

    def getDstPodFromFlow(self, flow):
        """Returns the pod of the source address for the
        givn flow"""
        dstpx = self.getSourcePrefixFromFlow(flow)
        dstgw = self.getGatewayRouter(dstpx)
        dst_pod = self.dc_graph_elep.get_router_pod(dstgw)
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

    def getPathFromCoreToEdge(self, core, edge):
        """Analogous to getPathFromEdgeToCore"""
        return self.core_to_edge_paths[core][edge]

    def _getPathBetweenNodes(self, node1, node2):
        """Used to construct the edge_to_core data structure
        """
        return nx.dijkstra_path(self.dc_graph_elep, node1, node2)

    def addFlowToPath(self, flow, path):
        edges = self.getEdgesFromPath(path)
        core = path[2]

        with self.mice_caps_lock:
            for (u, v) in edges:
                self.elephants_in_paths[u][v]['flows'].append(flow)
                self.elephants_in_paths[u][v]['capacity'] -= flow['size']
                self.mice_caps_graph[u][v]['elephants_capacity'] += flow['size']

        # Add it to flow_to_core datastructure too
        dst_prefix = self.getDestinationPrefixFromFlow(flow)
        if not self.flow_to_core[core].has_key(dst_prefix):
            self.flow_to_core[core][dst_prefix] = [flow]
        else:
            self.flow_to_core[core][dst_prefix].append(flow)

    def removeFlowFromPath(self, flow, path):
        edges = self.getEdgesFromPath(path)
        core = path[2]
        for (u, v) in edges:
            if flow in self.elephants_in_paths[u][v]['flows']:
                self.elephants_in_paths[u][v]['flows'].remove(flow)
            self.elephants_in_paths[u][v]['capacity'] += flow['size']

        # Remove it from flow_to_core
        dst_prefix = self.getDestinationPrefixFromFlow(flow)
        if flow in self.flow_to_core[core][dst_prefix]:
            self.flow_to_core[core][dst_prefix].remove(flow)

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
        return min(self.getPathCapacities(path))

    def getPathCapacities(self, path):
        path_edges = self.getEdgesFromPath(path)
        return [self.elephants_in_paths[u][v]['capacity'] for (u, v) in path_edges]

    def getAvailableCorePaths(self, src_gw, flow):
        """
        Returns the list of available core router together with
        their path from source gw.

        It checks if flow fits in path and also
        :param src_gw:
        :param flow:
        :return:
        """
        core_paths = []
        dst_px = self.getDestinationPrefixFromFlow(flow)
        dst_gw = self.getGatewayRouter(dst_px)
        for core in self.dc_graph_elep.core_routers():
            # Get both parts of the path
            pathSrcToCore = self.getPathFromEdgeToCore(src_gw, core)
            pathCoreToDst = self.getPathFromCoreToEdge(core, dst_gw)

            # Join complete path
            completePath = pathSrcToCore + pathCoreToDst[1:]

            # Check if it fits (in all path)
            if self.flowFitsInPath(flow, completePath):
                if not self.collidesWithPreviousFlows(src_gw, core, flow):
                    capacities = self.getPathCapacities(completePath)
                    core_paths.append({'core': core, 'path': completePath, 'capacities': capacities})

        # No available core was found... so rank them!
        if core_paths == []:
            # Take one at random --for the moment
            #TODO: think what to do with this
            log.error("No available core paths found!! Returning the non-colliding paths!")
            core_paths = []
            for core in self.dc_graph_elep.core_routers():
                # Get both parts of the path
                pathSrcToCore = self.getPathFromEdgeToCore(src_gw, core)
                pathCoreToDst = self.getPathFromCoreToEdge(core, dst_gw)

                # Join complete path
                completePath = pathSrcToCore + pathCoreToDst[1:]
                if not self.collidesWithPreviousFlows(src_gw, core, flow):
                    capacities = self.getPathCapacities(completePath)
                    core_paths.append({'core': core, 'path': completePath, 'capacities': capacities})

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
        ongoing_flows_same_pod = [flow for flow in ongoing_flows if self.dc_graph_elep.get_router_pod(self.getSourceGatewayFromFlow(flow)) == src_pod]

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

        src_pod = self.dc_graph_elep.get_router_pod(src_gw)

        # Get all flows from that pod going to dst_prefix
        all_flows_same_pod = [self.flow_to_core[c][dst_prefix] for c in self.dc_graph_elep.core_routers_iter()]
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
            log.error("Flow is not assigned to any core - that's weird")
            pass

    @abc.abstractmethod
    def chooseCore(self, available_cores, flow):
        """"""

    @time_func
    def allocateFlow(self, flow):
        # Check for which cores I can send (must remove those
        # ones that would collide with already ongoing flows

        src_gw = self.getGatewayRouter(flow['src'])
        src_gw_name = self.dc_graph_elep.get_router_name(src_gw)
        dst_prefix = self.getMatchingPrefix(flow['dst'])



        # Convert it to alias prefix
        #convert_to_elephant_ip()

        # Get ongonig flows to dst_prefix from the same gateway
        ongoingFlowsSameGateway = self.getongoingFlowsFromGateway(src_gw, dst_prefix)
        if ongoingFlowsSameGateway:
            src_gw_name = self.dc_graph_elep.get_router_name(src_gw)
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
            ac = [self.dc_graph_elep.get_router_name(c['core']) for c in available_cores]
            log.info("Available cores: {0}".format(ac))

            # Choose core
            chosen = self.chooseCore(available_cores, flow)

            # Extract data
            chosen_core = chosen['core']
            chosen_core_name = self.dc_graph_elep.get_router_name(chosen_core)
            chosen_path = chosen['path']
            chosen_overflow = chosen['overflow']

            # Log a bit
            log.info("{0} was chosen with a final overflow of {1}".format(chosen_core_name, self.base.setSizeToStr(chosen_overflow)))
            log.info("Path: {0}".format(self._printPath(chosen_path)))

            # Retrieve current DAG for destination prefix
            if dst_prefix in self.current_elephant_dags.keys():
                current_dag = self.current_elephant_dags[dst_prefix]['dag']
            else:
                log.error("dst_prefix not in current_dats")
                import ipdb; ipdb.set_trace()

            # Apply path
            current_dag.apply_path_to_core(src_gw, chosen_core)

            # Apply DAG
            self.sbmanager.add_dag_requirement(prefix=dst_prefix, dag=current_dag)
            src_gw_name = self.dc_graph_elep.get_router_name(src_gw)
            chosen_core_name = self.dc_graph_elep.get_router_name(chosen_core)
            log.info("A new path was forced from {0} -> {1} for prefix {2}".format(src_gw_name, chosen_core_name, dst_prefix))

        # Append flow to state variables
        self.addFlowToPath(flow, chosen_path)

    @time_func
    def deallocateFlow(self, flow):
        import ipdb; ipdb.set_trace()

        # Get prefix from flow
        (src_prefix, dst_prefix) = self.getPrefixesFromFlow(flow)

        # Get gateway router and pod
        src_gw = self.getGatewayRouter(src_prefix)
        dst_gw = self.getGatewayRouter(dst_prefix)
        src_pod = self.dc_graph_elep.get_router_pod(src_gw)

        # Get to which core the flow was directed to
        core = self.getCoreFromFlow(flow)

        # Compute its current path
        current_edge_to_core = self.getPathFromEdgeToCore(src_gw, core)
        current_core_to_edge = self.getPathFromCoreToEdge(core, dst_gw)
        current_path = current_edge_to_core + current_core_to_edge[1:]

        # Remove it from the path!!!!!
        self.removeFlowFromPath(flow, current_path)

        # Get ongoing flows from the same pod to the dst_prefix
        ongoing_flows_same_pod = self.getOngoingFlowsFromPod(src_pod, dst_prefix)

        if not ongoing_flows_same_pod:
            # Log a bit
            log.debug("No ongoing flows from the same pod {0} we found to prefix {1}".format(src_pod, dst_prefix))

            # Retrieve current DAG for destination prefix
            current_dag = self.current_elephant_dags[dst_prefix]['dag']

            # Restore the DAG
            current_dag.set_ecmp_uplinks_from_source(src_gw, current_edge_to_core, all_layers=True)

            # Apply DAG
            self.sbmanager.add_dag_requirement(prefix=dst_prefix, dag=current_dag)

            # Log a bit
            src_gw_name = self.dc_graph_elep.get_router_name(src_gw)
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
                current_dag = self.current_elephant_dags[dst_prefix]['dag']

                # Restore the DAG
                current_dag.set_ecmp_uplinks_from_source(src_gw, current_edge_to_core, all_layers=False)

                # Apply DAG
                self.sbmanager.add_dag_requirement(prefix=dst_prefix, dag=current_dag)

                # Log a bit
                src_gw_name = self.dc_graph_elep.get_router_name(src_gw)
                log.info("A new DAG was forced restoring PART OF the ECMP DAG from source {0} -> {1}".format(src_gw_name, dst_prefix))

            # There are colliding flows from the same source edge router
            else:
                # Log a bit
                log.debug("There are colliding flows with same edge gateway router {0}. We CAN NOT restore original DAG yet".format(src_gw))

    def reset(self):
        # Reset parent class first
        reset_time = super(CoreChooserController, self).reset()

        # Create structure where we store the ongoing elephant  flows in the graph
        self.elephants_in_paths = self._createElephantInPathsDict()

        # Store all paths --for performance reasons
        self.edge_to_core_paths = self._generateEdgeToCorePaths()
        self.core_to_edge_paths = self._generateCoreToEdgePaths()

        # Keeps track to which core is each flow directed to
        self.flow_to_core = self._generateFlowToCoreDict()

        log.debug("Time to perform the reset to the load balancer: {0}s".format(time.time() - reset_time))

class BestRankedCoreChooser(CoreChooserController):
    def __init__(self, doBalance=True, k=4, threshold=0.9):
        super(BestRankedCoreChooser, self).__init__(doBalance, k, threshold, algorithm="best-ranked-core")

    def chooseCore(self, available_cores, flow):
        log.info("Choosing available core: BEST RANKED CORE")

        virtual_capacities = []

        # Substract flow size to capacities in edges of each available core path
        for acore in available_cores:
            virtual_capacities.append({'core': acore['core'], 'path': acore['path'], 'vcaps': map(lambda x: x - flow['size'], acore['capacities'])})

        capacities_overflow = []
        for vcore in virtual_capacities:
            vcaps = vcore['vcaps']
            overflow = 0
            for vc in vcaps:
                if vc < 0:
                    overflow += vc
            capacities_overflow.append({'core': vcore['core'], 'path': vcore['path'], 'overflow': overflow})

        # Sort available cores from more to less available capacity
        sorted_available_cores = sorted(capacities_overflow, key=lambda x: x['overflow'])

        # From the ones with the same maximum available capacity, take one at random
        min_overflow = sorted_available_cores[0]['overflow']
        cores_max_cap = [c for c in sorted_available_cores if c['overflow'] == min_overflow]
        random.shuffle(cores_max_cap)
        chosen = cores_max_cap[0]
        #chosen = sorted_available_cores[0]

        return chosen

class RandomCoreChooser(CoreChooserController):
    def __init__(self, doBalance=True, k=4, threshold=0.9):
        super(RandomCoreChooser, self).__init__(doBalance, k, threshold, algorithm="random-core")

    def chooseCore(self, available_cores, flow):
        log.info("Choosing available core: AT RANDOM")

        # Shuffle list of av.cores
        random.shuffle(available_cores)

        #Take the first one
        chosen = available_cores[0]

        return chosen

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
                        help='Choose loadbalancing strategy: ecmp|random',
                        default = 'ecmp')

    parser.add_argument('-k', '--k', help='Fat-Tree parameter', type=int, default=4)

    parser.add_argument('-t', '--test', help='Test controller', action="store_true", default=False)

    args = parser.parse_args()

    log.setLevel(logging.DEBUG)
    log.info("Starting Controller - k = {0} , algorithm = {1}".format(args.k, args.algorithm))

    if args.test == True:
        lb = TestController()

    elif args.algorithm == 'ecmp':
        lb = ECMPController(doBalance = args.doBalance, k=args.k)

    elif args.algorithm == 'random-dags':
        lb = RandomUplinksController(doBalance=args.doBalance, k=args.k)

    elif args.algorithm == 'random-core':
        lb = RandomCoreChooser(doBalance=args.doBalance, k=args.k)

    elif args.algorithm == 'best-ranked-core':
        lb = BestRankedCoreChooser(doBalance=args.doBalance, k=args.k)

    # Run the controller
    lb.run()


    #import ipdb; ipdb.set_trace()
