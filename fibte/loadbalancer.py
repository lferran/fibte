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
import abc
import Queue
import numpy as np
import itertools as it

from fibbingnode.algorithms.southbound_interface import SouthboundManager#, DstSpecificSouthboundManager
from fibbingnode import CFG as CFG_fib

from fibte.misc.unixSockets import UnixServerTCP, UnixServer, UnixClient, UnixClientTCP
from fibte.trafficgen.flow import Base
from fibte.misc.dc_graph import DCGraph
from fibte.loadbalancer.mice_estimator import MiceEstimatorThread
from fibte.misc.topology_graph import TopologyGraph
from fibte.monitoring.getLoads import GetLoads
from fibte import tmp_files, db_topo, LINK_BANDWIDTH, UDS_server_name, UDS_server_traceroute, C1_cfg, getLoads_path
from fibte.misc.flowEstimation import EstimateDemands, EstimateDemandError

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

# class MyGraphProvider2(DstSpecificSouthboundManager):
#     """Uses the optimized version of the Southbound Manager"""
#     def __init__(self):
#         super(MyGraphProvider2, self).__init__()
#
#     def received_initial_graph(self):
#         super(MyGraphProvider2, self).received_initial_graph()
#         HAS_INITIAL_GRAPH.set()

class LBController(object):
    def __init__(self, doBalance = True, k=4, algorithm=None, load_variables=True):
        super(LBController, self).__init__()

        # Set fat-tree parameter
        self.k = k

        # Either we load balance or not
        self.doBalance = doBalance

        # Loadbalancing strategy/algorithm
        self.algorithm = algorithm

        # Makes the mice DAG shifter part to be active
        self.mice_dag_shifter = False

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
        #self.sbmanager = MyGraphProvider2()
        t = threading.Thread(target=self.sbmanager.run, name="Southbound Manager thread").start()

        # Blocks until initial graph received from SouthBound Manager
        HAS_INITIAL_GRAPH.wait()
        log.info("Initial graph received from SouthBound Controller")

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
        self.edges_to_flows = {(a, b): {} for (a, b) in self.dc_graph_elep.edges()}
        self.add_del_flow_lock = threading.Lock() # Access lock (!)

        # Start getLoads thread that reads from counters
        self._startGetLoadsThread()

        # Object useful to make some unit conversions
        self.base = Base()

        # Start all the traceroute tools
        self._startTracerouteService()

        # Start mice thread
        self.createMiceEstimatorThread()

        # Start object that estimates flow demands
        self.flowDemands = EstimateDemands()

        # This is for debugging purposes only --should be removed
        if load_variables == True and self.k == 4:
            self.r0e0 = self.topology.getRouterId('r_0_e0')
            self.r0e1 = self.topology.getRouterId('r_0_e1')

            self.r1e0 = self.topology.getRouterId('r_1_e0')
            self.r1e1 = self.topology.getRouterId('r_1_e1')

            self.r2e0 = self.topology.getRouterId('r_2_e0')
            self.r2e1 = self.topology.getRouterId('r_2_e1')

            self.r3e0 = self.topology.getRouterId('r_3_e0')
            self.r3e1 = self.topology.getRouterId('r_3_e1')

            self.r0a0 = self.topology.getRouterId('r_0_a0')
            self.r0a1 = self.topology.getRouterId('r_0_a1')

            self.r1a0 = self.topology.getRouterId('r_1_a0')
            self.r1a1 = self.topology.getRouterId('r_1_a1')

            self.r2a0 = self.topology.getRouterId('r_2_a0')
            self.r2a1 = self.topology.getRouterId('r_2_a1')

            self.r3a0 = self.topology.getRouterId('r_3_a0')
            self.r3a1 = self.topology.getRouterId('r_3_a1')

            self.rc0 = self.topology.getRouterId('r_c0')
            self.rc1 = self.topology.getRouterId('r_c1')
            self.rc2 = self.topology.getRouterId('r_c2')
            self.rc3 = self.topology.getRouterId('r_c3')

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

    @abc.abstractmethod
    def _getAlgorithmName(self):
        """Returns a string with the name of the algorithm!"""

    def _startGetLoadsThread(self):
        in_out_name = self._getAlgorithmName() # Get the name of the GetLoads result file
        self.gl = GetLoads(k=self.k, time_interval=1, lb_algorithm=in_out_name)
        self.gl_thread = Thread(target=self.gl.run, name="GetLoads thread")
        self.gl_thread.setDaemon(True)
        self.gl_thread.start()

    def _startTracerouteService(self):
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

        # Data structure where we store the flows waiting for traceroute answer
        self.waiting_tr_lock = threading.Lock()
        self.waiting_traceroutes = []

    def traceroute_sendInitialData(self):
        """
        :return:
        """
        log.info("Sending initial data to traceroute servers")

        hosts_couldnt_inform = []
        for host in self.topology.getHosts().keys():
            could_inform = self.traceroute_sendInitialDataToHost(host)
            if not could_inform:
                hosts_couldnt_inform.append(host)
        if hosts_couldnt_inform:
            log.error("The following flowServers could not be contacted: {0}".format(hosts_couldnt_inform))

    def traceroute_sendInitialDataToHost(self, host):
        could_inform = True

        # Get hosts in the same pod
        gw_name = self.topology.hostToGatewayMapping['hostToGateway'][host]
        gw_pod = self.topology.getRouterPod(gw_name)
        own_pod_ers = [r for r in self.topology.getRouters().keys() if
                       self.topology.isEdgeRouter(r) and self.topology.getRouterPod(r) == gw_pod]
        own_pod_hosts = [self.topology.getHostIp(h) for r in own_pod_ers for h in
                         self.topology.hostToGatewayMapping['gatewayToHosts'][r]]

        # Send list to traceroute client
        try:
            # Send command
            command = {'type': 'own_pod_hosts', 'data': own_pod_hosts}
            self.traceroute_client.send(json.dumps(command), host)
        except Exception as e:
            could_inform = False
        finally:
            return could_inform

    def isTracerouteWaitingForFlow(self, flow):
        """Check if traceroute is still waiting for flow's result"""
        fkey = self.flowToKey(flow)
        with self.waiting_tr_lock:
            return fkey in self.waiting_traceroutes

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
                fkey = self.flowToKey(flow)

                if not self.isTracerouteWaitingForFlow(flow):
                    # Omit the data
                    log.warning("discarded data!")
                    continue

                src_n = self.topology.getHostName(flow['src'])
                dst_n = self.topology.getHostName(flow['dst'])
                sport = flow['sport']
                dport = flow['dport']

                # Extracts the router traceroute data
                route = traceroute_data['route']

                if route:
                    # Traceroute fast is used
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

                    # Remove it from the waiting list
                    with self.waiting_tr_lock:
                        self.waiting_traceroutes.remove(fkey)

                    # Add flow -> path allocation
                    if not self.isOngoingFlow(flow):
                        # Add flow to path
                        self.addFlowToPath(flow, path)

                    else:
                        # Update its path
                        self.updateFlowPath(flow, path)
                else:
                    # Log a bit
                    log.warning("Path couldn't be found yet for Flow({0}:({1}) -> {2}:({3})). Traceroute again!".format(src_n, sport, dst_n, dport))

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

    def _sendMainThreadToSleep(self, seconds):
        """Makes the main thread jump to the sleep mode
        **ONLY FOR DEBUGGING PURPOSES!
        """
        self.q_server.put(json.dumps({'type': 'sleep', 'data': seconds}))

    def get_router_name(self, addr):
        """Addr can either the router id, the interface ip or the private ip"""
        for fun in [self.topology.getRouterName, self.topology.getRouterFromPrivateIp, self.topology.getRouterFromInterfaceIp]:
            try:
                return fun(addr)
            except KeyError:
                continue

        log.error("EEEEEP ERROR FORT")
        self._sendMainThreadToSleep(4000)
        import ipdb; ipdb.set_trace()

        for fun in [self.topology.getRouterName, self.topology.getRouterFromPrivateIp, self.topology.getRouterFromInterfaceIp]:
            try:
                return fun(addr)
            except KeyError:
                continue

        raise ValueError("{0} is neither a private ip, router id or interface ip".format(addr))

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

    def createMiceEstimatorThread(self):
        # Here we store the estimated mice levels
        self.hosts_notified = []
        self.total_hosts = ((self.k/2)**2)*self.k
        self.mice_caps_graph = self._createElephantsCapsGraph()
        self.mice_caps_lock = threading.Lock()
        self.mice_orders_queue= Queue.Queue(0)
        self.flowpath_queue= Queue.Queue(0)

        # Create the mice estimator thread
        self.miceEstimatorThread = MiceEstimatorThread(active=self.mice_dag_shifter,
                                                       sbmanager=self.sbmanager,
                                                       orders_queue=self.mice_orders_queue,
                                                       flowpath_queue=self.flowpath_queue,
                                                       capacities_graph = self.mice_caps_graph,
                                                       dags = self.current_mice_dags,
                                                       q_server=self.q_server)

    def _createElephantsCapsGraph(self):
        graph = DCGraph(k=self.k, prefix_type='secondary')
        for (u, v, data) in graph.edges_iter(data=True):
            data['elephants_capacity'] = 0
        return graph

    def _printPath(self, path):
        return [self.dc_graph_elep.get_router_name(n) for n in path]

    def _do_logging_stuff(self):
        # Config logging to dedicated file for this thread
        self.logfile = '{0}loadbalancer_{1}.log'.format(tmp_files, self.algorithm)
        handler = logging.FileHandler(filename=self.logfile)
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

    def _waitOSPFPropagation(self):
        # Wait for fibbing to apply the dag and the new requirements to propagate
        WAIT_PROPAGATION_TIME_MS = 250
        time.sleep(WAIT_PROPAGATION_TIME_MS / 1000.0)

    @time_func
    def saveCurrentDag(self, prefix, dag, fib=True):
        """"""
        if ipalias.is_secondary_ip_prefix(prefix):
            self.current_mice_dags[prefix]['dag'] = dag
        else:
            self.current_elephant_dags[prefix]['dag'] = dag

        if fib:
            # Add new requirement for Fibbing
            self.sbmanager.add_dag_requirement(prefix, dag)

            # Wait a bit for OSPF propagation
            self._waitOSPFPropagation()

    @time_func
    def saveCurrentDags_from(self, dag_requirements, fib=True):
        # Update current structures
        for (dst, dag) in dag_requirements.iteritems():
            if ipalias.is_secondary_ip_prefix(dst):
                self.current_mice_dags[dst]['dag'] = dag
            else:
                self.current_elephant_dags[dst]['dag'] = dag

        if fib:
            # Force new dags
            self.sbmanager.add_dag_requirements_from(dag_requirements)

            # Wait a bit
            self._waitOSPFPropagation()

    def resetToInitialDag(self, prefix, fib=True, src_pod=None):
        """
        Reset DAG of prefix to its initial state! Limit it to source pod if specified.
        """
        if not src_pod:
            idag = self.initial_elep_dags[prefix]['dag'].copy()
            self.saveCurrentDag(prefix, idag, fib=fib)

        else:
            cdag = self.current_elephant_dags[prefix]['dag']
            cdag.set_ecmp_uplinks_from_pod(src_pod=src_pod)
            self.saveCurrentDag(prefix, cdag, fib=fib)

    def getMatchingPrefix(self, hostip):
        """
        Given a ip address of a host in the mininet network, returns
        the longest prefix currently being advertised by the OSPF routers.

        :param hostip: string representing a host's ip of an IPv4Address object
                             address. E.g: '192.168.233.254/30'
        Returns: an ipaddress.IPv4Network object
        """
        try:
            if not isinstance(hostip, ip.IPv4Address) and (isinstance(hostip, str) or isinstance(hostip, unicode)):
                # Convert it to ipv4address type
                hostip = ip.ip_address(hostip)

            longest_match = (None, 0)
            for prefix in self.ospf_prefixes:
                prefix_len = prefix.prefixlen
                if hostip in prefix and prefix_len > longest_match[1]:
                    longest_match = (prefix, prefix_len)
            return longest_match[0].compressed
        except:
            import ipdb; ipdb.set_trace()

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

        for prefix in self.fibbing_network_graph.prefixes:
            if not ipalias.is_secondary_ip_prefix(prefix):
                # Get prefix gateway router
                gatewayRouter = self._getGatewayRouter(prefix)

                # Compute initial dag (default OSPF)
                dc_dag = self.dc_graph_elep.get_default_ospf_dag(prefix)

                # Add dag
                dags[prefix] = {'gateway': gatewayRouter, 'dag': dc_dag.copy()}

        return dags

    def _createInitialMiceDags(self):
        """
        Populates the self.current_mice_dags dictionary
        for each existing prefix in the network
        """
        # Result is stored here
        log.info("Creating initial DAGs for mice prefixes in the network")
        dags = {}

        for prefix in self.fibbing_network_graph.prefixes:
            if not ipalias.is_secondary_ip_prefix(prefix):
                secondary_prefix = ipalias.get_secondary_ip_prefix(prefix)
                gatewayRouter = self.getGatewayRouter(prefix)
                dc_dag = self.dc_graph_mice.get_default_ospf_dag(secondary_prefix)

                # Add dag
                if secondary_prefix not in dags:
                    dags[secondary_prefix] = {'gateway': gatewayRouter, 'dag': dc_dag.copy()}

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
        # Set all dags to original ospf dag
        self.current_elephant_dags = copy.deepcopy(self.initial_elep_dags)
        self.current_mice_dags = copy.deepcopy(self.initial_mice_dags)

        # Remove all attraction points and lsas
        self.sbmanager.remove_all_dag_requirements()
        time.sleep(2)

        # Terminate pevious mice estimator thread
        self.miceEstimatorThread.orders_queue.put({'type': 'terminate'})
        try:
            self.miceEstimatorThread.join()
        except:
            pass

        # Restart Mice Estimator Thread
        self.createMiceEstimatorThread()

        # Empty flow to path and edges data structures
        with self.add_del_flow_lock:
            self.flows_to_paths = {}
            self.edges_to_flows = {(a, b): {} for (a, b) in self.dc_graph_elep.edges()}

        # Reset elephants capacity
        with self.mice_caps_lock:
            for (u, v, data) in self.mice_caps_graph.edges_iter(data=True):
                data['elephants_capacity'] = 0.0

        # Reset flow demands
        self.flowDemands = EstimateDemands()

        # Flush all events in the queue
        with self.q_server.mutex:
            self.q_server.queue.clear()

        log.info('Loadbalancer was reset successfully')

    def getLinkCapacity(self, link, exclude=[]):
        """Returns the difference between link bandwidth and
        the sum of all flows going through the link"""
        return LINK_BANDWIDTH - self.getLinkLoad(link, exclude=exclude)

    def getLinkLoad(self, link, exclude=[]):
        """
        Returns the current load of the link, assuming flows
        are limited by sender and receiver's NIC

        Counts only the sizes of the flows that are not present
        in the exclude list.
        """
        if link in self.edges_to_flows.keys():
            if self.edges_to_flows[link]:
                return sum([self.flowDemands.getDemand(fkey) for fkey in self.edges_to_flows[link].iterkeys() if fkey not in exclude])
            else:
                return 0.0
        else:
            raise ValueError("Link doesn't exist {0}".format(self.dc_graph_elep.print_stuff(link)))

    def isOngoingFlow(self, flow):
        """Returns True if flow is in the data structures,
         meaning that the flow is still ongoing"""
        fkey = self.flowToKey(flow)

        with self.add_del_flow_lock:
            return fkey in self.flows_to_paths.keys()

    def logFlowToPath(self, action, flow, path, old_path=None):
        """Log action"""
        src_n = self.topology.getHostName(flow['src'])
        dst_n = self.topology.getHostName(flow['dst'])
        size_n = self.base.setSizeToStr(flow['size'])
        path_n = [self.topology.getRouterName(r) for r in path]
        sport = flow['sport']
        dport = flow['dport']
        path_str = ', '.join(path_n)

        if action == 'add':
            log.info("Flow({0}:({1}) -> {2}:({3}) #{4}) added to path ({5})".format(src_n, sport, dst_n, dport, size_n, path_str))

        elif action == 'delete':
            log.info("Flow({0}:({1}) -> {2}:({3}) #{4}) deleted from path ({5})".format(src_n, sport, dst_n, dport, size_n, path_str))

        elif action == 'update' and old_path:
            old_path_n = [self.topology.getRouterName(r) for r in old_path]
            old_path_str = ', '.join(old_path_n)
            log.info("Flow({0}:({1}) -> {2}:({3}) #{4}) changed path from ({5}) to ({6})".format(src_n, sport, dst_n, dport, size_n, old_path_str, path_str))
        else:
            raise ValueError("Check the arguments...")

    def setFlowToUpdateStatus(self, flow):
        """"""
        # Get key from flow
        fkey = self.flowToKey(flow)

        # Get lock
        with self.add_del_flow_lock:
            # Update flows_to_paths
            if fkey in self.flows_to_paths.keys():
                self.flows_to_paths[fkey]['to_update'] = True
                path = self.flows_to_paths[fkey]['path']
            else:
                raise ValueError("Weird: flow isn't in the data structure")

            # Upadte links too
            for link in self.get_links_from_path(path):
                if link in self.edges_to_flows.keys():
                    if fkey in self.edges_to_flows[link]:
                        self.edges_to_flows[link][fkey]['to_update'] = True
                    else:
                        raise ValueError("Weird: fkey not in link")
                else:
                    raise ValueError("Weird: flow was already in the data structure")

    def addFlowToPath(self, flow, path, do_log=False):
        """Adds a new flow to path datastructures"""
        # Get key from flow
        fkey = self.flowToKey(flow)

        # Get lock
        with self.add_del_flow_lock:
            # Update flows_to_paths
            if fkey not in self.flows_to_paths.keys():
                self.flows_to_paths[fkey] = {'path': path, 'to_update': False}
            else:
                raise ValueError("Weird: flow was already in the data structure")
            # Upadte links too
            for link in self.get_links_from_path(path):
                if link in self.edges_to_flows.keys():
                    self.edges_to_flows[link][fkey] = {'flow': flow, 'to_update': False}
                else:
                    raise ValueError("Weird: flow was already in the data structure")

        # Send notification to mice LB
        data = (flow, path, 'add')
        self.flowpath_queue.put(data)

        # Log a bit
        if do_log:
            self.logFlowToPath(action='add', flow=flow, path=path)

    def delFlowFromPath(self, flow, do_log=False):
        """Removes an ongoing flow from path"""
        # Get key from flow
        fkey = self.flowToKey(flow)

        with self.add_del_flow_lock:
            # Update data structures
            if fkey in self.flows_to_paths.keys():
                old_path = self.flows_to_paths[fkey]['path']
                self.flows_to_paths.pop(fkey)
            else:
                raise ValueError("Flow wasn't in data structure")

            for link in self.get_links_from_path(old_path):
                if link in self.edges_to_flows.keys():
                    if fkey in self.edges_to_flows[link]:
                        self.edges_to_flows[link].pop(fkey)
                    else:
                        log.error("Flow wasn't in data structure")
                        return
                else:
                    log.error("Weird: link not in the data structure")
                    return

        # Send notificatoin to mice LB
        data = (flow, old_path, 'del')
        self.flowpath_queue.put(data)

        # Log a bit
        if do_log:
            self.logFlowToPath(action='delete', flow=flow, path=old_path)

        return old_path

    def updateFlowPath(self, flow, new_path, do_log=False):
        """Updates path from an already existing flow"""
        if not self.isOngoingFlow(flow):
            raise ValueError("We are updating a flow that is not yet registered!!!")

        # Delete it first
        old_path = self.delFlowFromPath(flow, do_log=False)

        # Add it to the new path
        self.addFlowToPath(flow, new_path, do_log=False)

        # Log a bit
        if do_log:
            self.logFlowToPath(action='update', flow=flow, path=new_path, old_path=old_path)

    def allocateFlow(self, flow):
        """
        Subclass this method
        """
        # Update flow demands
        self.flowDemands.estimateDemands(flow, action='add')

        src = self.topology.getHostName(flow['src'])
        dst = self.topology.getHostName(flow['dst'])
        rate = self.flowDemands.getDemand(flow)
        rate = self.base.setSizeToStr(rate * LINK_BANDWIDTH)
        sport = flow['sport']
        dport = flow['dport']
        proto = flow['proto']
        log.debug("Flow STARTED: {5}Flow({0}:({1}) -> {2}:({3}) | Demand: {4})".format(src, sport, dst, dport, rate, proto))

    def deallocateFlow(self, flow):
        """
        Subclass this method
        """
        # Return if dealocate flow was successful
        successful = True

        # Get final rate of flow
        try:
            rate = self.flowDemands.getDemand(flow)
        except EstimateDemandError:
            log.warning("Flow couldn't be deallocated: it wasn't in flowDemands data structure")
            successful = False
            return successful

        # Remove flow from estimation matrix
        self.flowDemands.estimateDemands(flow, action='del')

        src = self.topology.getHostName(flow['src'])
        dst = self.topology.getHostName(flow['dst'])
        sport = flow['sport']
        dport = flow['dport']
        rate = self.base.setSizeToStr(rate * LINK_BANDWIDTH)
        proto = flow['proto']
        log.debug("Flow FINISHED: {5}Flow({0}:({1}) -> {2}:({3}) | #{4})".format(src, sport, dst, dport, rate, proto))
        return successful

    def tracerouteFlow(self, flow):
        """Starts a traceroute"""
        # Get source hostname from ip
        src_name = self.topology.getHostName(flow['src'])

        # Compute flow key
        fkey = self.flowToKey(flow)
        try:
            # Send instruction to start traceroute to specific flowServer
            command = {'type': 'flow', 'data': flow}
            self.traceroute_client.send(json.dumps(command), src_name)
        except Exception as e:
            log.error("{0} could not be reached. Exception: {1}".format(src_name, e))
        finally:
            with self.waiting_tr_lock:
                if fkey not in self.waiting_traceroutes:
                    self.waiting_traceroutes.append(fkey)

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

        # Finally exit
        os._exit(0)

    def isFlowPathUpdated(self, flow):
        """"""
        fkey = self.flowToKey(flow)

        with self.add_del_flow_lock:
            if fkey not in self.flows_to_paths.keys():
                return False
            elif self.flows_to_paths[fkey]['to_update'] == False:
                return True
            else:
                return False

    def getFlowPath(self, flow):
        # Get key of flow
        fkey = self.flowToKey(flow)
        with self.add_del_flow_lock:
            if fkey in self.flows_to_paths:
                if not self.flows_to_paths[fkey]['to_update']:
                    return self.flows_to_paths[fkey]['path']
                else:
                    return None
            else:
                return None

    def findNewPathsForFlows(self, flows):
        """This function only returns if the path for all flows in the input were found"""
        # Set flows state to be updates
        for f in flows:
            if self.isOngoingFlow(f):
                self.setFlowToUpdateStatus(f)

        # Send traceroutes
        self.tracerouteFlows(flows)

        # Wait for flows to finish
        self.waitForTraceroutes(flows)

    def tracerouteFlows(self, flows):
        """Send traceroutes for flows"""
        for f in flows:
            self.tracerouteFlow(f)

    def waitForTraceroutes(self, ongoing_flows):
        """Waits for traceroutes to have finished"""
        found_all_paths = all([True if self.isFlowPathUpdated(f) else False for f in ongoing_flows])
        i = 0
        while not found_all_paths:
            found_all_paths = all([True if self.isFlowPathUpdated(f) else False for f in ongoing_flows])
            i += 1
        #log.debug("{0} iterations needed".format(i))

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
                    try:
                        # Read from TCP server's queue
                        event = json.loads(self.q_server.get(timeout=0.01))
                    except Queue.Empty:
                        continue

                    if event["type"] == "reset":
                        self.reset()

                    elif event["type"] in ["startingFlow", "stoppingFlow"]:
                        self.handleFlow(event)

                    elif event['type'] == 'sleep':
                        log.info("<SLEEP> event received. Sleeping for {0} seconds".format(event['data']))
                        time.sleep(event['data'])
                        log.warning("WOKE UP from <SLEEP> event!")

                    else:
                        log.error("<Unknown> event: {0}".format(event))

                    # Inform server about task done
                    self.q_server.task_done()

            except KeyboardInterrupt:
                # Exit load balancer
                self.exitGracefully()

class ECMPController(LBController):
    def __init__(self, *args, **kwargs):
        super(ECMPController, self).__init__(algorithm='ecmp', *args, **kwargs)

    def _getAlgorithmName(self):
        return "ecmp_k_{0}".format(self.k)

    def allocateFlow(self, flow):
        """Just checks the default route for this path,
        and uptades a flow->path data structure"""
        super(ECMPController, self).allocateFlow(flow)

        # Check if flow is within same subnetwork
        if self.isFlowToSameNetwork(flow):
            # No need to loadbalance anything
            # For TCP: we need to update the matrix
            return

        # Get default path of flow
        self.findNewPathsForFlows([flow])

    def deallocateFlow(self, flow):
        """Removes flow from flow->path data structure"""
        if super(ECMPController, self).deallocateFlow(flow):

            # Check if flow is within same subnetwork
            if self.isFlowToSameNetwork(flow):
                # No need to loadbalance anything
                # For TCP: we need to update the matrix
                return

            # Remove flow from path
            path = self.delFlowFromPath(flow)


class MiceDAGShifter(LBController):
    def __init__(self, *args, **kwargs):
        super(MiceDAGShifter, self).__init__(algorithm='mice-dag-shifter', *args, **kwargs)

        # Start thread
        self.mice_dag_shifter = True
        self.createMiceEstimatorThread()
        self.miceEstimatorThread.start()

        test = False
        if test:
            # Invent flow
            h00 = self.topology.getHostIp('h_0_0')
            h10 = self.topology.getHostIp('h_1_0')
            h20 = self.topology.getHostIp('h_2_0')
            h30 = self.topology.getHostIp('h_3_0')

            flow1 = {'src': h00, 'sport': 5522, 'dport': 1155, 'dst': h30, 'proto': 'TCP', 'size': LINK_BANDWIDTH * 30,
                    'duration': None, 'rate': LINK_BANDWIDTH}

            flow2 = {'src': h10, 'sport': 5555, 'dport': 5555, 'dst': h20, 'proto': 'TCP', 'size': LINK_BANDWIDTH * 30,
                    'duration': None, 'rate': LINK_BANDWIDTH}

            import ipdb; ipdb.set_trace()

            # Add it to the demands
#            self.flowDemands.estimateDemands(flow1, action='add')
            self.flowDemands.estimateDemands(flow2, action='add')

            # Invent path
            path1 = (self.r0e0, self.r0a0, self.rc0, self.r3a0, self.r3e0)
            path2 = (self.r1e0, self.r1a0, self.rc1, self.r2a0, self.r2e0)
            time.sleep(2)

            # Add it to the path
            self.addFlowToPath(flow1, path1)
            self.addFlowToPath(flow2, path2)

    def reset(self):
        super(MiceDAGShifter, self).reset()

    def _getAlgorithmName(self):
        return "mice-dag-shifter_k_{0}".format(self.k)

    def adaptMiceDags(self, path):
        # Send order to mice thread!
        order = {'type': 'adapt_mice_to_elephants', 'path': path}
        self.mice_orders_queue.put(order)

    def allocateFlow(self, flow):
        """Just checks the default route for this path,
        and uptades a flow->path data structure"""
        super(MiceDAGShifter, self).allocateFlow(flow)

        # Check if flow is within same subnetwork
        if self.isFlowToSameNetwork(flow):
            # No need to loadbalance anything
            # For TCP: we need to update the matrix
            return

        # Traceroute new flow
        self.findNewPathsForFlows([flow])

    def deallocateFlow(self, flow):
        """Removes flow from flow->path data structure"""
        if super(MiceDAGShifter, self).deallocateFlow(flow):

            # Check if flow is within same subnetwork
            if self.isFlowToSameNetwork(flow):
                # No need to loadbalance anything
                # For TCP: we need to update the matrix
                return

class ElephantDAGShifter(LBController):
    def __init__(self, capacity_threshold=1, congProb_threshold=0.0, sample=False, *args, **kwargs):
        # We consider congested a link more than threshold % of its capacity
        self.capacity_threshold = capacity_threshold
        self.congestion_threshold = self.capacity_threshold*LINK_BANDWIDTH
        self.congProb_threshold = congProb_threshold
        self.sample = sample

        # Used to communicate with flowServers at the hosts.
        self.unixClient = UnixClientTCP(tmp_files + "flowServer_{0}")

        # Call init of subclass
        super(ElephantDAGShifter, self).__init__(algorithm='elephant-dag-shifter', *args, **kwargs)

        tests = False
        if tests:
            h33 = self.topology.getHostIp('h_3_3')
            h32 = self.topology.getHostIp('h_3_2')
            h00 = self.topology.getHostIp('h_0_0')
            sport = 2000
            starttime = 10

            flow1 = {'src': h00, 'sport': sport, 'dport': 1111, 'dst': h33, 'proto': 'TCP', 'size': LINK_BANDWIDTH*30,
                    'duration': None, 'rate': LINK_BANDWIDTH}

            import ipdb; ipdb.set_trace()

            self.startFlow(flow1, starttime=2)

    def startFlow(self, flow, starttime=2):
        # Get source ip
        srcip = flow.get('src')
        dstip = flow.get('dst')
        srcname = self.topology.getHostName(srcip)
        dstname = self.topology.getHostName(dstip)

        BUFFER_TIME_S = 2

        # Insert startime
        flow.update({'starttime': starttime})

        # Send flowlist
        if flow.get('proto') == 'UDP':
            flowlist = [flow]
            self.unixClient.send(json.dumps({"type": "flowlist", "data": flowlist}), srcname)

        else:
            starttime = flow.get('start_time')
            rcv_start = max(0, starttime - BUFFER_TIME_S)
            receivelist = [(rcv_start, flow.get('dport'))]

            # Sends flowlist to the sender's server
            self.unixClient.send(json.dumps({"type": "receivelist", "data": receivelist}), dstname)

            flowlist = [flow]
            self.unixClient.send(json.dumps({"type": "flowlist", "data": flowlist}), srcname)

        # Update demands
        self.flowDemands.estimateDemands(flow, action='add')

    def _getAlgorithmName(self):
        """"""
        return "elephant-dag-shifter_k_{3}_cap_{0}_cong_{1}_sample_{2}".format(self.capacity_threshold, self.congProb_threshold, self.sample, self.k)

    @time_func
    def allocateFlow(self, flow):
        super(ElephantDAGShifter, self).allocateFlow(flow)

        # Check if flow is within same subnetwork
        if self.isFlowToSameNetwork(flow):
            # No need to loadbalance anything
            # For TCP: we need to update the matrix
            return

        # Get matching destination
        dst_px = self.getMatchingPrefix(flow['dst'])

        # Get current DAG
        cdag = self.getCurrentDag(dst_px)

        # Get source pod
        src_pod = self.getSourcePodFromFlow(flow)

        # Get the flows originating at same pod towards same destination
        dst_ongoing_flowkeys = self.getOngoingFlowKeysToDst(dst=dst_px, from_pod=src_pod)

        dst_name = self.topology.getHostName(flow['dst'])
        log.info("{0} more flows from pod{2} are currently ongoing towards {1}".format(len(dst_ongoing_flowkeys), dst_name, src_pod))

        # Get dag with spare capacities
        idag = self.getInitialDagWithoutFlows(dst_px=dst_px, ongoing_flow_keys=dst_ongoing_flowkeys)

        # Convert them to flows and add current flow to computations too
        dst_ongoing_flows = [self.keyToFlow(fkey) for fkey in dst_ongoing_flowkeys]
        dst_ongoing_flows.append(flow)

        # Choose new DAG minimizing probability
        if self.sample:
            log.info("Sampling on DAG space...")
            (best_dag, best_assessment) = self.getBestOfDagSamples(src_pod, idag, dst_ongoing_flows)

        else:
            log.info("Iterating ALL DAG space...")
            (best_dag, best_assessment) = self.findBestSourcePodDag(src_pod, idag, dst_ongoing_flows)

        # Extract cost and karma expected values
        dagcost = best_assessment['cost']['mean']
        dagkarma = best_assessment['karma']['mean']

        # Plot it
        src_name = self.topology.getHostName(flow['src'])
        dst_name = self.topology.getHostName(flow['dst'])
        sport = flow['sport']
        dport = flow['dport']
        size = self.base.setSizeToStr(flow['size'])
        img_name = './images/bestSampledDag{0}_{1}_{2}_{3}_{4}___C{5}___K{6}.png'.format(src_name, sport, dst_name, dport, size, dagcost, dagkarma)
        #best_dag.plot(img_name)

        # Log a bit
        #log.info("Best DAG was found with a cost: {0} and karma: {1} (img: {2})".format(dagcost, dagkarma, img_name))
        log.info("Best DAG was found with a cost: {0} and karma: {1}".format(dagcost, dagkarma))

        # Force it with Fibbing
        self.saveCurrentDag(dst_px, best_dag)

        # Find new paths for flows
        self.findNewPathsForFlows(dst_ongoing_flows)

    def deallocateFlow(self, flow):
        successful = super(ElephantDAGShifter, self).deallocateFlow(flow)
        if successful:

            # Get destination
            dst = self.getDestinationPrefixFromFlow(flow)

            # Check if flow is within same subnetwork
            if self.isFlowToSameNetwork(flow):
                # No need to loadbalance anything
                # For TCP: we need to update the matrix
                return

            # Deallocate if from the network
            old_path = self.delFlowFromPath(flow)

    def findBestSourcePodDag(self, src_pod, complete_dag, ongoing_flows):
        """"""
        # Compute edge indexes for which no flows are starting towards dst
        to_exclude = self.getExcludedEdgeIndexes(ongoing_flows)

        # Start the generator of all possible random edge choices
        all_edge_choices = complete_dag.all_random_uplinks_iter(src_pod=src_pod,
                                                                exclude_edge_indexes=to_exclude)

        # Accumulate best result here
        best_assessment = {}
        best_uplinks = []

        # Get egress router
        dst_gw = self.getGatewayRouter(self.getDestinationPrefixFromFlow(ongoing_flows[0]))

        # Create a new dag of the complete dag with these edges only
        ndag = complete_dag.copy()

        for i, edges_choice in enumerate(all_edge_choices):
            # Modify corresponding uplinks
            ndag.modify_uplinks_from(edges_choice)

            # Compute dag assessment
            dagAssessment = self.computeExpectedFlowsCost(complete_dag, ndag, dst_gw, ongoing_flows)

            # Add also the number of edges
            dagAssessment.update({'edges': len(edges_choice)})

            #log.info("DAG {0} -> cost {1}".format(i, eFlowCost))
            cost = dagAssessment['cost']['mean']
            karma = dagAssessment['karma']['mean']

            #ndag.plot("./images/dag_{0}_C{1}_K{2}.png".format(i, float(cost), float(karma)))

            # If we find one with cost == 0, return it directly!
            #if dagAssessment['cost']['mean'] == 0.0:
            #    return (ndag, dagAssessment)

            if self.isBetterAssessment(dagAssessment, best_assessment):
                # Update max
                best_uplinks = edges_choice
                best_assessment = dagAssessment

        # Modify finally the dag with the best uplinks found
        ndag.modify_uplinks_from(best_uplinks)

        # Return the best one!
        return (ndag, best_assessment)

    def getBestOfDagSamples(self, src_pod, complete_dag, ongoing_flows):
        """"""
        # Compute edge indexes for which no flows are starting towards dst
        to_exclude = self.getExcludedEdgeIndexes(ongoing_flows)

        # Accumulate best result here
        best_assessment = {}
        best_uplinks = []

        # Get egress router
        dst_gw = self.getGatewayRouter(self.getDestinationPrefixFromFlow(ongoing_flows[0]))

        # Create a new dag of the complete dag with these edges only
        ndag = complete_dag.copy()

        # Compute how many DAG samples to perform
        if len(to_exclude) == 1:
            # Out of 15
            n_samples = 15
        else:
            # Out of 225
            n_samples = 100

        log.info("Taking {0} DAG samples...".format(n_samples))

        # Compute n samples at random
        for i in range(n_samples):

            # Generate random uplinks
            ruplinks = ndag.get_random_uplinks(src_pod, exclude_edge_indexes=to_exclude)

            # Modify dag
            ndag.modify_uplinks_from(ruplinks)

            # Compute dag assessment
            dagAssessment = self.computeExpectedFlowsCost(complete_dag, ndag, dst_gw, ongoing_flows)
            dagAssessment.update({'edges': len(ruplinks)})

            #log.info("DAG {0} -> cost {1}".format(i, eFlowCost))
            cost = dagAssessment['cost']['mean']
            karma = dagAssessment['karma']['mean']
            #ndag.plot("./images/dag_{0}_C{1}_K{2}.png".format(i, float(cost), float(karma)))
            # If we find one with cost == 0, return it directly!
            #if dagAssessment['cost']['mean'] == 0.0:
            #    return (ndag, dagAssessment)

            if self.isBetterAssessment(dagAssessment, best_assessment):
                # Update max
                best_uplinks = ruplinks
                best_assessment = dagAssessment

        # Modify finally the dag with the best uplinks found
        ndag.modify_uplinks_from(best_uplinks)

        # Return the best one!
        return (ndag, best_assessment)

    @staticmethod
    def isBetterAssessment(a, b):
        """
        Returns True if assessment a is better than assessment b
        """
        if not b:
            return True

        if not a:
            return False

        acost = a['cost']['mean']
        bcost = b['cost']['mean']
        akarma = a['karma']['mean']
        bkarma = b['karma']['mean']

        # If different cost
        if acost != bcost:
            return acost < bcost

        # Check karma if equal
        else:
            # If different karma
            if akarma != bkarma:
                return akarma > bkarma
            else:
                acost_std = a['cost']['std']
                bcost_std = b['cost']['std']
                akarma_std = a['karma']['std']
                bkarma_std = b['karma']['std']

                # Check for the least std observed
                if acost_std != bcost_std:
                    return acost_std < bcost_std
                elif akarma_std != bkarma_std:
                    return akarma_std < bkarma_std
                else:
                    a_nedges = a['edges']
                    b_nedges = b['edges']
                    if a_nedges != b_nedges:
                        return a_nedges < b_nedges

                    else:
                        #log.warning("ALL equal! ...")
                        return True

    def getExcludedEdgeIndexes(self, ongoing_flows):
        """
        Compute edge indexes for which no flows are starting towards dst
        """
        all_indexes_set = set(range(0, self.k/2))
        used_indexes_set = {self.dc_graph_elep.get_router_index(self.getGatewayRouter(self.getSourcePrefixFromFlow(flow))) for flow in ongoing_flows}
        return list(all_indexes_set.difference(used_indexes_set))

    def computeExpectedFlowsCost(self, complete_dag, ndag, dst_gw, ongoing_flows):
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

        # Accumulate flows cost and karma for all path combinations in the given DAG
        flows_cost = []
        flows_karma = []

        # For each path combination
        for pcomb in all_path_combinations:

            # Compute flow congestion
            (flowsCost, flowsKarma) = self.computeFlowsCost(complete_dag, ongoing_flows, pcomb)

            # Append it
            flows_cost.append(flowsCost)
            flows_karma.append(flowsKarma)

        # Convert it to a numpy array
        flows_cost = np.asarray(flows_cost)
        flows_karma = np.asarray(flows_karma)

        # Return expected values
        return {'cost': {'mean': flows_cost.mean(), 'std': flows_cost.std()},
                'karma': {'mean': flows_karma.mean(), 'std': flows_karma.std()}}

    def computeFlowsCost(self, complete_dag, ongoing_flows, path_combination):
        """Given a list of flows and a list representing a path taken by each flow, returns
        the total overflow observed by the flows
        """
        # Add the flow sizes to their paths first
        for i, flow in enumerate(ongoing_flows):
            # Get flow estiamted demand
            flow_demand = self.flowDemands.getDemand(flow)
            fpath = path_combination[i]
            links_from_path = self.get_links_from_path(fpath)
            if i == 0:
                # Set all current dag loads to zero
                for (u, v) in complete_dag.edges_iter():
                    if complete_dag[u][v]['current_dag_load'] != 0.0:
                        complete_dag[u][v]['current_dag_load'] = 0.0

                    if (u, v) in links_from_path:
                        complete_dag[u][v]['current_dag_load'] = flow_demand
            else:
                for (u, v) in links_from_path:
                    complete_dag[u][v]['current_dag_load'] += flow_demand

        # Here we acumulate the two measures to asess DAGs
        totalKarma = 0
        totalCost = 0

        # Compute overload and karma on the flow paths
        for fpath in path_combination:

            # Iterate links
            for (u, v) in self.get_links_from_path(fpath):

                # Compute total load used by flows in these paths
                totalLoad = complete_dag[u][v]['fixed_load'] + complete_dag[u][v]['current_dag_load']

                # Compute capacity threshold
                capThreshold = LINK_BANDWIDTH * self.capacity_threshold

                # Compute the spare capacity after the flows are allocated
                spareCapacity = capThreshold - totalLoad

                # Compute values
                if spareCapacity >= 0:
                    overLoad = 0.0
                    underLoad = spareCapacity

                else:
                    overLoad = abs(spareCapacity)
                    underLoad = 0.0

                totalCost += overLoad
                totalKarma += underLoad

        # Return both measures
        return (totalCost, totalKarma)

    def getInitialDagWithoutFlows(self, dst_px, ongoing_flow_keys):
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
            for edge in idag_c.edges_iter():
                # Check src and dst
                (u, v) = edge

                # Get all the loads except those of ongoing flows
                edge_load = self.getLinkLoad(edge, exclude=ongoing_flow_keys)

                # Insert it in the new dag
                idag_c[u][v]['fixed_load'] = edge_load
                idag_c[u][v]['current_dag_load'] = 0.0

        # Return the dag
        return idag_c

    def getOngoingFlowKeysToDst(self, dst, from_pod=None):
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
                fws = []
                for edge in incoming_edges:
                    for fkey, data in self.edges_to_flows[edge].iteritems():
                        if self.getDestinationPrefixFromFlow(data['flow']) == dst:
                            fws.append(fkey)
            else:
                # Only flows from source pod
                fws = []
                for edge in incoming_edges:
                    for fkey, data in self.edges_to_flows[edge].iteritems():
                        flow = data['flow']
                        if self.getDestinationPrefixFromFlow(flow) == dst:
                            if self.getSourcePodFromFlow(flow) == from_pod:
                                fws.append(flow)
            return fws

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

        # Get flow demand
        flow_demand = self.flowDemands.getDemand(flow)

        # Iterate paths anad add flow size
        for path in paths_to_loads.iterkeys():
            capacities = paths_to_loads[path]
            paths_to_loads[path] = [c + flow_demand for c in capacities]

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


class FullDAGShifter(ElephantDAGShifter):
    def __init__(self, sample=False, *args, **kwargs):
        super(FullDAGShifter, self).__init__(sample=sample, *args, **kwargs)

        # Start thread
        self.mice_dag_shifter = True
        self.createMiceEstimatorThread()
        self.miceEstimatorThread.start()

    def _getAlgorithmName(self):
        return "full-dag-shifter_k_{0}".format(self.k)

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
                        choices=["ecmp", "mice-dag-shifter", "elephant-dag-shifter", "full-dag-shifter"],
                        default = 'ecmp')

    # General arguments
    parser.add_argument('-k', '--k', help='Fat-Tree parameter', type=int, default=4)
    parser.add_argument('-t', '--test', help='Test controller', action="store_true", default=False)


    # DAG-shifter arguments
    parser.add_argument('--cong_prob', help='Threshold of congestion probability', type=float, default=0.0)
    parser.add_argument('--cap_threshold', help='Capacity threshold at which we consider congestion', type=float, default=1)
    parser.add_argument('--sample', help='Wether to sample on DAGs or always find the best DAG', action="store_true", default=False)


    args = parser.parse_args()

    log.setLevel(logging.DEBUG)
    log.info("Starting Controller - k = {0} , algorithm = {1}".format(args.k, args.algorithm))

    if args.algorithm.lower() == 'ecmp':
        lb = ECMPController(doBalance=args.doBalance, k=args.k)

    elif args.algorithm.lower() == 'mice-dag-shifter':
        lb = MiceDAGShifter(doBalance=args.doBalance, k=args.k)

    elif args.algorithm.lower() == 'elephant-dag-shifter':
        log.info("Capacity threshold: {0}".format(args.cap_threshold))
        log.info("Max congestion probability: {0}".format(args.cong_prob))
        log.info("Sample on DAGs? {0}".format(args.sample))

        lb = ElephantDAGShifter(doBalance=args.doBalance, k=args.k,
                                congProb_threshold=args.cong_prob,
                                capacity_threshold=args.cap_threshold,
                                sample=args.sample)
    elif args.algorithm.lower() == 'full-dag-shifter':
        lb = FullDAGShifter(doBalance=args.doBalance, k=args.k,
                                sample=args.sample)
    else:
        print("Unknown algorithm: {0}".format(args.algorithm))
        exit()

    # Run the controller
    lb.run()