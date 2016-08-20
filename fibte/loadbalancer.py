#!/usr/bin/python

from fibbingnode.algorithms.southbound_interface import SouthboundManager
from fibbingnode import CFG as CFG_fib
from fibte.misc.unixSockets import UnixServer
import threading
import os
import time
import argparse
import json
import networkx as nx
from fibte.misc.dc_dag import DCDag
import random
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
    def __init__(self, doBalance = True, k=4):

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
        self.dc_graph = self._addDCInfoToGraph(self.network_graph.copy())


        # Here we store the current dags for each destiantion
        self.dags = self._createInitialDags()

        # Start getLoads thread that reads from counters
        os.system(getLoads_path + ' -k {0} &'.format(self.k))

        # Start getLoads thread reads from link usage
        thread = threading.Thread(target=self._getLoads, args=([1]))
        thread.setDaemon(True)
        thread.start()

    def isCoreRouter(self, router):
        return (self.dc_graph.is_router(router) and self.dc_graph.node[router].has_key('core'))

    def isAggregationRouter(self, router):
        return (self.dc_graph.is_router(router) and self.dc_graph.node[router].has_key('aggregation'))

    def isEdgeRouter(self, router):
        return (self.dc_graph.is_router(router) and self.dc_graph.node[router].has_key('edge'))

    def getRouterPod(self, router):
        """
        Returns the pod of the given router. Raises error
        if router is not edge or aggregation.
        :param router:
        :return: pod number
        """
        if self.dc_graph.is_router(router):
            if self.isAggregationRouter(router) or self.isEdgeRouter(router):
                return self.dc_graph.node[router]['pod']
            else:
                raise ValueError("{0} is not an aggregation or edge router".format(router))
        else:
            raise ValueError("{0} is not a router!".format(router))

    def getAllRouters(self):
        return [router for router in self.dc_graph.routers
                if any([True for type in ['edge', 'aggregation', 'core']
                        if type in self.dc_graph.node[router].keys()])]

    def getEdgeRouters(self, pod=-1):
        """
        Returns a list of edge routers. If filtered by pod,
        only the aggregation routers for that pod are returned.

        :param pod:
        :return:
        """
        if pod == -1:
            return [router for router in self.dc_graph.routers if self.isEdgeRouter(router)]
        else:
            return [router for router in self.dc_graph.routers if self.isEdgeRouter(router) and self.getRouterPod(router) == pod]

    def getAggregationRouters(self, pod=-1):
        """
        Returns a list of aggregation routers. If filtered by pod,
        only the aggregation routers for that pod are returned.

        :param pod:
        :return:
        """
        if pod == -1:
            return [router for router in self.dc_graph.routers if self.isAggregationRouter(router)]
        else:
            return [router for router in self.dc_graph.routers if self.isAggregationRouter(router) and self.getRouterPod(router) == pod]

    def getCoreRouters(self):
        """
        :return: a list of all core routers in the topology
        """
        return [router for router in self.dc_graph.routers if self.isCoreRouter(router)]

    def getConnectedCoreRouters(self, aggregationRouter):
        """
        Given an aggregation router, returns the connected core routers.
        :param aggregationRouter:
        :return:
        """
        if self.isAggregationRouter(aggregationRouter):
            return [r for r in self.getCoreRouters() if self.dc_graph[aggregationRouter].has_key(r)]
        else:
            raise ValueError('{0} is not an aggregation router'.format(aggregationRouter))

    def getConnectedAggregation(self, coreRouter, pod=-1):
        """
        Given a core router, returns the connected aggregatoin routers. If pod number is given, it returns
        the corresponding aggregatoin router connected in that pod.
        :param coreRouter:
        :param pod:
        :return:
        """
        if self.isCoreRouter(coreRouter):
            if pod == -1:
                return [r for r in self.dc_graph[coreRouter].keys() if
                        self.dc_graph.is_router(r) and self.dc_graph[coreRouter][r]['direction'] == 'downlink']
            else:
                return [r for r in self.dc_graph[coreRouter].keys()
                        if self.dc_graph.is_router(r)
                        and self.dc_graph[coreRouter][r]['direction'] == 'downlink'
                        and self.getRouterPod(r) == pod][0]
        else:
            raise ValueError("{0} is not a core router".format(coreRouter))

    def getGatewayRouter(self, prefix):
        """
        Given a prefix, returns the connected edge router
        :param prefix:
        :return:
        """
        if self.dc_graph.is_prefix(prefix):
            edgeRouters = self.getEdgeRouters()
            return [edgeRouter for edgeRouter in edgeRouters if self.dc_graph[edgeRouter].has_key(prefix)][0]
        else:
            raise ValueError

    def getRouterFromIndex(self, type, index, pod=-1):
        """
        Given the router type, index and optionally pod number,
        returns the corresponding router id

        :param type: edge, aggregation or core
        :param index:
        :param pod:
        :return:
        """
        if type == 'core' and pod == -1:
            if (index >= 0 and index <= ((self.k**2)/4-1)):
                template = "r_c{0}".format(index)
                return self.topology.routersIdMapping["nameToId"][template]
            else:
                raise ValueError("Index out of core router range: 0 =< index =< k**2/4 -1")

        elif pod != -1:
            if index >= 0 and index <= ((self.k/2)-1):
                if type == 'aggregation':
                    template = "r_{0}_a{1}".format(pod, index)
                    return self.topology.routersIdMapping["nameToId"][template]

                elif type == 'edge':
                    template = "r_{0}_e{1}".format(pod, index)
                    return self.topology.routersIdMapping["nameToId"][template]

            else:
                raise ValueError("Index out of {0} router range: 0 =< index =< k**2/4 -1".format(type))

        else:
            raise ValueError("Wrong router type: {0}".format(type))

    def getConnectedPrefixes(self, edgeRouter):
        """
        Given an edge router, returns the connected prefixes
        :param edgeRouter:
        :return:
        """
        if self.isEdgeRouter(edgeRouter):
            return [r for r in self.dc_graph[edgeRouter].keys() if self.dc_graph.is_prefix(r)]
        else:
            raise ValueError("{0} is not an edge router".format(edgeRouter))

    def _addDCInfoToGraph(self, graph):

        # Now iterate router links and say if they are uplink or downlink
        visited = []
        for (u,v) in graph.router_links:

            if (u, v) not in visited and (v, u) not in visited:
                uname = self.topology.getRouterName(u)
                utype = self.topology.getRouterType(uname)
                vname = self.topology.getRouterName(v)
                vtype = self.topology.getRouterType(vname)

                # Add pod number if not core
                if utype != 'core':
                    upod = self.topology.getRouterPod(uname)
                    graph.node[u]['pod'] = upod
                if vtype != 'core':
                    vpod = self.topology.getRouterPod(vname)
                    graph.node[v]['pod'] = vpod

                # Case uplink from edge to aggregation
                if utype == 'edge' and vtype == 'aggregation':
                    graph.node[u]['edge'] = True
                    graph.node[v]['aggregation'] = True
                    graph.edge[u][v]['direction'] = 'uplink'
                    graph.edge[v][u]['direction'] = 'downlink'

                # Case downling from aggregation to edge
                elif utype == 'aggregation' and vtype == 'edge':
                    graph.node[u]['aggregation'] = True
                    graph.node[v]['edge'] = True
                    graph.edge[u][v]['direction'] = 'downlink'
                    graph.edge[v][u]['direction'] = 'uplink'

                # Case downling from core to aggregation
                elif utype == 'core' and vtype == 'aggregation':
                    graph.node[u]['core'] = True
                    graph.node[v]['aggregation'] = True
                    graph.edge[u][v]['direction'] = 'downlink'
                    graph.edge[v][u]['direction'] = 'uplink'

                # Case upling aggregation to core
                elif utype == 'aggregation' and vtype == 'core':
                    graph.node[u]['aggregation'] = True
                    graph.node[v]['core'] = True
                    graph.edge[u][v]['direction'] = 'uplink'
                    graph.edge[v][u]['direction'] = 'downlink'

                visited.append((u, v))

        return graph

    def _getPrintableDag(self, dag):
        dag_c = nx.DiGraph()
        for (u,v) in dag.edges_iter():
            dag_c.add_edge(self.topology.getRouterName(u), self.topology.getRouterName(v))
        return dag_c

    def _printRouterName(self, router):
        print self.topology.getRouterName(router)

    def _createInitialDags(self):
        """
        Populates the self.dags dictionary for each existing prefix in the network
        """
        # Result is stored here
        dags = {}

        # Get edge routers
        edgeRouters = self.getEdgeRouters()

        # Get aggregation routers
        aggregationRouters = self.getAggregationRouters()

        # Get Core routers
        coreRouters = self.getCoreRouters()


        for prefix in self.network_graph.prefixes:
            # Create a new dag
            dag = nx.DiGraph()

            # Get Edge router connected to prefix
            gatewayRouter = self.getGatewayRouter(prefix)

            # Get pod
            pod = self.getRouterPod(gatewayRouter)

            # Add uplinks edge->aggr
            for edgeRouter in edgeRouters:
                if edgeRouter != gatewayRouter:
                    # Get pod of edgeRouter
                    pod_eR = self.getRouterPod(edgeRouter)

                    # Add edges to aggregation routers
                    add_aggRouterLinks = [dag.add_edge(edgeRouter, aR) for aR in self.getAggregationRouters(pod=pod_eR)]

            #  Get aggregaton routers of the same pod
            aggrRoutersSamePod = self.getAggregationRouters(pod=pod)

            # Add uplinks aggr->core
            for aggrRouter in aggregationRouters:
                if aggrRouter in aggrRoutersSamePod:
                    # Add downlins to gateway edge router
                    dag.add_edge(aggrRouter, gatewayRouter)

                else:
                    # Add uplinks to core routers
                    add_coreRouterLinks = [dag.add_edge(aggrRouter, cR) for cR in self.getConnectedCoreRouters(aggrRouter)]

            # Add downlinks core -> pod
            for coreRouter in coreRouters:
                # Only downling from core router to aggregation router of specific pod
                action = [dag.add_edge(coreRouter, aR) for aR in  self.getConnectedAggregation(coreRouter, pod=pod)]

            # Add dag
            dags[prefix] = dag

        return dags

    def getRandomUpLinkChoice(self, max_depth=2, curr_depth=0):
        """
        Recursive function that returns a random number of random choices
        that correspond to the paths taken  from an edge node to a set of
        core routers.
        """
        if curr_depth == max_depth:
            return []

        else:
            if curr_depth == 0:
                # Draw how many
                n = random.randint(1, self.k/2)

                # Draw which
                indexes = range(0, self.k/2)
                random.shuffle(indexes)
                chosen = indexes[:n]

            elif curr_depth == 1:
                # Draw how many
                n = random.randint(1, (self.k**2/4))

                # Draw which
                indexes = range(0, (self.k**2/4))
                random.shuffle(indexes)
                chosen = indexes[:n]

            new_current_depth = curr_depth + 1
            return [(c, self.getRandomUpLinkChoice(max_depth, curr_depth=new_current_depth)) for c in chosen]

    def getRandomDag(self, prefix):
        """
        Returns a random dag (upwards) from every
        other prefix to specified target prefix.

        :param prefix:
        :return:
        """
        # Result is stored here
        rdag = nx.DiGraph()

        # Get first the gateway edge router of the prefix
        gwRouter = self.getGatewayRouter(prefix)

        # Get pod number
        prefixPod = self.getRouterPod(gwRouter)

        # Then starting at every other edge
        for router in self.getEdgeRouters():

            # Get pod of the router
            rPod = self.getRouterPod(router)

            # If they are in different pods
            if rPod != prefixPod:

                # Make uplink choices
                choices = self.getRandomUpLinkChoice(max_depth=2)

                # Add corresponding edges
                for choice in choices:
                    # Fetch aggregation router chosen
                    aggChosen = self.getRouterFromIndex(type='aggregation', index=choice[0], pod=rPod)

                    # Add the edge from edge router to randomly chosen aggregation router
                    rdag.add_edge(router, aggChosen)

                    # Traverse now the core router choices from each aggregation
                    for subchoice in choice[1]:
                        # Fetch core router
                        coreRouter = self.getRouterFromIndex(type='core', index=subchoice[0])

                        # Add edge aggregation -> core
                        rdag.add_edge(aggChosen, coreRouter)

                        # Add also here the downlink edge core -> aggregation
                        corrAggr = self.getConnectedAggregation(coreRouter=coreRouter, pod=prefixPod)
                        rdag.add_edge(coreRouter, corrAggr)

            # Edge router in the same pod
            else:
                # Check that it's not himself
                if router != gwRouter:
                    # Draw also the used links (inside pod)
                    choices = self.getRandomUpLinkChoice(max_depth=1)

                    # Add corresponding edges
                    for choice in choices:
                        # Fetch aggregation router chosen
                        aggChosen = self.getRouterFromIndex(type='aggregation', index=choice[0], pod=rPod)

                        # Add the edge from edge router to randomly chosen aggregation router
                        rdag.add_edge(router, aggChosen)

                        # Add also the downlink fixed link: aggregation -> edge router
                        rdag.add_edge(aggChosen, gwRouter)

        return rdag

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


    prefix = lb.dags.keys()[3]
    pdag = lb.getRandomDag(prefix)
    import ipdb; ipdb.set_trace()

    lb.run()

    #import ipdb; ipdb.set_trace()
