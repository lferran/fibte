from networkx import DiGraph
from fibte.misc.topology_graph import TopologyGraph
from fibte import CFG
import os
import random
import networkx as nx
import fibte.misc.ipalias as ipalias
import itertools as it
from fibte import log
import logging
import time

import matplotlib.pyplot as plt

def time_func(function):
    def wrapper(*args,**kwargs):
        t = time.time()
        res = function(*args,**kwargs)
        log.debug("{0} took {1}s to execute".format(function.func_name, time.time()-t))
        return res
    return wrapper

"""
Module that defines the DCGraph object, which extends the nx.DiGraph class
to support a DataCenter-like DAG structure.
"""

tmp_files = CFG.get("DEFAULT", "tmp_files")
db_topo = CFG.get("DEFAULT", "db_topo")

class DCDiGraph(DiGraph):
    """
    Common parent class that holds all the methods to add and remove nodes and edges
    in a fat-tree-like datacenter network.
    """
    def __init__(self, k=4, prefix_type='primary', *args, **kwargs):
        super(DCDiGraph, self).__init__(*args, **kwargs)
        # DC parameter
        self.k = k

        # Set prefix type
        self.prefix_type = prefix_type

        # We store prefix-gateway here
        self.prefix_gateway_bindings = {'prefixToGateway':{}, 'gatewayToPrefix': {}}

        # Prefixes are not stored as nodes
        self.prefixes = {}

    def is_correct_prefix(self, prefix):
        """Checks validity of prefix wrt specified prefix type"""
        return (ipalias.get_ip_prefix_type(prefix) == self.prefix_type)

    def add_edge_router(self, routerid, routername, index, pod):
        if pod >= 0 and pod <= self.k - 1:
            if index >= 0 and index <= self.k / 2 - 1:
                self.add_node(n=routerid, attr_dict={'name': routername, 'type': 'edge','index': index, 'pod': pod})
            else:
                raise ValueError("Index number {0} out of range 0 <= i <= k/2 - 1".format(pod))
        else:
            raise ValueError("Pod number {0} out of range 0 <= i <= k - 1".format(pod))

    def add_aggregation_router(self, routerid, routername, index, pod):
        if pod >= 0 and pod <= self.k - 1:
            if index >= 0 and index <= self.k / 2 - 1:
                self.add_node(n=routerid, attr_dict={'name': routername, 'type': 'aggregation', 'index': index, 'pod': pod})
            else:
                raise ValueError("Index number {0} out of range 0 <= i <= k/2 - 1".format(pod))
        else:
            raise ValueError("Pod number {0} out of range 0 <= i <= k - 1".format(pod))

    def add_core_router(self, routerid, routername, index):
        if index >= 0 and index <= ((self.k/2)**2)-1:
            self.add_node(n=routerid, attr_dict={'name': routername, 'type': 'core', 'index': index})
        else:
            raise ValueError("Index number {0} out of range 0 <= i <= (k/2)**2 - 1".format(index))

    def add_destination_prefix(self, prefix, gateway):
        """Adds a destination prefix attached to gateway router"""
        if not self.is_correct_prefix(prefix):
            raise ValueError("Prefix not valid: {0}, DCDigraph prefiy_type is {1}".format(prefix, self.prefix_type))

        if self.is_edge(gateway):
            pod = self.get_router_pod(gateway)
            self.prefixes[prefix] = {'pod':pod, 'gateway':gateway}

            # Add prefix in structure
            if prefix not in self.prefix_gateway_bindings['prefixToGateway'].keys():
                self.prefix_gateway_bindings['prefixToGateway'][prefix] = gateway

            if gateway not in self.prefix_gateway_bindings['gatewayToPrefix'].keys():
                self.prefix_gateway_bindings['gatewayToPrefix'][gateway] = [prefix]

            else:
                if prefix not in self.prefix_gateway_bindings['gatewayToPrefix'][gateway]:
                    self.prefix_gateway_bindings['gatewayToPrefix'][gateway] += [prefix]
        else:
            raise ValueError("Router {0} is not an edge router, so we can't set him a destination prefix".format(gateway))

    def is_edge(self, routerid):
        return self.node[routerid]['type'] == 'edge'

    def is_aggregation(self, routerid):
        return self.node[routerid]['type'] == 'aggregation'

    def is_core(self, routerid):
        return self.node[routerid]['type'] == 'core'

    def is_destination_prefix(self, node):
        return node in self.prefixes.keys()

    def edge_routers(self, data=False):
        if data:
            return [(r, d) for (r, d) in self.nodes_iter(data=True) if self.is_edge(r)]
        else:
            return [r for r in self.nodes_iter() if self.is_edge(r)]

    def edge_routers_iter(self, data=False):
        if data:
            return iter([(r, d) for (r, d) in self.nodes_iter(data=True) if self.is_edge(r)])
        else:
            return iter([r for r in self.nodes_iter() if self.is_edge(r)])

    def aggregation_routers(self, data=False):
        if data:
            return [(r, d) for (r, d) in self.nodes_iter(data=True) if self.is_aggregation(r)]
        else:
            return [r for r in self.nodes_iter() if self.is_aggregation(r)]

    def aggregation_routers_iter(self, data=False):
        if data:
            return iter([(r, d) for (r, d) in self.nodes_iter(data=True) if self.is_aggregation(r)])
        else:
            return iter([r for r in self.nodes_iter() if self.is_aggregation(r)])

    def core_routers(self, data=False):
        if data:
            return [(r, d) for (r, d) in self.nodes_iter(data=True) if self.is_core(r)]
        else:
            return [r for r in self.nodes_iter() if self.is_core(r)]

    def core_routers_iter(self, data=False):
        if data:
            return iter([(r, d) for (r, d) in self.nodes_iter(data=True) if self.is_core(r)])
        else:
            return iter([r for r in self.nodes_iter() if self.is_core(r)])

    def destination_prefixes(self, data=False):
        if data:
            return [(r, d) for (r, d) in self.prefixes.iteritems()]
        else:
            return [r for r in self.prefixes.keys()]

    def destination_prefixes_iter(self, data=False):
        if data:
            return iter([(r, d) for (r, d) in self.prefixes.iteritems()])
        else:
            return iter([r for r in self.prefixes.keys()])

    def get_router_name(self, routerid):
        return self.node[routerid]['name']

    def get_router_type(self, routerid):
        return self.node[routerid]['type']

    def get_router_index(self, routerid):
        return self.node[routerid]['index']

    def get_router_pod(self, routerid):
        if not self.is_core(routerid):
            return self.node[routerid]['pod']
        else:
            raise ValueError("Router {0} is a core router - it has no pod number".format(routerid))

    def get_destination_prefix_pod(self, dest):
        if not self.is_correct_prefix(dest):
            raise ValueError("Prefix not valid: {0}, DCDigraph prefiy_type is {1}".format(dest, self.prefix_type))

        if dest in self.prefixes.keys():
            return self.prefixes[dest]['pod']
        else:
            raise ValueError("{0} is not a destination prefix".format(dest))

    def get_router_position(self, routerid):
        if self.is_core(routerid):
            return {key: value for key, value in self.node[routerid].iteritems() if key in ['type', 'index']}
        else:
            return {key: value for key, value in self.node[routerid].iteritems() if key in ['type', 'index', 'pod']}

    def get_router_from_position(self, type, index, pod=None):
        """
        Given a router position, returs the corresponding routerid.
        """
        if pod == None and type == 'core' and self._valid_core_index(index):
            return [r for (r, data) in self.core_routers_iter(data=True) if data['index'] == index][0]

        elif pod != None and self._valid_pod_number(pod) and type in ['edge', 'aggregation'] and (self._valid_edge_index(index) or self._valid_aggregation_index(index)):
            if type == 'edge':
                return [r for (r, data) in self.edge_routers_iter(data=True) if data['index'] == index and data['pod'] == pod][0]
            else:
                return [r for (r, data) in self.aggregation_routers_iter(data=True) if data['index'] == index and data['pod'] == pod][0]
        else:
            raise ValueError("Wrong router position")

    def get_destination_prefix_gateway(self, prefix):
        if self.is_destination_prefix(prefix):
            return self.prefix_gateway_bindings['prefixToGateway'][prefix]
        else:
            raise ValueError("{0} is not a destiantion prefix".format(prefix))

    def get_connected_destination_prefixes(self, routerid):
        if self.is_edge(routerid):
            if routerid in self.prefix_gateway_bindings['gatewayToPrefix'].keys():
                return self.prefix_gateway_bindings['gatewayToPrefix'][routerid]
            else:
                return []
        else:
            raise ValueError("{0} is not an edge router".format(routerid))

    def have_same_pod(self, a, b):
        """
        Returns true only if a and b are from the same pod
        """
        if not self.is_core(a) and not self.is_core(b):
            return self.get_router_pod(a) == self.get_router_pod(b)
        else:
            raise ValueError("Either a:{0} or b:{1} are core routers".format(a, b))

    def _valid_edge_index(self, edge_index):
        return self._valid_aggregation_index(edge_index)

    def _valid_aggregation_index(self, agg_index):
        return agg_index >= 0 and agg_index <= ((self.k / 2) - 1)

    def _valid_core_index(self, core_index):
        return core_index >= 0 and core_index <= (((self.k / 2) ** 2) - 1)

    def _valid_pod_number(self, pod):
        return (pod >= 0) and (pod <= self.k - 1)

    def _valid_aggregation_core_indexes(self, agg_index, core_index):
        if self._valid_aggregation_index(agg_index) and self._valid_core_index(core_index):
            return core_index >= agg_index * (self.k / 2) and core_index <= (((agg_index + 1) * (self.k / 2)) - 1)
        else:
            return False

    def _printable_nodes(self, data=False):
        """
        """
        if not data:
            return [self.node[node]['name'] for node in self.nodes_iter()]
        else:
            return [(self.node[node]['name'], d) for node, d in self.nodes_iter(data=True)]

    def _printable_edges(self, data=False):
        """"""
        if not data:
            return [(self.get_router_name(u), self.get_router_name(v)) for (u, v) in self.edges_iter()]
        else:
            return [((self.get_router_name(u), self.get_router_name(v)), d) for (u, v), d in self.edges_iter(data=True)]

    def get_upper_tier(self, routerid):
        """
        Returns a list of the immediate valid upper layer
        routers (even if there is no edge between them yet)

        :param routerid:
        :return:
        """
        if not self.is_core(routerid):
            if self.is_edge(routerid):
                return [ag for ag in self.aggregation_routers_iter() if self.is_valid_uplink(routerid, ag)]

            # Is aggregation router
            else:
                # Return agg->core
                return [cr for cr in self.core_routers_iter() if self.is_valid_uplink(routerid, cr)]
        else:
            raise ValueError("Core routers do not have upper layer: {0}".format(routerid))

    def get_lower_tier(self, routerid):
        """
        Retunrs a list of the immediate lower layer routers
        (even if there is no edge between them yet)

        :param routerid:
        :return:
        """
        if not self.is_edge(routerid):

            if self.is_core(routerid):
                # Check for core->aggregation link validity
                return [r for r in self.aggregation_routers_iter() if self.is_valid_downlink(routerid, r)]

            # Aggregation router
            else:
                return [r for r in self.edge_routers_iter() if self.is_valid_downlink(routerid, r)]
        else:
            raise ValueError("Edge routers do not have lower layer: {0}".format(routerid))

    def is_valid_uplink(self, lower_rid, upper_rid):
        """
        Checks if the uplink (lower_rid, upper_rid) is valid in a fat-tree topology
        :param lower_rid: router id
        :param upper_rid: router id
        :return: boolean
        """
        # Check right combination of router types
        if (self.is_edge(lower_rid) and self.is_aggregation(upper_rid)) or (self.is_aggregation(lower_rid) and self.is_core(upper_rid)):
            # Edge->aggregation uplink
            if self.is_edge(lower_rid):
                # Check if they are from the same pod
                if self.get_router_pod(lower_rid) == self.get_router_pod(upper_rid):
                    return True

            # Aggregation->core uplink
            else:
                # Get index of aggregation and core router
                ai = self.get_router_index(lower_rid)
                ri = self.get_router_index(upper_rid)
                # Check index validity
                if self._valid_aggregation_core_indexes(ai, ri):
                    return True

        return False

    def is_valid_downlink(self, upper_rid, lower_rid):
        """
        Checks if the downlink (upper_rid, lower_rid) is valid in a fat-tree topology
        :param upper_rid: router id
        :param lower_rid: router id
        :return: boolean
        """
        # Check correct router type combination for downlink
        if (self.is_aggregation(upper_rid) and self.is_edge(lower_rid)) or (self.is_aggregation(lower_rid) and self.is_core(upper_rid)):
            # aggregation->edge downlink
            if self.is_edge(lower_rid):
                # If they are from the same pod
                lpod = self.get_router_pod(lower_rid)
                rpod = self.get_router_pod(upper_rid)
                # Check if they are from the same pod
                if lpod == rpod:
                    return True

            # Core->aggregation downlink
            else:
                # Get index of aggregation and core router
                ri = self.get_router_index(upper_rid)
                ai = self.get_router_index(lower_rid)

                # Check that uplink is valid wrt the DC structure
                if self._valid_aggregation_core_indexes(ai, ri):
                    return True

        return False

    def add_uplink(self, lower_rid, upper_rid):
        """
        Adds a DC uplink by checking the corresponding constrains.
        Assumes lower_rid and upper_rid already in the graph.

        If uplink is not valid, the function silently finishes
        """
        if self.is_valid_uplink(lower_rid, upper_rid):
            # Add uplink
            self.add_edge(lower_rid, upper_rid, {'direction': 'uplink'})
        else:
            lr_name = self.get_router_name(lower_rid)
            ur_name = self.get_router_name(upper_rid)
            raise ValueError("This uplink is not valid: {0} -> {1}".format(lr_name, ur_name))

    def add_downlink(self, upper_rid, lower_rid):
        """
        Adds a DC downlink by checking the corresponding constrains
        Assumes lower_rid and upper_rid already in the graph.

        If uplink is not valid, the function silently finishes
        """
        if self.is_valid_downlink(upper_rid, lower_rid):
            # Add downlink
            self.add_edge(upper_rid, lower_rid, {'direction': 'downlink'})
        else:
            lr_name = self.get_router_name(lower_rid)
            ur_name = self.get_router_name(upper_rid)
            raise ValueError("This downlink is not valid: {0} -> {1}".format(ur_name, lr_name))

    def print_stuff(self, to_print):
        if to_print:
            if isinstance(to_print[0], list) or isinstance(to_print[0], tuple):
                new_to_print = type(to_print)()
                for a in to_print:
                    ep = []
                    for inner in a:
                        try:
                            s = self.get_router_name(inner)
                            ep.append(s)
                        except:
                            ep.append(inner)
                    new_to_print.append(ep)
                return new_to_print
            else:
                if isinstance(to_print, list) or isinstance(to_print, tuple):
                    if isinstance(to_print, list):
                        new_to_print = type(to_print)()
                        for a in to_print:
                            try:
                                e = self.get_router_name(a)
                                new_to_print.append(e)
                            except:
                                new_to_print.append(a)
                        return new_to_print
                    else:
                        return map(self.get_router_name, to_print)
                else:
                    return self.get_router_name(to_print)
        else:
            return []

class DCGraph(DCDiGraph):
    """
    Models the DC graph as a networkx DiGraph subclass object

    There should be only once instance of DCGraph in the controller.
    It builds itself thanks to the topologyDB object, at startup.
    """
    def __init__(self, *args, **kwargs):
        super(DCGraph, self).__init__(*args, **kwargs)

        # Object that connects to the topology db dile
        self.topo = TopologyGraph(getIfindexes=True, db=os.path.join(tmp_files, db_topo))

        # Build himself from topoDB information
        self.build_from_topo_db()

    def build_from_topo_db(self):
        # Add first the cores
        coreRouters=self.topo.getCoreRouters()
        for router in coreRouters:
            routerid = self.topo.getRouterId(router)
            index = self.topo.getRouterIndex(router)
            self.add_core_router(routerid=routerid, routername=router, index=index)

        # Add the aggregation
        aggrRouters = self.topo.getAggregationRouters()
        for router in aggrRouters:
            routerid = self.topo.getRouterId(router)
            index = self.topo.getRouterIndex(router)
            pod = self.topo.getRouterPod(router)
            self.add_aggregation_router(routerid=routerid, routername=router, index=index, pod=pod)
            # Add uplinks and downlinks
            for cr in coreRouters:
                crindex = self.topo.getRouterIndex(cr)
                if self._valid_aggregation_core_indexes(index, crindex):
                    crid = self.topo.getRouterId(cr)
                    self.add_uplink(routerid, crid)
                    self.add_downlink(crid, routerid)

        # Add the edge
        edgeRouters = self.topo.getEdgeRouters()
        for router in edgeRouters:
            routerid = self.topo.getRouterId(router)
            index = self.topo.getRouterIndex(router)
            pod = self.topo.getRouterPod(router)
            self.add_edge_router(routerid=routerid, routername=router, index=index, pod=pod)
            # Add uplinks and downlinks
            for ar in aggrRouters:
                arpod = self.topo.getRouterPod(ar)
                if arpod == pod:
                    arid = self.topo.getRouterId(ar)
                    self.add_uplink(routerid, arid)
                    self.add_downlink(arid, routerid)

        # Add the destination prefixes
        prefixes = self.topo.getInitialNetworkPrefixes()
        for prefix in prefixes:
            gw = self.topo.getGatewayRouterFromNetworkPrefix(prefix)
            gwid = self.topo.getRouterId(gw)

            # Convert it to secondary prefix if necessary
            if self.prefix_type == 'secondary':
                prefix = ipalias.get_secondary_ip_prefix(prefix)

            self.add_destination_prefix(prefix=prefix, gateway=gwid)

    def get_default_ospf_dag(self, prefix):
        """
        Given a sink edge router, returns the default fat-tree dag from
        all other hosts to the sink.

        :param sink: router id of the sink's router
        :return: DCDAg
        """
        prefix_pod = self.get_destination_prefix_pod(prefix)
        gateway = self.get_destination_prefix_gateway(prefix)
        if self.is_destination_prefix(prefix):
            # Create empty-links dag from self DCGraph
            dc_dag = DCDag(sink_id=gateway, dst_prefix=prefix, k=self.k, dc_graph=self, prefix_type=self.prefix_type)

            # Add uplinks
            add_uplinks1 = [dc_dag.add_uplink(er, ar) for er in self.edge_routers_iter() for ar in self.get_upper_tier(er) if er != gateway]
            add_uplinks2 = [dc_dag.add_uplink(ar, cr) for ar in self.aggregation_routers_iter() for cr in self.get_upper_tier(ar) if self.get_router_pod(ar) != prefix_pod]

            # Add downlinks
            add_downlinks1 = [dc_dag.add_downlink(cr, ar) for cr in self.core_routers_iter() for ar in self.get_lower_tier(cr) if self.get_router_pod(ar) == prefix_pod and dc_dag.is_valid_downlink(cr, ar)]
            add_downlinks2 = [dc_dag.add_downlink(ar, er) for ar in self.aggregation_routers_iter() for er in self.get_lower_tier(ar) if self.get_router_pod(ar) == prefix_pod and dc_dag.is_valid_downlink(ar, er)]

            # Add links to gateway
            dc_dag.add_destination_prefix(prefix, gateway)

            # Set initial edges
            dc_dag._set_initial_edges()

            # Return the DCDag object
            return dc_dag
        else:
            raise ValueError("Prefix must be an existing destination prefix. {0} isn't".format(prefix))

    @staticmethod
    def weighted_choice(choices):
        """
        *args example: [("e",weight_e),('m',weight_m)]
        """
        total = sum(w for c, w in choices)
        r = random.uniform(0, total)
        upto = 0
        for c, w in choices:
            if upto + w >= r:
                return c
            upto += w
        assert False, "Shouldn't get here"

    def get_random_dag_from_probability_dcgraph(self, prefix, link_probabilities):
        """Computes a random DAG towards prefix based on the individual link
        probabilities specified in prob_dc_graph.

        link probabilities is a dictionary of dictionaries:
          keyed by source -> dst -> {'probability' attribute in [0,1] and 'changed' [True, False]}
        """
        prefix_pod = self.get_destination_prefix_pod(prefix)
        gateway = self.get_destination_prefix_gateway(prefix)
        if self.is_destination_prefix(prefix):
            # Create empty-links dag from self DCGraph
            dc_dag = DCDag(sink_id=gateway, dst_prefix=prefix, k=self.k, dc_graph=self, prefix_type=self.prefix_type)

            # Add uplinks: first the edge -> aggr links
            for er in dc_dag.edge_routers_iter():
                #log.debug("Edge router: {0}".format(dc_dag.get_router_name(er)))
                if er != gateway:
                    possible_aggr = dc_dag.get_upper_tier(er)
                    #log.debug("Upper tier aggregation routers: {0}".format(self.print_stuff(possible_aggr)))

                    # Compute all possible links
                    possible_edge_to_aggr_links = [(er, ar) for ar in possible_aggr]

                    # Check if probability is zero for all links
                    if all([link_probabilities[link]['final_probability'] <= 0 for link in possible_edge_to_aggr_links]):
                        #log.error("ALL LINKS HAVE PROBABILITY 0 -> Activate ECMP on all links!")
                        for link in possible_edge_to_aggr_links:
                            dc_dag.add_uplink(link[0], link[1])
                        continue

                    # Make some iterations until at least one link is chosen
                    at_least_one_link = False
                    start_time = time.time()
                    links_used = []
                    n_iterations = 0
                    while not at_least_one_link and n_iterations < 10:
                        for link in possible_edge_to_aggr_links:
                            link_prob = link_probabilities[link]['final_probability']
                            #log.debug("{0} with probability {1} of being chosen in DAG".format(self.print_stuff(link), link_prob))
                            use_link = self.weighted_choice([(True, link_prob), (False, 1-link_prob)])
                            if use_link:
                                dc_dag.add_uplink(link[0], link[1])
                                at_least_one_link = True
                                links_used.append(link)
                        n_iterations += 1

                    if not at_least_one_link:
                        #log.error("Probabilities might be too low for all possible links -> Activate ECMP on all links!")
                        for link in possible_edge_to_aggr_links:
                            dc_dag.add_uplink(link[0], link[1])
                        continue

                    #log.debug("It took {0}s to find (edge -> aggr) links".format(time.time()- start_time))
                    #log.debug("Used links: {0}".format(self.print_stuff(links_used)))

            # Compute aggregation routers in destiantoin pod
            for ar in dc_dag.aggregation_routers_iter():
                #log.debug("Aggregation router: {0}".format(self.print_stuff(ar)))
                ar_pod = dc_dag.get_router_pod(ar)
                if ar_pod != prefix_pod:
                    possible_cores = dc_dag.get_upper_tier(ar)
                    #log.debug("Upper tier core routers: {0}".format(self.print_stuff(possible_cores)))

                    possible_agg_core_links = [(ar, cr) for cr in possible_cores]
                    if all([link_probabilities[link]['final_probability'] <= 0 for link in possible_agg_core_links]):
                        #log.error("ALL LINKS HAVE PROBABILITY 0 -> Activate ECMP on all links!")
                        for link in possible_agg_core_links:
                            dc_dag.add_uplink(link[0], link[1])
                        continue

                    at_least_one_link = False
                    start_time = time.time()
                    links_used = []
                    n_iterations = 0
                    while not at_least_one_link and n_iterations < 10:
                        for link in possible_agg_core_links:
                            link_prob = link_probabilities[link]['final_probability']
                            use_link = self.weighted_choice([(True, link_prob), (False, 1 - link_prob)])
                            if use_link:
                                dc_dag.add_uplink(link[0], link[1])
                                at_least_one_link = True
                                links_used.append(link)
                        n_iterations += 1

                    if not at_least_one_link:
                        #log.error("Probabilities might be too low for all possible links -> Activate ECMP on all links!")
                        for link in possible_agg_core_links:
                            dc_dag.add_uplink(link[0], link[1])
                        continue

                    #log.debug("It took {0}s to find (aggr -> core) links".format(time.time() - start_time))
                    #log.debug("Used links: {0}".format(self.print_stuff(links_used)))

            # Add downlinks
            # core -> aggr
            add_downlinks1 = [dc_dag.add_downlink(cr, ar) for cr in self.core_routers_iter() for ar in self.get_lower_tier(cr)
                              if self.get_router_pod(ar) == prefix_pod and dc_dag.is_valid_downlink(cr, ar)]

            # aggr -> edge
            add_downlinks2 = [dc_dag.add_downlink(ar, er) for ar in self.aggregation_routers_iter() for er in self.get_lower_tier(ar)
                              if self.get_router_pod(ar) == prefix_pod and dc_dag.is_valid_downlink(ar, er)]

            # Add links to gateway
            dc_dag.add_destination_prefix(prefix, gateway)

            # Return the DCDag object
            return dc_dag
        else:
            raise ValueError("Prefix must be an existing destination prefix. {0} isn't".format(prefix))

class DCDag(DCDiGraph):
    """
    This class represents a Fat-Tree like Direct Acyclic Graph for a
    certain destination, which must be the single sink in the graph.
    """
    def __init__(self, sink_id, dst_prefix, *args, **kwargs):
        super(DCDag, self).__init__(*args, **kwargs)

        # Set DCDag's sink
        self.sink_id = sink_id

        # Set DCDag's destination prefix
        self.dst_prefix = dst_prefix

        # Parse kwargs
        if 'dc_graph' in kwargs and isinstance(kwargs['dc_graph'], DCGraph):
            # If dc_graph is given, add all routers of dc_graph
            self._build_from_dc_graph(kwargs['dc_graph'])

        self.plot_positions = self.get_plot_positions()

    def _set_initial_edges(self):
        self.all_edges = self.edges()[:]

    def _get_printable(self):
        """
        Helper function that returns a DCDag with router names instead.
        :return:
        """
        other = DCDag(self.sink, self.dst_prefix, self.k)
        for (u, v, data) in self.edges_iter(data=True):
            other.add_edge(self.get_router_name(u), self.get_router_name(v))
        return other

    def _get_printable_uplinks(self, source):
        other = DCDag(self.sink, self.k)
        for ar in self.successors(source):
            other.add_edge(self.get_router_name(source), self.get_router_name(ar))
            for cr in self.successors(ar):
                other.add_edge(self.get_router_name(ar), self.get_router_name(cr))
        return other

    def _build_from_dc_graph(self, dc_graph):
        """
        Adds nodes to DCDag from a DCGraph
        :param dc_graph:
        """
        # Copy all nodes from dc_graph
        self.add_nodes_from(dc_graph.nodes(data=True))

        # Copy all px - gw bindings
        self.prefix_gateway_bindings = dc_graph.prefix_gateway_bindings

        # Set sink data
        sink_name = self.get_router_name(self.sink_id)
        sink_pos = self.get_router_position(self.sink_id)

        # Set the sink
        self._set_sink(self.sink_id, sink_name, sink_pos['index'], sink_pos['pod'], prefix=self.dst_prefix)

    def _set_sink(self, routerid, routername, index, pod, prefix):
        """
        Sets the sink of the DAG to the corresponding edge router
        """
        self.sink_id = routerid
        self.sink = {'id': routerid, 'name': routername, 'index': index, 'pod': pod}
        self.add_edge_router(routerid, routername, index, pod)
        self.add_destination_prefix(prefix, routerid)

    def is_valid_dc_dag(self):
        # Check that number of nodes is complete
        if len(self.nodes()) == (self.k**2 + (self.k**2)/4):
            # Check that at least one path exists from all other nodes to sink
            sink_id = self.get_sink_id()
            return all([nx.has_path(self, node, sink_id) for node in self.nodes_iter() if node != sink_id])
        else:
            print 'Nodes are missing: {0} nodes'.format(len(self.nodes()))
            return False

    def is_sink(self, routerid):
        return routerid == self.sink['id']

    def get_sink_id(self):
        return self.sink['id']

    def get_sink_index(self):
        return self.sink['index']

    def get_sink_name(self):
        return self.sink['name']

    def get_sink_pod(self):
        return self.sink['pod']

    def is_valid_uplink(self, lower_rid, upper_rid):
        """
        Adds a DC uplink by checking the corresponding constrains.
        Aggregation layer of sink's pod can't have uplinks.
        """
        # Check that source is not sink
        if not self.is_sink(lower_rid):
            # Check correct combination of router types
            if (self.is_edge(lower_rid) and self.is_aggregation(upper_rid)) or (self.is_aggregation(lower_rid) and self.is_core(upper_rid)):
                # edge->aggregation uplink
                if self.is_edge(lower_rid):
                    # Check that they are in the same pod
                    if self.get_router_pod(lower_rid) == self.get_router_pod(upper_rid):
                        return True

                # aggregation->core uplink
                else:
                    # Prevent uplink in aggregation layers in same pod as sink
                    apod = self.get_router_pod(lower_rid)
                    if apod != self.get_sink_pod():
                        # Get index of aggregation and core router
                        ai = self.get_router_index(lower_rid)
                        ri = self.get_router_index(upper_rid)

                        # Check that uplink is valid wrt the DC structure
                        if self._valid_aggregation_core_indexes(ai, ri):
                            return True
        return False

    def is_valid_downlink(self, upper_rid, lower_rid):
        """
        Checks if the DC downlink is valid by checking the corresponding constrains
        """
        # Check correct combination of router types for downlink
        if (self.is_aggregation(upper_rid) and self.is_edge(lower_rid)) or (self.is_aggregation(lower_rid) and self.is_core(upper_rid)):
            # aggregation->edge downlink
            if self.is_edge(lower_rid):
                # If they are from the same pod
                lpod = self.get_router_pod(lower_rid)
                rpod = self.get_router_pod(upper_rid)
                # Check that is in same pod as the sink and that the downlinks are directed towards the sink.
                if lpod == rpod == self.get_sink_pod() and lower_rid == self.sink['id']:
                    return True

            # Core->aggregation downlink
            else:
                # Get index of aggregation and core router
                ri = self.get_router_index(upper_rid)
                ai = self.get_router_index(lower_rid)

                # Check that uplink is valid wrt the DC structure
                if self._valid_aggregation_core_indexes(ai, ri):
                    # Check that downlink is only directed to sink's pod
                    if self.get_router_pod(lower_rid) == self.get_sink_pod():
                        return True
        return False

    def add_uplink(self, lower_rid, upper_rid):
        """
        Adds a DC uplink by checking the corresponding constrains.
        Assumes lower_rid and upper_rid already in the graph.
        """
        if self.is_valid_uplink(lower_rid, upper_rid):
            # Add uplink
            self.add_edge(lower_rid, upper_rid, {'direction': 'uplink'})

        else:
            lr_name = self.get_router_name(lower_rid)
            ur_name = self.get_router_name(upper_rid)
            sink_name = self.get_sink_name()
            raise ValueError("This uplink ({0} -> {1}) is not valid for DAG to {2}".format(lr_name, ur_name, sink_name))

    def add_downlink(self, upper_rid, lower_rid):
        """
        Adds a DC downlink by checking the corresponding constrains
        Assumes lower_rid and upper_rid already in the graph.
        """
        if self.is_valid_downlink(upper_rid, lower_rid):
            # Add downlink
            self.add_edge(upper_rid, lower_rid, {'direction': 'downlink'})
        else:
            lr_name = self.get_router_name(lower_rid)
            ur_name = self.get_router_name(upper_rid)
            sink_name = self.get_sink_name()
            raise ValueError("This downlink ({0} -> {1}) is not valid for DAG to {2}".format(ur_name, lr_name, sink_name))

    def add_destination_prefix(self, prefix, gateway):
        """Adds a destination prefix attached to gateway router"""
        if self.get_sink_id() == gateway:
            super(DCDag, self).add_destination_prefix(prefix, gateway)
        else:
            raise ValueError("Router {0} is not an edge router, so we can't set him a destination prefix".format(gateway))

    def get_gateway(self):
        return self.get_sink_id()

    def get_next_hops(self, routerid):
        """
        Returns the list of next hop routers of routerid in the direction
        towards the sink the current DAG.

        :param routerid:
        :return: list of next-hop routers towards the sink. [] if routerid is sink.
        """
        return self.successors(routerid)

    def set_next_hops(self, routerid):
        #TODO:
        # Remove old next hops
        pass

    def modify_random_uplinks(self, src_pod, exclude_edge_indexes=[]):
        """
        Given a source pod, choses a random set of upwards paths
        towards the sink, and modifies the DAG accordingly.
        """
        # Generate random uplink choice from edge
        random_edges = self.get_random_uplinks(src_pod, exclude_edges_indexes=exclude_edge_indexes)
        self.modify_uplinks_from(random_edges)

    def modify_uplinks_from(self, edge_list):
        """
        Given a source pod, choses a random set of upwards paths
        towards the sink, and modifies the DAG accordingly.
        """
        # Simply remove uplink edges from source nodes in edge list
        # that are not in edge_list,
        for (u, v) in edge_list:
            # Add it only if is a valid uplink
            if self.is_valid_uplink(u, v):
                if not self.has_edge(u, v):
                    self.add_uplink(u, v)
            else:
                raise ValueError("Invalid uplink! {0}".format((u, v)))

            # Remove other outgoing edges from u that are not in edge list
            for suc in self.successors(u):
                if (u, suc) not in edge_list:
                    self.remove_edge(u, suc)

    def get_random_uplinks(self, src_pod, exclude_edge_indexes=[]):
        """
        Given a source pod, choses a random set of upwards paths
        towards the sink, and returns the chosen edges.
        """
        # Generate random uplink choice from edge
        (edge_to_aggr, aggr_to_core) = self._get_random_uplink_choice_2(exclude_edge_indexes=exclude_edge_indexes)

        same_sink_pod = False
        if src_pod == self.get_sink_pod():
            same_sink_pod = True

        # Accumulate result here
        chosen_edges = []

        # We are not in sink's pod
        if not same_sink_pod:
            # Iterate choice for edges and aggr uplinks
            for eindex in edge_to_aggr.keys():
                # Get edge router
                er = self.get_router_from_position(type='edge', index=eindex, pod=src_pod)

                # Add edge->aggr uplinks
                for aggr_index in edge_to_aggr[eindex]:
                    # Get chosen aggr
                    aggr = self.get_router_from_position(type='aggregation', index=aggr_index, pod=src_pod)

                    # Append edge
                    chosen_edges.append((er, aggr))

            # Make also the aggr->core ulinks
            for aindex in aggr_to_core.keys():
                # Get aggregation router
                ar = self.get_router_from_position(type='aggregation', index=aindex, pod=src_pod)
                for core_index in aggr_to_core[aindex]:
                    # Get the core
                    core = self.get_router_from_position(type='core', index=core_index)
                    # Make uplink
                    chosen_edges.append((ar, core))

        # We are in sink's pod
        else:
            # Iterate choice for edges and aggr uplinks
            for eindex in edge_to_aggr.keys():
                if eindex == self.get_sink_index():
                    continue
                else:
                    # Get edge router
                    er = self.get_router_from_position(type='edge', index=eindex, pod=src_pod)

                    # Add edge->aggr uplinks
                    for aggr_index in edge_to_aggr[eindex]:
                        # Get chosen aggr
                        aggr = self.get_router_from_position(type='aggregation', index=aggr_index, pod=src_pod)
                        # Make uplink
                        chosen_edges.append((er, aggr))

        # Return the chosen edges
        return chosen_edges

    #@time_func
    def all_random_uplinks_iter(self, src_pod, exclude_edge_indexes=[]):
        already_seen = []

        # Generate random uplink choice from edge
        choices_list = all_possible_uplink_choices_gen(k=self.k,
                                                       exclude_edge_indexes=exclude_edge_indexes)

        same_sink_pod = False
        if src_pod == self.get_sink_pod():
            same_sink_pod = True

        for choice in choices_list:
            # Accumulate result here
            chosen_edges = []

            # Create dicts
            edge_to_aggr = {i: list(c) for i, c in enumerate(choice[:self.k/2]) if c}
            aggr_to_core = {i: list(c) for i, c in enumerate(choice[self.k/2:]) if c}

            # Compute the aggr routers that have been chosen at least by one edge router
            chosen_aggs_indexes = {a for e, alist in edge_to_aggr.iteritems() for a in alist}

            # Compute choice identifier
            ifier = set()
            for i, c in enumerate(choice):
                # We are on edges
                if i < self.k/2:
                    if c:
                        ifier.add(('e', i, c))
                # We are on aggr
                else:
                    j = i - self.k/2
                    if j in chosen_aggs_indexes:
                        ifier.add(('a', j, c))

            # Check if we've seen it already
            seen = any([True for o_ifier in already_seen if o_ifier.difference(ifier) == set()])
            if seen:
                # Go to next choice!
                continue
            else:
                # Append new ifier
                already_seen.append(ifier)

                # We are not in sink's pod
                if not same_sink_pod:
                    # Iterate choice for edges and aggr uplinks
                    for eindex in edge_to_aggr.keys():
                        # Get edge router
                        er = self.get_router_from_position(type='edge', index=eindex, pod=src_pod)

                        # Add edge->aggr uplinks
                        for aggr_index in edge_to_aggr[eindex]:
                            # Get chosen aggr
                            aggr = self.get_router_from_position(type='aggregation', index=aggr_index, pod=src_pod)

                            # Append edge
                            chosen_edges.append((er, aggr))

                    # Make also the aggr->core ulinks
                    for aindex in chosen_aggs_indexes:
                        ar = self.get_router_from_position(type='aggregation', index=aindex, pod=src_pod)
                        for core_index in aggr_to_core[aindex]:
                            # Get the core
                            core = self.get_router_from_position(type='core', index=core_index)
                            # Make uplink
                            chosen_edges.append((ar, core))

                # We are in sink's pod
                else:
                    # Iterate choice for edges and aggr uplinks
                    for index in edge_to_aggr.keys():
                        if index == self.get_sink_index():
                            continue
                        else:
                            # Get edge router
                            er = self.get_router_from_position(type='edge', index=index, pod=src_pod)

                            # Add edge->aggr uplinks
                            for aggr_index in edge_to_aggr[index]:
                                # Get chosen aggr
                                aggr = self.get_router_from_position(type='aggregation', index=aggr_index, pod=src_pod)
                                # Make uplink
                                chosen_edges.append((er, aggr))

                # Return the chosen edges
                yield chosen_edges

    def set_ecmp_uplinks_from_pod(self, src_pod):
        """
        Sets the original subdag from source pod to destination
        :param source:
        :return:
        """
        same_sink_pod = False
        if src_pod == self.get_sink_pod():
            same_sink_pod = True

        sink_id = self.get_sink_id()

        # Iterate pod routers
        for e in range(0, self.k/2):
            er = self.get_router_from_position(type='edge', index=e, pod=src_pod)
            if er == sink_id:
                continue

            for a in range(0, self.k/2):
                ar = self.get_router_from_position(type='aggregation', index=a, pod=src_pod)
                if not self.has_edge(er, ar):
                    self.add_uplink(er, ar)

            if not same_sink_pod:
                ar = self.get_router_from_position(type='aggregation', index=e, pod=src_pod)
                for c in range((self.k / 2) * e, (e + 1) * (self.k / 2)):
                    cr = self.get_router_from_position(type='core', index=c)
                    if not self.has_edge(ar, cr):
                        self.add_uplink(ar, cr)

    def set_ecmp_uplinks_from_source(self, src, previous_path, all_layers=True):
        """Given a source edge, and the previous path taken by a flow, it returns the
        dag to its original ecmp state.

        all_layers determines if only the edge->aggregation layer is reset to ecmp, or if
        also the aggregation->core layer.
        """
        if self.is_edge(src):
            src_pod = self.get_router_pod(src)
            src_index = self.get_router_index(src)

            same_sink_pod = False
            if src_pod == self.get_sink_pod():
                same_sink_pod = True

            for a in range(0, self.k / 2):
                ar = self.get_router_from_position(type='aggregation', index=a, pod=src_pod)
                if not self.has_edge(src, ar):
                    self.add_uplink(src, ar)

            if all_layers == True:
                previous_agg = previous_path[1]
                previous_agg_index = self.get_router_index(previous_agg)
                e = previous_agg_index
                if not same_sink_pod:
                    ar = self.get_router_from_position(type='aggregation', index=e, pod=src_pod)
                    for c in range((self.k / 2) *e, (e + 1) * (self.k / 2)):
                        cr = self.get_router_from_position(type='core', index=c)
                        if not self.has_edge(ar, cr):
                            self.add_uplink(ar, cr)

    #@time_func
    def _get_random_uplink_choice(self):
        """
        Get random uplink DAG for a whole pod.
        :return:
        """
        edges_to_aggr = {}
        aggr_to_core = {}

        # Iterate edge/aggr indexes - make a choice for each of them
        for index in range(0, self.k/2):
            # Draw how many aggregations each edge is using
            n_agg = random.randint(1, self.k / 2)
            n_core = random.randint(1, self.k / 2)

            # Draw which aggregation ones
            aggr_indexes = range(0, self.k/2)
            random.shuffle(aggr_indexes)
            edges_to_aggr[index] = aggr_indexes[:n_agg]

            # Draw which core ones
            core_indexes = range((self.k / 2) * index, (index + 1) * (self.k / 2))
            random.shuffle(core_indexes)
            aggr_to_core[index] = core_indexes[:n_core]

        return (edges_to_aggr, aggr_to_core)

    #@time_func
    def _get_random_uplink_choice_2(self, exclude_edge_indexes=[]):
        """"""
        # Initialize structure
        edges_to_aggr = {i: [] for i in range(0, self.k / 2) if i not in exclude_edge_indexes}

        # Number of edge routers chosen
        n_edges_chosen = self.k/2 - len(exclude_edge_indexes)

        # Draw the number of edge->aggr links
        n_etoa_links = random.randint(n_edges_chosen, n_edges_chosen*(self.k/2))

        # Compute all possible pairs of these edges to the aggr indexes
        all_pairs_etoa = it.product(edges_to_aggr.keys(), range(self.k/2))
        all_pairs_etoa = set([p for p in all_pairs_etoa])

        # Accumulate the indexes of the aggregation routers that are chosen
        chosen_agg_indexes = set()

        # Take one random first for each aggreagtion/edge router
        for er in edges_to_aggr.keys():
            # Take one aggregation and core at random
            ar = random.randint(0, self.k/2 - 1)
            new_edge = (er, ar)
            edges_to_aggr[er].append(ar)
            chosen_agg_indexes.add(ar)
            all_pairs_etoa = all_pairs_etoa - {new_edge}

        # Compute how many edge->aggr links are left
        n_remaining = n_etoa_links - n_edges_chosen

        # If there are edges to draw yet
        if n_remaining:
            # Shuffle the remaining pairs
            all_pairs_etoa = list(all_pairs_etoa)
            random.shuffle(all_pairs_etoa)
            rem_edges = all_pairs_etoa[:n_remaining]
            for (er, ar) in rem_edges:
                edges_to_aggr[er].append(ar)
                chosen_agg_indexes.add(ar)

        # Let's now do the aggr->core edges

        # Initialize structure
        aggr_to_core = {i: [] for i in chosen_agg_indexes}

        # Maximum of aggr->core links we can have
        max_atoc_links = len(chosen_agg_indexes)*self.k/2
        min_atoc_links = len(chosen_agg_indexes)

        # Draw the number of aggr->core links
        n_atoc_links = random.randint(min_atoc_links, max_atoc_links)

        # Create the pairs
        all_pairs_atoc = set()
        for i in chosen_agg_indexes:
            pairs = [all_pairs_atoc.add(p) for p in it.product([i], range((self.k / 2) * i, (i + 1) * (self.k / 2))) if i in chosen_agg_indexes]

        # Do the same as before!
        for ar in chosen_agg_indexes:
            # Take one core at random
            cr = random.randint((self.k / 2)*ar, (ar + 1)*(self.k / 2) - 1)
            new_edge = (ar, cr)
            aggr_to_core[ar].append(cr)
            all_pairs_atoc = all_pairs_atoc - {new_edge}
            n_atoc_links -= 1

        n_remaining = n_atoc_links
        if n_remaining:
            all_pairs_atoc = list(all_pairs_atoc)
            random.shuffle(all_pairs_atoc)
            rem_edges = all_pairs_atoc[:n_remaining]
            for (e, a) in rem_edges:
                aggr_to_core[e].append(a)

        return (edges_to_aggr, aggr_to_core)

    def all_paths_to_sink(self, source):
        """Returns all paths from source to sink in the DAG"""
        return nx.all_simple_paths(self, source, self.sink['id'])

    def apply_path_to_core(self, source, core):
        """Force the patch from source to core router, by removing side edges
         that woudl create ECMP"""

        if source != self.get_sink_id():
            # Get source pod
            src_pod = self.get_router_pod(source)

            # Check if we are in the same pod as sink
            same_sink_pod = False
            if src_pod == self.get_sink_pod():
                same_sink_pod = True

            # Get core index
            core_index = self.get_router_index(core)

            # Get the aggregation index valid with core index
            agg_index = [index for index in range(self.k/2) if self._valid_aggregation_core_indexes(index, core_index)]
            if len(agg_index) == 1: agg_index = agg_index[0]
            else: raise ValueError("Something wrong in the topology")

            # Get aggregation router
            ar = self.get_router_from_position(type='aggregation', index=agg_index, pod=src_pod)

            #Add source -> ar edge
            if not self.has_edge(source, ar): self.add_uplink(source, ar)

            # Remove side edges
            removal = [self.remove_edge(source, a) for a in self.successors(source) if a != ar]

            if not same_sink_pod:
                # Add ar->cr and remove others
                if not self.has_edge(ar, core): self.add_uplink(ar, core)
                removal = [self.remove_edge(ar, c) for c in self.successors(ar) if c != core]
        else:
            import ipdb; ipdb.set_trace()
            raise ValueError("Sink can't send flows to himself!")

    def get_plot_positions(self):
        positions = {}

        edgeStartPos = (1, 1)
        aggStartPos = (1, 2)

        # allocate edge routers
        for pod in range(self.k):
            for sub_pod in range(self.k / 2):
                # Get router first
                router = self.get_router_from_position('edge', index=sub_pod, pod=pod)
                positions[router] = edgeStartPos
                edgeStartPos = (edgeStartPos[0] + 3, edgeStartPos[1])
            edgeStartPos = (edgeStartPos[0] + 2.5, edgeStartPos[1])

        # allocate aggregation routers
        for pod in range(self.k):
            for sub_pod in range(self.k / 2):
                router = self.get_router_from_position('aggregation', index=sub_pod, pod=pod)
                positions[router] = aggStartPos
                aggStartPos = (aggStartPos[0] + 3, aggStartPos[1])
            aggStartPos = (aggStartPos[0] + 2.5, aggStartPos[1])

        totalDistance = positions[self.get_router_from_position('edge', self.k/2 - 1, self.k - 1)][0] - 1
        step = totalDistance / float(((self.k / 2) ** 2))

        coreStartPos = (1 + step / 2, 3.5)

        # Allocate core routers
        for pod in range((self.k / 2) ** 2):
            router = self.get_router_from_position('core', index=pod)
            positions[router] = (coreStartPos[0], coreStartPos[1])
            coreStartPos = (coreStartPos[0] + step, coreStartPos[1])

        # Compute sink position
        dst = self.dst_prefix
        gw = self.get_sink_id()
        gwPos = positions[gw]
        positions[dst] = (gwPos[0], gwPos[1] - 1)

        return positions

    def plot(self, plotname=None):
        """Plots himself!"""
        routers = [n for n in self.nodes_iter()]

        # Draw nodes and edges
        nx.draw_networkx_nodes(self, ax=None, nodelist=[self.dst_prefix], pos=self.plot_positions, node_shape='o',
                               node_color='r')
        nx.draw_networkx_nodes(self, ax=None, nodelist=routers, pos=self.plot_positions, node_shape='s',
                               node_color='b')

        # Draw edges of current dag
        nx.draw_networkx_edges(self, ax=None, width=1.5, pos=self.plot_positions)

        # Compute edges that are not currently used
        all_edges = set(self.all_edges)
        current_edges = set(self.edges())
        not_used = list(all_edges - current_edges)

        # Generate new graph copy of self, but with non-used edges
        g = DiGraph()
        g.add_nodes_from(self.nodes())
        g.add_edge(self.sink_id, self.dst_prefix)

        # Draw first sink-> dst edge
        nx.draw_networkx_edges(g, ax=None, width=1.5, pos=self.plot_positions)

        g.remove_edge(self.sink_id, self.dst_prefix)
        g.add_edges_from(not_used)

        # Draw edges of current dag
        nx.draw_networkx_edges(g, ax=None, width=1.0, arrows=False, edge_color='grey', style='dashdot', pos=self.plot_positions)

        if plotname:
            plt.savefig(plotname)
        else:
            plt.show()

        plt.gcf().clear()

# Auxiliary funcitions

#@time_func
def all_possible_uplink_choices_gen(k, exclude_edge_indexes=[]):
    # Accumulate choices here
    edges_choice_lists = []

    # For each edge router
    for edge_index in range(k/2):
        if edge_index not in exclude_edge_indexes:
            # Iterate range differently at every loop
            edges_choice_lists.append([c for n_edges in range(1, k / 2 + 1) for c in it.combinations(range(k/2), n_edges)])
        else:
            edges_choice_lists.append([(),])

    for aggr_index in range(k/2):
        edges_choice_lists.append([c for n_edges in range(1, k/2 + 1) for c in it.combinations(range((k / 2) * aggr_index, (aggr_index + 1) * (k / 2)), n_edges)])

    return it.product(*edges_choice_lists)

if __name__ == "__main__":
    #log.setLevel(logging.DEBUG)
    choices = all_possible_uplink_choices_gen(4, exclude_edge_indexes=[1])
    dcGraph = DCGraph(k=4, prefix_type='primary')
    prefixes = dcGraph.destination_prefixes()
    prefix = prefixes[random.randint(3, 7)]
    dcDag = dcGraph.get_default_ospf_dag(prefix=prefix)
    #edges = dcDag.all_random_uplinks_iter(src_pod=0, exclude_edge_indexes=[1])
    #edges = list(edges)

    edges = dcDag._get_random_uplink_choice_2(exclude_edge_indexes=[])
    import ipdb; ipdb.set_trace()

    dcDag.plot()


    #source = dcDag.get_destination_prefix_gateway(prefix)
    #srcPod = dcDag.get_router_pod(source)
    #for i in range(20000):
    #    dcDag.modify_random_uplinks(src_pod=srcPod)
    #    if not dcDag.is_valid_dc_dag():
    #        import ipdb; ipdb.set_trace()

