from networkx import DiGraph
from fibte.misc.topology_graph import TopologyGraph
from fibte import CFG
import os
import random
import networkx as nx

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
    def __init__(self, k=4, *args, **kwargs):
        super(DCDiGraph, self).__init__(*args, **kwargs)
        # DC parameter
        self.k = k

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

    def is_edge(self, routerid):
        return self.node[routerid]['type'] == 'edge'

    def is_aggregation(self, routerid):
        return self.node[routerid]['type'] == 'aggregation'

    def is_core(self, routerid):
        return self.node[routerid]['type'] == 'core'

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

    def _get_printable(self):
        """
        :return: the same dag with the nodes replaced by router names
        """
        pass

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

    def add_downlink(self, upper_rid, lower_rid):
        """
        Adds a DC downlink by checking the corresponding constrains
        Assumes lower_rid and upper_rid already in the graph.

        If uplink is not valid, the function silently finishes
        """
        if self.is_valid_downlink(upper_rid, lower_rid):
            # Add downlink
            self.add_edge(upper_rid, lower_rid, {'direction': 'downlink'})

class DCGraph(DCDiGraph):
    """
    Models the DC graph as a networkx DiGraph subclass object

    There should be only once instance of DCGraph in the controller.
    It builds itself thanks to the topologyDB object, at startup.
    """
    def __init__(self, k=4, *args, **kwargs):
        super(DCGraph, self).__init__(k, *args, **kwargs)

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
        aggrRouters = self.topo.getAgreggationRouters()
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

    def get_default_ospf_dag(self, sinkid):
        """
        Given a sink edge router, returns the default fat-tree dag from
        all other hosts to the sink.

        :param sink: router id of the sink's router
        :return: DCDAg
        """
        if self.is_edge(sinkid):
            # Create empty-links dag from self DCGraph
            dc_dag = DCDag(sinkid, k=self.k, dc_graph=self)

            # Add uplinks
            add_uplinks1 = [dc_dag.add_uplink(er, ar) for er in self.edge_routers_iter() for ar in self.get_upper_tier(er)]
            add_uplinks2 = [dc_dag.add_uplink(ar, cr) for ar in self.aggregation_routers_iter() for cr in self.get_upper_tier(ar)]

            # Add downlinks
            add_downlinks1 = [dc_dag.add_downlink(cr, ar) for cr in self.core_routers_iter() for ar in self.get_lower_tier(cr) if self.is_valid_downlink(cr, ar)]
            add_downlinks2 = [dc_dag.add_downlink(ar, er) for ar in self.aggregation_routers_iter() for er in self.get_lower_tier(ar) if self.is_valid_downlink(ar, er)]

            # Return the DCDag object
            return dc_dag
        else:
            raise ValueError("Sink must be an edge router. {0} isn't".format(sinkid))
    # old getRandomDag code

    # def getRandomDag(self, prefix):
    #     """
    #     Returns a random dag (upwards) from every
    #     other prefix to specified target prefix.
    #
    #     :param prefix:
    #     :return:
    #     """
    #     # Result is stored here
    #     rdag = nx.DiGraph()
    #
    #     # Get first the gateway edge router of the prefix
    #     gwRouter = self.getGatewayRouter(prefix)
    #
    #     # Get pod number
    #     prefixPod = self.getRouterPod(gwRouter)
    #
    #     # Then starting at every other edge
    #     for router in self.getEdgeRouters():
    #
    #         # Get pod of the router
    #         rPod = self.getRouterPod(router)
    #
    #         # If they are in different pods
    #         if rPod != prefixPod:
    #
    #             # Make uplink choices
    #             choices = self.getRandomUpLinkChoice(max_depth=2)
    #
    #             # Add corresponding edges
    #             for choice in choices:
    #                 # Fetch aggregation router chosen
    #                 aggChosen = self.getRouterFromIndex(type='aggregation', index=choice[0], pod=rPod)
    #
    #                 # Add the edge from edge router to randomly chosen aggregation router
    #                 rdag.add_edge(router, aggChosen)
    #
    #                 # Traverse now the core router choices from each aggregation
    #                 for subchoice in choice[1]:
    #                     # Fetch core router
    #                     coreRouter = self.getRouterFromIndex(type='core', index=subchoice[0])
    #
    #                     # Add edge aggregation -> core
    #                     rdag.add_edge(aggChosen, coreRouter)
    #
    #                     # Add also here the downlink edge core -> aggregation
    #                     corrAggr = self.getConnectedAggregation(coreRouter=coreRouter, pod=prefixPod)
    #                     rdag.add_edge(coreRouter, corrAggr)
    #
    #         # Edge router in the same pod
    #         else:
    #             # Check that it's not himself
    #             if router != gwRouter:
    #                 # Draw also the used links (inside pod)
    #                 choices = self.getRandomUpLinkChoice(max_depth=1)
    #
    #                 # Add corresponding edges
    #                 for choice in choices:
    #                     # Fetch aggregation router chosen
    #                     aggChosen = self.getRouterFromIndex(type='aggregation', index=choice[0], pod=rPod)
    #
    #                     # Add the edge from edge router to randomly chosen aggregation router
    #                     rdag.add_edge(router, aggChosen)
    #
    #                     # Add also the downlink fixed link: aggregation -> edge router
    #                     rdag.add_edge(aggChosen, gwRouter)
    #
    #     return rdag

class DCDag(DCDiGraph):
    """
    This class represents a Fat-Tree like Direct Acyclic Graph for a
    certain destination, which must be the single sink in the graph.
    """
    def __init__(self, sink_id, k=4, *args, **kwargs):
        super(DCDag, self).__init__(k, *args, **kwargs)

        # Set DCDag's sink
        self.sink = sink_id

        # Parse kwargs
        if 'dc_graph' in kwargs and isinstance(kwargs['dc_graph'], DCGraph):
            # If dc_graph is given, add all routers of dc_graph
            self._build_from_dc_graph(kwargs['dc_graph'])

    def _get_printable(self):
        """
        Helper function that returns a DCDag with router names instead.
        :return:
        """
        other = DCDag(self.sink, self.k)
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

        # Set sink data
        sink_name = self.get_router_name(self.sink)
        sink_pos = self.get_router_position(self.sink)
        self._set_sink(self.sink, sink_name, sink_pos['index'], sink_pos['pod'])

    def _set_sink(self, routerid, routername, index, pod):
        """
        Sets the sink of the DAG to the corresponding edge router
        :param routerid:
        :param routername:
        :param index:
        :param pod:
        """
        self.sink = {'id': routerid, 'name': routername, 'index': index, 'pod': pod}
        self.add_edge_router(routerid, routername, index, pod)

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

    def modify_random_uplinks(self, source):
        """
        Given a source, choses a random set of upwards paths
        towards the sink, and modifies the DAG accordingly.
        """
        # Get source pod
        src_pod = self.get_router_pod(source)

        # Generate random uplink choice from edge
        random_choice = self._get_random_uplink_choice()

        same_sink_pod = False
        if src_pod == self.get_sink_pod():
            same_sink_pod = True

        # Here we accumulate chosen edges
        chosen_edge_to_aggr = []
        chosen_aggr_to_core = []

        # Iterate choice
        for aggr_index, core_indexes in random_choice.iteritems():

            # Check if uplink exists
            ar = self.get_router_from_position(type='aggregation', index=aggr_index, pod=src_pod)

            # Add uplinks
            self.add_uplink(source, ar)

            # Add it to chosen links
            chosen_edge_to_aggr.append((source, ar))

            if not same_sink_pod:
                # Get core routers from index
                crs = [self.get_router_from_position(type='core', index=ci) for ci in core_indexes]
                add_crs = [self.add_uplink(ar, cr) for cr in crs]

                # Append them to chosen edges
                add_crs = [chosen_aggr_to_core.append((ar, cr)) for cr in crs]

        # Check for uplinks that should not be there and remove them
        remove1 = [self.remove_edge(source, arx) for arx in self.successors(source) if (source, arx) not in chosen_edge_to_aggr]
        if not same_sink_pod:
            # Check also for the core routers
            remove2 = [self.remove_edge(arx, crx) for arx in self.successors(source) for crx in self.successors(arx) if (arx, crx) not in chosen_aggr_to_core]

    def set_ecmp_uplinks(self, source):
        """
        Sets the original subdag from source to destination
        :param source:
        :return:
        """
        # Get source pod
        src_pod = self.get_router_pod(source)

        same_sink_pod = False
        if src_pod == self.get_sink_pod():
            same_sink_pod = True

        # Add edge->aggregation default edges
        action = [self.add_uplink(source, ar) for ar in self.successors(source)]

        # If not in same pod as sink
        if not same_sink_pod:
            # Add aggr->core too
            action = [self.add_uplink(ar, cr) for ar in self.successors(source) for cr in self.successors(ar)]

    def _get_random_uplink_choice(self):
        """
        Assume you are in an edge node
        :return:
        """
        random_choice = {}

        # Draw how many aggregations from same pod
        n_agg = random.randint(1, self.k/2)

        # Draw which ones
        indexes = range(0, self.k/2)
        random.shuffle(indexes)
        chosen = indexes[:n_agg]

        # Append aggregation choices
        append_aggregation_choices = [random_choice.setdefault(c, []) for c in chosen]
        for i in chosen:
            # Compute valid core_indexes
            core_indexes = range((self.k/2)*i, (i+1)*(self.k/2))

            # Compute how many
            n_cr = random.randint(1, self.k/2)

            random.shuffle(core_indexes)

            # Choose which of them
            random_choice[i] = core_indexes[:n_cr]

        return random_choice

    def all_paths_to_sink(self, source):
        #TODO
        pass

if __name__ == "__main__":

    dcGraph = DCGraph(k=4)
    edgeRouters = dcGraph.edge_routers()
    sink = edgeRouters[random.randint(3, 7)]
    dcDag = dcGraph.get_default_ospf_dag(sinkid=sink)
    source = edgeRouters[0]
    for i in range(20000):
        dcDag.modify_random_uplinks(source=source)
        if not dcDag.is_valid_dc_dag():
            import ipdb; ipdb.set_trace()

