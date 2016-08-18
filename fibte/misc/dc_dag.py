from networkx import DiGraph

"""
Module that defines the DCGraph object, which extends the nx.DiGraph class
to support a DataCenter-like DAG structure.
"""

class DCDag(DiGraph):
    def __init__(self, sink_id, sink_name, sink_index, sink_pod, k=4, *args, **kwargs):
        super(DCDag, self).__init__(*args, **kwargs)

        # Set DCDag's sink
        self._set_sink(sink_id, sink_name, sink_index, sink_pod)

        # Set DC's parameter
        self.k = k

    def _get_printable(self):
        """
        :return: the same dag with the nodes replaced by router names
        """
        pass

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
            pass
        else:
            print 'Nodes are missing: {0} nodes'.format(len(self.nodes()))
            return False

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

    def is_sink(self, routerid):
        return routerid == self.sink['id']

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

    def get_sink_id(self):
        return self.sink['id']

    def get_sink_index(self):
        return self.sink['index']

    def get_sink_name(self):
        return self.sink['name']

    def get_sink_pod(self):
        return self.sink['pod']

    def have_same_pod(self, a, b):
        """
        Returns true only if a and b are from the same pod
        """
        if not self.is_core(a) and not self.is_core(b):
            return self.get_router_pod(a) == self.get_router_pod(b)
        else:
            raise ValueError("Either a:{0} or b:{1} are core routers".format(a, b))

    def add_uplink(self, lower_rid, upper_rid):
        """
        Adds a DC uplink by checking the corresponding constrains
        """
        if not self.is_sink(lower_rid):
            if (self.is_edge(lower_rid) and self.is_aggregation(upper_rid)) or (self.is_aggregation(lower_rid) and self.is_core(upper_rid)):
                # edge->aggregation uplink
                if self.is_edge(lower_rid):
                    if self.get_router_pod(lower_rid) == self.get_router_pod(upper_rid):
                        # Add uplink
                        self.add_edge(lower_rid, upper_rid, {'direction': 'uplink'})
                    else:
                        raise ValueError("Edge {0} and aggregation {1} routers are not from the same pod".format(lower_rid, upper_rid))

                # aggregation->core uplink
                else:
                    # Prevent uplink in aggregation layers in same pod as sink
                    apod = self.get_router_pod(lower_rid)
                    if apod != self.get_sink_pod():
                        # Get index of aggregation and core router
                        ai = self.get_router_index(lower_rid)
                        ri = self.get_router_index(upper_rid)
                        # Check that uplink is valid wrt the DC structure
                        if (ri >= ai*(self.k/2) and ri <= ((ai+1)*(self.k/2))-1):
                            # Add uplink
                            self.add_edge(lower_rid, upper_rid, {'direction': 'uplink'})
                        else:
                            raise ValueError("Invalid uplink. Wrong router index match for uplink ({0} -> {1})".format(lower_rid, upper_rid))
                    else:
                        raise ValueError("Invalid uplink. Aggregation routers in same pod as sink can't have uplinks: {0}".format(lower_rid))
            else:
                raise ValueError("Wrong router index match for an uplink ({0} -> {1})".format(lower_rid, upper_rid))
        else:
            raise ValueError("The sink {0} can't have an uplink".format(lower_rid))

    def add_donwlink(self, upper_rid, lower_rid):
        """
        Adds a DC downlink by checking the corresponding constrains
        """
        if (self.is_aggregation(upper_rid) and self.is_edge(lower_rid)) or (self.is_aggregation(lower_rid) and self.is_core(upper_rid)):
            # aggregation->edge downlink
            if self.is_edge(lower_rid):
                # If they are from the same pod
                lpod = self.get_router_pod(lower_rid)
                rpod = self.get_router_pod(upper_rid)
                if lpod == rpod:
                    # Check that downlink is only in same pod as sink:
                    if lpod == self.get_sink_pod():
                        # Add downlink
                        self.add_edge(upper_rid, lower_rid, {'direction': 'downlink'})
                    else:
                        raise ValueError("Invalid donwlink. Aggregation -> Edge dowlink can only happen in sink's pod")
                else:
                    raise ValueError("Edge {0} and aggregation {1} routers are not from the same pod".format(lower_rid, upper_rid))

            # core->aggregation downlink
            else:
                # Get index of aggregation and core router
                ri = self.get_router_index(upper_rid)
                ai = self.get_router_index(lower_rid)

                # Check that uplink is valid wrt the DC structure
                if (ri >= ai*(self.k/2) and ri <= ((ai+1)*(self.k/2))-1):
                    # Check that downlink is only directed to sink's pod
                    if self.get_router_pod(ai) == self.get_sink_pod():
                        # Add downlink
                        self.add_edge(upper_rid, lower_rid, {'direction': 'downlink'})
                    else:
                        raise ValueError("Invalid downlink. Cores can only have downlinks towards sink's pod")
                else:
                    raise ValueError("Invalid downlink. Wrong router index match for downlink ({0} -> {1})".format(upper_rid, lower_rid))
        else:
            raise ValueError("Wrong router index match for an downlinklink ({0} -> {1})".format(lower_rid, upper_rid))

    def _valid_edge_index(self, edge_index):
        return self._valid_aggregation_index(edge_index)

    def _valid_aggregation_index(self, agg_index):
        return agg_index >= 0 and agg_index <= ((self.k/2) - 1)

    def _valid_core_index(self, core_index):
        return core_index >= 0 and core_index <= (((self.k/2)**2)-1)

    def _valid_aggregation_core_indexes(self, agg_index, core_index):
        if self._valid_aggregation_index(agg_index) and self._valid_core_index(core_index):
            return core_index >= agg_index*(self.k/2) and core_index <= (((agg_index+1)*(self.k/2))-1)
        else:
            return False

    def get_upper_tier(self, routerid):
        """
        Returns a list of the immediate upper layer routers
        :param routerid:
        :return:
        """
        if not self.is_core(routerid):
            rpos = self.get_router_position(routerid)
            if self.is_edge(routerid):
                return [ag for ag in self.nodes_iter() if self.have_same_pod(ag, routerid) and self.is_aggregation(ag)]

            # Is aggregation router
            else:
                # Check for agg->core link validity
                return [r for r in self.nodes_iter()
                        if self.is_core(r) and self._valid_aggregation_core_indexes(rpos['index'],
                                                                                    self.get_router_index(r))]
        else:
            raise ValueError("Core routers do not have upper layer: {0}".format(routerid))

    def get_lower_tier(self, routerid):
        """
        Retunrs a list of the immediate lower layer routers
        :param routerid:
        :return:
        """
        if not self.is_edge(routerid):
            rpos = self.get_router_position(routerid)
            if self.is_core(routerid):
                # Check for core->link link validity
                return [r for r in self.nodes_iter() if self.is_aggregation(r)
                        and self.get_router_pod(r) == self.get_sink_pod()
                        and self._valid_aggregation_core_indexes(rpos['index'], self.get_router_index(r))]

            # Aggregation router
            else:
                return [r for r in self.nodes_iter() if self.have_same_pod(r, routerid) and self.is_edge(r)]
        else:
            raise ValueError("Edge routers do not have lower layer: {0}".format(routerid))
