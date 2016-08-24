#!/usr/bin/python

from fibbingnode.misc.mininetlib.ipnet import TopologyDB
import os
import json
import networkx as nx
from ipaddress import ip_interface
import subprocess
import copy
import matplotlib

#matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
#from lb.logger import log

import logging
import time

from fibte import CFG

tmp_files = CFG.get("DEFAULT", "tmp_files")
db_topo = CFG.get("DEFAULT", "db_topo")

class NetworkGraph(object):
    """
    This object handles the interactive plotting features
    mainly.

    """
    def __init__(self, topologyDB):
        # TopoDB object
        self.topologyDB = topologyDB

        # Loads the networkx graph from the topoDB object
        self.graph = self.loadGraphFromDB(self.topologyDB)

    def loadGraphFromDB(self, topologyDB):
        g = nx.Graph()
        for node in topologyDB.network:
            if node not in g.nodes() and (node not in ["default_controller", "sw-mon", "c1"]):
                g.add_node(node)
                g.node[node]['type'] = topologyDB.type(node)

                # TODO IMPORTANT, here we will differenciate a type of routers. The ones that contain the letter e, will be
                # classified as edge routers. Edge routers are used to compute all the possible paths within the
                # topology
                if '_e' in node:
                    g.node[node]['edge'] = True

                elif '_a' in node:
                    g.node[node]['aggregation'] = True

                elif 'r_c' in node:
                    g.node[node]['core'] = True

                for itf in topologyDB.network[node]:
                    # we should ignore routerid, type.
                    # TODO have to find a better way to do this
                    if itf in ["routerid", 'type', 'gateway']:
                        continue

                    # else itf its a real interface so we add an edge,
                    # only if the connected node has been created.
                    connectedTo = topologyDB._interface(node, itf)["connectedTo"]
                    if connectedTo in g.nodes():
                        # add edge
                        g.add_edge(node, connectedTo)
        return g

    def keepOnlyRouters(self):

        to_keep = [x for x in self.graph.node if self.graph.node[x]['type'] == 'router']

        return self.graph.subgraph(to_keep)

    def keepRoutersAndNormalSwitches(self):
        to_keep = [x for x in self.graph.node if self.graph.node[x]['type'] == 'router'] + self.getNormalSwitches()

        return self.graph.subgraph(to_keep)

    def getNormalSwitches(self):
        return [x for x in self.graph.node if "sw" in x]

    def getOVSSwitches(self):
        return [x for x in self.graph.node if "ovs" in x]

    def setNodeShape(self, node, shape):
        self.graph.node[node]['node_shape'] = shape

    def setNodeColor(self, node, color):
        self.graph.node[node]['node_color'] = color

    def setNodeTypeShape(self, type, shape):
        for node in self.graph.node:
            if self.graph.node[node]['type'] == type:
                self.setNodeShape(node, shape)

    def setNodeTypeColor(self, type, color):
        for node in self.graph.node:
            if self.graph.node[node]['type'] == type:
                self.setNodeColor(node, color)

    def getFatTreePositions(self, k=4, normalSwitches=True):
        # assume that g is already the reduced graph
        # assume that we named the nodes using the fat tree "structure"
        # assume that we know k

        positions = {}

        normalSwitchStartPos = (1, 0)
        edgeStartPos = (1, 1)
        aggStartPos = (1, 2)

        normalSwitchBaseName = "sw_{0}_{1}"
        edgeBaseName = "r_{0}_e{1}"
        aggBaseName = "r_{0}_a{1}"
        coreBaseName = "r_c{0}"

        if normalSwitches:
            # allocate switches
            for pod in range(k):
                for sub_pod in range(k / 2):
                    positions[normalSwitchBaseName.format(pod, sub_pod)] = normalSwitchStartPos
                    normalSwitchStartPos = (normalSwitchStartPos[0] + 3, normalSwitchStartPos[1])
                normalSwitchStartPos = (normalSwitchStartPos[0] + 2.5, normalSwitchStartPos[1])

        # allocate edge routers
        for pod in range(k):
            for sub_pod in range(k / 2):
                positions[edgeBaseName.format(pod, sub_pod)] = edgeStartPos
                edgeStartPos = (edgeStartPos[0] + 3, edgeStartPos[1])
            edgeStartPos = (edgeStartPos[0] + 2.5, edgeStartPos[1])

        # allocate aggregation routers
        for pod in range(k):
            for sub_pod in range(k / 2):
                positions[aggBaseName.format(pod, sub_pod)] = aggStartPos
                aggStartPos = (aggStartPos[0] + 3, aggStartPos[1])
            aggStartPos = (aggStartPos[0] + 2.5, aggStartPos[1])

        totalDistance = positions[edgeBaseName.format(k - 1, (k / 2) - 1)][0] - 1
        print totalDistance
        step = totalDistance / float(((k / 2) ** 2))
        print step
        coreStartPos = (1 + step / 2, 3.5)

        # allocate core routers
        for pod in range((k / 2) ** 2):
            positions[coreBaseName.format(pod)] = (coreStartPos[0], coreStartPos[1])
            coreStartPos = (coreStartPos[0] + step, coreStartPos[1])

        print positions
        return positions

    def setEdgeWeights(self, link_loads={}):
        pass

    def plotGraphAnimated(self, k=4, queue=None):
        plt.ion()
        fig = plt.figure()
        g = self.keepOnlyRouters()  # .to_directed()
        pos = self.getFatTreePositions(k)
        nx.draw(g, arrows=False, width=1.5, pos=pos, node_shape='o', node_color='b')
        plt.show(block=False)
        while True:
            # read new link_loads
            link_loads = queue.get()
            queue.task_done()

            weights = {x: y for x, y in link_loads.items() if all("sw" not in e for e in x)}
            tt = time.time()
            nx.draw_networkx_edge_labels(g, pos, edge_labels=weights, label_pos=0.15, font_size=8, font_color="k",
                                         font_weight='bold')
            print time.time() - tt

            # plt.show()
            fig.canvas.update()
            fig.canvas.flush_events()
            # plt.pause(0.0001)
            print time.time() - tt

    def plotGraph(self, k=4):
        g = self.keepOnlyRouters()  # .to_directed()
        pos = self.getFatTreePositions(k)
        nx.draw(g, arrows=False, width=1.5, pos=pos, node_shape='o', node_color='b')
        plt.show()

    def getHosts(self):
        return [x for x in self.graph.node if self.graph.node[x]['type'] == 'host']

    def getEdgeRouters(self):
        return [x for x in self.graph.node if self.graph.node[x].has_key("edge")]

    def getAggregationRouters(self):
        return [x for x in self.graph.node if self.graph.node[x].has_key("aggregation")]

    def getCoreRouters(self):
        return [x for x in self.graph.node if self.graph.node[x].has_key("core")]

    def getRouters(self):
        return [x for x in self.graph.node if self.graph.node[x]["type"] == "router"]

    def getGatewayRouter(self, host):
        # So here we make the assumption that host: h_0_0 its always connected to ovs_0_0 switch.
        # Therefore, we will start from there. From ovs_x_y, two scenarios can happen. First, the switch is connected
        # to another switch named s_x_y, or is connected to an edge router.
        if not (self.graph.has_node(host)):
            # replace that for a debug...
            print "Host %s does not exist" % host
            return None

        x = host.split("_")[1]
        y = host.split("_")[2]
        ovs_switch = "ovs_%s_%s" % (x, y)

        ovs_adjacent_nodes = self.graph.adj[ovs_switch].keys()

        # now i try to get the router or the switch
        for node in ovs_adjacent_nodes:
            node_type = self.graph.node[node]['type']
            if node_type == "switch":

                # we are in the second switch
                sw_adjacent_nodes = self.graph.adj[node].keys()
                for node2 in sw_adjacent_nodes:
                    if self.graph.node[node2]['type'] == "router":
                        return node2

            elif node_type == "router":
                return node
        # gateway was not found
        return None

    def getPathsBetweenHosts(self, srcHost, dstHost):

        """
        compute the paths between two hosts
        :param srcHost:
        :param dstHost:
        :return:
        """

        # first we get the gateways
        srcEdge = self.getGatewayRouter(srcHost)
        dstEdge = self.getGatewayRouter(dstHost)

        paths = list(nx.all_shortest_paths(self.graph, srcEdge, dstEdge))
        paths = [tuple(x) for x in paths]

        return paths

    def getHostsBehindRouter(self, router):
        # Returns all the hosts that have router as a gateway.
        # !!! IMPORTANT: We make the assumption that the topology used here is a fat tree
        # and that the edge router can face a switch or ovs switches

        hosts = []
        if not (self.graph.has_node(router)):
            # replace that for a debug...
            print "Host %s does not exist" % router
            return None

        router_neighbors = self.graph.neighbors(router)

        # if the router is connected to a normal switch
        if any("sw" in x for x in router_neighbors):

            # we get swtich neighbors and in theory we should find ovs switches
            # we assume that there is only one switch touching that router
            for node in router_neighbors:
                if self.graph.node[node]['type'] == 'switch':
                    sw = node
            switch_neighbors = self.graph.neighbors(sw)

            for switch in switch_neighbors:
                # if its an ovs switch
                if self.graph.node[switch]['type'] == 'switch':
                    hosts.append(switch.replace("ovs", "h"))

        # we go over the ovs switches to find the hosts
        elif any("ovs" in x for x in router_neighbors):

            for switch in router_neighbors:
                # if its an ovs switch
                if self.graph.node[switch]['type'] == 'switch':
                    hosts.append(switch.replace("ovs", "h"))

        # the edge router has no switches next to it.
        else:
            return None

        return hosts

    def inSameSubnet(self, srcHost, dstHost):
        """
        Function to check if two hosts belong to the same subnetwork, in that case they do not use external paths
        so flows that do not live a the subnetwork can not be loadbalanced.

        To check if they belong to the same subnetwork ,since we do not store
        :param srcHost:
        :param dstHost:
        :return:
        """
        return self.getGatewayRouter(srcHost) == self.getGatewayRouter(dstHost)

    def getEdgesBetweenRouters(self):
        """
        Returns all edges between routers in the network
        :return:
        """
        allEdges = self.graph.to_directed().edges(nbunch=self.getRouters())

        # Filter the edges and remove the ones that have switches connected
        return [x for x in allEdges if not (any("sw" in s for s in x) or any("ovs" in s for s in x))]

    def getBisectionEdges(self):
        """
        Returns all edges between routers in the network
        :return:
        """
        allEdges = self.graph.to_directed().edges(nbunch=self.getRouters())

        # Filter the edges and remove the ones that have switches connected
        return [x for x in allEdges if not (any("sw" in s for s in x) or any("ovs" in s for s in x))]


class TopologyGraph(TopologyDB):
    def __init__(self, getIfindexes=True, interfaceToRouterName=False, *args,
                 **kwargs):

        super(TopologyGraph, self).__init__(*args, **kwargs)

        # Router interface name to interface index mappings
        self.getIfIndexes = getIfindexes

        # Holds the data of router interfaces
        self.routersInterfaces = {}

        # Retrieces the interface names
        self.getIfNames()

        # Starts the network graph
        self.networkGraph = NetworkGraph(self)

        # Populates self.hostsIpMapping dictionary
        self.hostsIpMapping()
        # Populataes self.routersIpMappin dictionary
        self.routersIdMapping()

        # Populates self.interfaceIPToRouterName dict
        if interfaceToRouterName:
            self.loadInterfaceToRouterName()

    def loadInterfaceToRouterName(self):
        """
        Used to find as fast as possible a router from one of its interface IP's
        """
        self.interfaceIPToRouterName = {}
        for name, interfaces in self.getRouters().iteritems():
            for interface, parameters in interfaces.iteritems():
                if isinstance(parameters, dict):
                    self.interfaceIPToRouterName[parameters["ip"].split("/")[0]] = name

    def getRouterFromInterfaceIp(self, ip):
        """
        Returns the name of the router given any of its interfaces' ip

        :param ip: any router interface ip
        :return: router name
        """
        return self.interfaceIPToRouterName[ip]

    def inSameSubnetwork(self, srcHost, dstHost):
        """
        Checks if src host and dst host belong to the same subnet.
        The function assumes that every host has only one interface
        :param srcHost:
        :param dstHost:
        :return: Returns a boolean
        """
        if srcHost not in self.network:
            raise ValueError("{0} does not exist".format(srcHost))

        if dstHost not in self.network:
            raise ValueError("{0} does not exist".format(dstHost))

        return self.networkGraph.inSameSubnet(srcHost, dstHost)

    def getPathsBetweenHosts(self, srcHost, dstHost):
        """
        Returns all the possible paths with same cost between two hosts
        :param srcHost:
        :param dstHost:
        :return:
        """
        if srcHost not in self.network:
            raise ValueError("{0} does not exist".format(srcHost))

        if dstHost not in self.network:
            raise ValueError("{0} does not exist".format(dstHost))

        return self.networkGraph.getPathsBetweenHosts(srcHost, dstHost)

    def getGatewayRouter(self, host):
        """
        Given a host it returns its gateway router
        :param host:
        :return:
        """
        if host not in self.network:
            raise ValueError("{0} does not exist".format(host))

        return self.network[host]["gateway"]

    def getHostsBehindRouter(self, router):
        """
        Returns a list with all the hosts that have router as a gateway
        :param router:
        :return:
        """

        if router not in self.network:
            raise ValueError("{0} does not exist".format(router))

        return self.networkGraph.getHostsBehindRouter(router)

    def getHostsBehindPod(self, pod):
        """
        Returns all the hosts under a given pod

        :param pod: a pod number is expected in string form "pod_X"
        :return: a list of hosts under that pod
        """
        # Get pod number
        pod_number = pod.split('_')[1]

        # Calculate corresponding edge router regex
        regex = "r_{0}_e".format(pod_number)

        # Get all edge routers
        edge_routers = self.getEdgeRouters()

        # Filter those that match the regex and get a set of hosts
        host_lists = [self.getHostsBehindRouter(edge_router) for edge_router in edge_routers if regex in edge_router]
        host_set = {host for host_list in host_lists for host in host_list}
        return list(host_set)

    def getHostsInOtherSubnetworks(self, host):
        """
        Returns all the hosts from other subnetworks. This is used to generate traffic only to "remote" hosts
        :param host:
        :return:
        """
        if host not in self.network:
            raise ValueError("{0} does not exist".format(host))

        # Returns all the hosts that are in other subnetworks
        gatewayEdgeRouter = self.getGatewayRouter(host)
        edgeRouters = self.getEdgeRouters()

        otherHosts = []
        for edgeRouter in edgeRouters:
            if edgeRouter == gatewayEdgeRouter:
                continue
            otherHosts += self.getHostsBehindRouter(edgeRouter)

        return otherHosts

    def numberDevicesInPath(self, src, dst):
        """
        Returns the number of devices between two nodes in the network
        :param src:
        :param dst:
        :return:
        """
        if src not in self.network:
            raise ValueError("{0} does not exist".format(src))

        if dst not in self.network:
            raise ValueError("{0} does not exist".format(dst))
        return len(nx.shortest_path(self.networkGraph.graph, src, dst))

    def getRouters(self):
        """
        Returns a dictionary with the routers from the topologyDB
        :return: router-only dict
        """
        return {node: self.network[node] for node in self.network if self.network[node]["type"] == "router"}

    def getEdgeRouters(self):
        """
        Only edge routers
        :return:
        """
        return self.networkGraph.getEdgeRouters()

    def getAgreggationRouters(self):
        """
        Only aggregation routers
        :return: list of aggregation routers
        """
        return self.networkGraph.getAggregationRouters()

    def getCoreRouters(self):
        """
        Only core routers
        :return: list of core routers
        """
        return self.networkGraph.getCoreRouters()

    def getHosts(self):
        """
        Gets the hosts from the topologyDB

        :return: host-only dictionary
        """
        return {node: self.network[node] for node in self.network if self.network[node]["type"] == "host"}

    def getSwitches(self):
        """
        Gets the switches from the topologyDB

        :return: switches-only dictionary
        """
        return {node: self.network[node] for node in self.network if self.network[node]["type"] == "switch"}

    def getEdgesUsageDictionary(self, element=0):

        if isinstance(element, list):
            return {x: copy.deepcopy(element) for x in self.networkGraph.getEdgesBetweenRouters()}
        elif isinstance(element, dict):
            return {x: copy.deepcopy(element) for x in self.networkGraph.getEdgesBetweenRouters()}
        elif isinstance(element, set):
            return {x: copy.deepcopy(element) for x in self.networkGraph.getEdgesBetweenRouters()}
        else:
            return {x: element for x in self.networkGraph.getEdgesBetweenRouters()}

    def hostsIpMapping(self):
        """
        Creates a mapping between host names and ip and viceversa
        :return:
        """

        self.hostsIpMapping = {}
        hosts = self.getHosts()
        self.hostsIpMapping["ipToName"] = {}
        self.hostsIpMapping["nameToIp"] = {}
        for host in hosts:
            self.hostsIpMapping["ipToName"][(hosts[host]["%s-eth0" % (host)]["ip"]).split("/")[0]] = host
            self.hostsIpMapping["nameToIp"][host] = (hosts[host]["%s-eth0" % (host)]["ip"]).split("/")[0]

    def routersIdMapping(self):
        self.routersIdMapping = {}
        self.routersIdMapping["idToName"] = {}
        self.routersIdMapping["nameToId"] = {}
        routers = self.getRouters()
        for name, data in routers.iteritems():
            routerid = data['routerid']
            self.routersIdMapping["idToName"][routerid] = name
            self.routersIdMapping["nameToId"][name] = routerid

    def getHostName(self, ip):

        """
        Returns the host name of the host that has the ip address
        :param ip:
        :return:
        """

        if ip not in self.hostsIpMapping["ipToName"]:
            raise ValueError("Any host of the network has the ip {0}".format(ip))

        return self.hostsIpMapping["ipToName"][ip]

    def getRouterName(self, routerid):
        return self.routersIdMapping['idToName'][routerid]

    def getRouterId(self, routername):
        return self.routersIdMapping['nameToId'][routername]

    def getRouterPod(self, routername):
        if not self.isCoreRouter(routername):
            return int(routername.split('_')[1])

    def getRouterIndex(self, routername):
        if '_e' in routername:
            return int(routername.split('_')[-1].strip('e'))
        elif '_a' in routername:
            return int(routername.split('_')[-1].strip('a'))
        else:
            return int(routername.split('_')[-1].strip('c'))

    def getRouterType(self, routername):
        e = [key for key in self.networkGraph.graph.node[routername].keys() if key in ['edge', 'core', 'aggregation']]
        if e:
            return e[0]
        else:
            raise ValueError

    def getHostIp(self, name):

        """
        returns the ip of host name
        :param name:
        :return:
        """

        if name not in self.hostsIpMapping["nameToIp"]:
            raise ValueError("Any host of the network has the ip {0}".format(name))

        return self.hostsIpMapping["nameToIp"][name]

    def loadIfNames(self):
        """
        Loads from the interface index corresponding to each router interface.
        The process is repeated for every router in the network.

        It will also update the interface information. it will add the mac and the ifindex
        to the dictionary so we can easliy do the mappings.
        :return:
        """

        # Holds the router interfaces names
        self.routersInterfaces = {}

        # Get the network routers
        routers = self.getRouters()

        for router in routers:
            # If we monitor from inside the network we use the router id,
            # otherwise we should get the interface-mon ips
            routerId = self.routerid(router)

            # If we want to use ifindexes
            if self.getIfIndexes:
                path = "{1}ifDescr_{0}".format(router, tmp_files)
                try:
                    with open(path, "r") as f:
                        nameToIfindex = json.load(f)
                except IOError:
                    raise KeyError("File {0} does not exist".format(path))

                tmp_dict = {}
                for interface in nameToIfindex:
                    if 'sit' not in interface:
                        mac = self.network[router][interface]['mac']
                        tmp_dict[mac] = {'ifname': interface, 'ifindex': nameToIfindex[interface]}

            # If we dont use ifindexes
            else:
                tmp_dict = {}
                for interface in self.network[router]:
                    # This is a little bit ugly. Change it in the future if I have time
                    if interface == "routerid" or "mon" in interface or interface == "type":
                        continue
                    mac = self.network[router][interface]['mac']
                    tmp_dict[mac] = {'ifname': interface}

            self.routersInterfaces[routerId] = tmp_dict

        return self.routersInterfaces

    def getIfNames(self):
        """
        Collects routers interfaces indexes if called for first time, and returns the mappings.

        :return:
        """
        if not (self.routersInterfaces):
            self.loadIfNames()

            # If its still empty, it means that countersDev is not working.
            if not (self.routersInterfaces):
                raise RuntimeError('Could not get data from the routers: check countersDev.py')

        return self.routersInterfaces

    def routerMacs(self, router):
        """
        Returns the MACs associated to the interfaces of a certain router

        :param router: router id
        :return: list of MACs
        """
        return self.routersInterfaces[router].keys()

    def getIfIndex(self, router, mac):
        """
        Returns the interface index of a router interface described by mac

        :param router: router id
        :param mac: mac of the interface
        :return: interface index
        """
        if self.getIfIndexes:
            return self.routersInterfaces[router][mac]['ifindex']
        else:
            # Since we don't know the index we return -1
            return -1

    def getIfName(self, router, mac):
        """
        Returns the interface name of a certain router interface

        :param router: router id
        :param mac: interface MAC
        :return: interface name
        """
        return self.routersInterfaces[router][mac]['ifname']

    def routerUsageToLinksLoad(self, routersUsage, link_loads):
        # For every router
        for router in routersUsage:
            isEdge = self.networkGraph.graph.node[router].has_key("edge")
            isAggr = self.networkGraph.graph.node[router].has_key("aggregation")
            # For every router interface
            routerId = self.routerid(router)
            for intfData in self.routersInterfaces[routerId].values():
                if isEdge:
                     # Using /proc/net/dev
                     if routersUsage[router]["out"].has_key(intfData["ifname"]):

                         link_loads[(router, self.network[router][intfData["ifname"]]["connectedTo"])] = round(routersUsage[router]["out"][intfData['ifname']], 3)
                         # however if it connect to a switch (sw, or ovs) we get the input value of that ifindex and compute the link cost (switch/ovs -> router)

                         if self.network[self.network[router][intfData["ifname"]]["connectedTo"]]["type"] == "switch":
                             link_loads[(self.network[router][intfData["ifname"]]["connectedTo"], router)] = round(routersUsage[router]["in"][intfData['ifname']], 3)
                elif isAggr:
                    # Using /proc/net/dev
                    if routersUsage[router]["out"].has_key(intfData["ifname"]):
                        link_loads[(router, self.network[router][intfData["ifname"]]["connectedTo"])] = round(routersUsage[router]["out"][intfData['ifname']], 3)
                    if routersUsage[router]["in"].has_key(intfData["ifname"]):
                        link_loads[(self.network[router][intfData["ifname"]]["connectedTo"], router)] = round(routersUsage[router]["in"][intfData['ifname']], 3)
                else:
                    # Only for countersDev class
                    if routersUsage[router]["out"].has_key(intfData["ifname"]):
                        link_loads[(router, self.network[router][intfData["ifname"]]["connectedTo"])] = round(routersUsage[router]["out"][intfData['ifname']], 3)

    def isEdgeRouter(self, router):
        """
        Check if router is edge router

        :param router: router name
        :return: boolean
        """
        return self.networkGraph.graph.node[router].has_key("edge")

    def isAggregationRouter(self, router):
        return self.networkGraph.graph.node[router].has_key("aggregation")

    def isCoreRouter(self, router):
        return self.networkGraph.graph.node[router].has_key("core")

if __name__ == "__main__":

    topology = TopologyGraph(getIfindexes=True, db=os.path.join(tmp_files, db_topo))
#    import ipdb; ipdb.set_trace()
    g = topology.networkGraph

    # topology.loadIfNames()
    # print

    # routers = topology.getIfNames()
    
    # roprint
