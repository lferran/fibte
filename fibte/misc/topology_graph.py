#!/usr/bin/python

import os
import json
import networkx as nx
import copy
import ipaddress
import matplotlib.pyplot as plt
import time

from fibbingnode.misc.mininetlib.ipnet import TopologyDB

from fibte import CFG

tmp_files = CFG.get("DEFAULT", "tmp_files")
db_topo = CFG.get("DEFAULT", "db_topo")
private_ip_binding_file = CFG.get("DEFAULT", 'private_ip_binding_file')

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

class TopologyGraph(TopologyDB):
    def __init__(self, getIfindexes=True, interfaceToRouterName=False, *args, **kwargs):
        super(TopologyGraph, self).__init__(*args, **kwargs)

        # Router interface name to interface index mappings
        self.getIfIndexes = getIfindexes

        # Holds the data of router interfaces
        self.routersInterfaces = {}

        # Retrieces the interface names
        self.getIfNames()

        # Starts the network graph
        self.networkGraph = self.loadGraphFromDB()

        # Populates self.hostsIpMapping dictionary
        self.hostsIpMapping()

        # Populataes self.routersIpMappin dictionary
        self.routersIdMapping()

        # Fill gateway into host information
        self.fillHostGateways()

        # Fills a list of initial OSPF prefixes
        self.initialPrefixes = self._getInitialNetworkPrefixes()

        # Populates self.hostToNetworksMapping dictionary
        self._hostsToNetworksMapping()

        # Populates self.interfaceIPToRouterName dict
        if interfaceToRouterName:
            self.loadInterfaceToRouterName()

        self.private_ip_binding = self.loadPrivateIpBindings()

    def loadPrivateIpBindings(self):
        bindings = {'privateToPublic': {}, 'publicToPrivate': {}}
        with open(private_ip_binding_file, 'r') as f:
            bds = json.loads(f.read())

        for priv_net, data in bds.iteritems():
            for rid, prv_ip in data.iteritems():
                private_ip = prv_ip[0].split('/')[0]

                if rid not in bindings['publicToPrivate'].keys():
                    bindings['publicToPrivate'][rid] = {priv_net: private_ip}
                else:
                    bindings['publicToPrivate'][rid][priv_net] = private_ip

                bindings['privateToPublic'][private_ip] = rid
        return bindings

    def guess_router_name(self, ip):
        for fun in [self.getRouterName, self.getRouterFromPrivateIp, self.getRouterFromInterfaceIp]:
            try:
                return fun(ip)
            except KeyError:
                continue
        return ip

    def getRouterIdFromPrivateIp(self, private_ip):
        return self.private_ip_binding['privateToPublic'][private_ip]

    def getRouterFromPrivateIp(self, private_ip):
        rid = self.getRouterIdFromPrivateIp(private_ip)
        return self.getRouterName(rid)

    def loadGraphFromDB(self):
        g = nx.Graph()
        for node in self.network:
            if node not in g.nodes() and (node not in ["default_controller", "sw-mon", "c1"]):
                g.add_node(node)
                g.node[node]['type'] = self.type(node)

                # TODO IMPORTANT, here we will differenciate a type of routers. The ones that contain the letter e, will be
                # classified as edge routers. Edge routers are used to compute all the possible paths within the
                # topology
                if '_e' in node:
                    g.node[node]['edge'] = True

                elif '_a' in node:
                    g.node[node]['aggregation'] = True

                elif 'r_c' in node:
                    g.node[node]['core'] = True

                elif 'h_' in node:
                    g.node[node]['host'] = True

                for itf in self.network[node]:
                    # we should ignore routerid, type.
                    # TODO have to find a better way to do this
                    if itf in ["routerid", 'type', 'gateway']:
                        continue

                    # else itf its a real interface so we add an edge,
                    # only if the connected node has been created.
                    connectedTo = self._interface(node, itf)["connectedTo"]

                    if connectedTo in g.nodes():
                        # add edge
                        g.add_edge(node, connectedTo)
        return g

    def fillHostGateways(self):
        self.hostToGatewayMapping = {}
        self.hostToGatewayMapping['hostToGateway'] = {}
        self.hostToGatewayMapping['gatewayToHosts'] = {}
        hosts = self.getHosts()
        for host, host_data in hosts.iteritems():
            for iface, iface_data in host_data.iteritems():
                if iface != 'type':
                    connected_to = iface_data['connectedTo']
                    # Case in which intermediate switch is not there
                    if self.type(connected_to) == 'router':
                        gw = connected_to
                    elif self.type(connected_to) == 'switch':
                        gw = [data['connectedTo'] for _, data in self.network[connected_to].iteritems()
                              if type(data) == dict and self.type(data['connectedTo']) == 'router'][0]

                    iface_data['gateway'] = gw
                    self.hostToGatewayMapping['hostToGateway'][host] = gw
                    if gw not in self.hostToGatewayMapping['gatewayToHosts']:
                        self.hostToGatewayMapping['gatewayToHosts'][gw] = [host]
                    else:
                        self.hostToGatewayMapping['gatewayToHosts'][gw].append(host)

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

        return self.getGatewayRouter(srcHost) == self.getGatewayRouter(dstHost)

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

        srcEdge = self.getGatewayRouter(srcHost)
        dstEdge = self.getGatewayRouter(dstHost)

        paths = list(nx.all_shortest_paths(self.networkGraph, srcEdge, dstEdge))
        paths = [tuple(x) for x in paths]

        return paths

    def getGatewayRouter(self, host):
        """
        Given a host it returns its gateway router
        :param host:
        :return:
        """
        if host not in self.network:
            raise ValueError("{0} does not exist".format(host))

        return self.hostToGatewayMapping['hostToGateway'][host]

    def getHostsBehindRouter(self, router):
        """
        Returns a list with all the hosts that have router as a gateway
        :param router:
        :return:
        """

        if router not in self.network:
            raise ValueError("{0} does not exist".format(router))

        if not self.isEdgeRouter(router):
            raise ValueError("{0} is not an edge router".format(router))

        return [h for h in self.getHosts().keys() if self.getGatewayRouter(h) == router]

    def getHostsBehindPod(self, pod):
        """
        Returns all the hosts under a given pod

        :param pod: a pod number is expected in string form "pod_X"
        :return: a list of hosts under that pod
        """

        # Get pod number
        if isinstance(pod, int):
            pod_number = pod
        elif isinstance(pod, str):
            pod_number = pod.split('_')[1]
        else:
            raise ValueError("{0} is neither in int form (X) or string form (pod_X)")

        # Calculate corresponding edge router regex
        regex = "r_{0}_e".format(pod_number)

        # Get all edge routers
        edge_routers = self.getEdgeRouters()

        # Filter those that match the regex and get a set of hosts
        host_lists = [self.getHostsBehindRouter(edge_router) for edge_router in edge_routers if regex in edge_router]
        host_set = {host for host_list in host_lists for host in host_list}
        return list(host_set)

    def getRouters(self):
        """
        Returns a dictionary with the routers from the topologyDB
        :return: router-only dict
        """
        return {node: data for node, data in self.network.iteritems() if data["type"] == "router"}

    def getEdgeRouters(self):
        return [x for x in self.networkGraph.node if self.networkGraph.node[x].has_key("edge")]

    def getAggregationRouters(self):
        return [x for x in self.networkGraph.node if self.networkGraph.node[x].has_key("aggregation")]

    def getCoreRouters(self):
        """
        Only core routers
        :return: list of core routers
        """
        return [x for x in self.networkGraph.node if self.networkGraph.node[x].has_key("core")]

    def getHosts(self):
        """
        Gets the hosts from the topologyDB

        :return: host-only dictionary
        """
        return {node: data for node, data in self.network.iteritems() if data["type"] == "host"}

    def getSwitches(self):
        """
        Gets the switches from the topologyDB

        :return: switches-only dictionary
        """
        return {node: self.network[node] for node in self.network if self.network[node]["type"] == "switch"}

    def getEdgesUsageDictionary(self, element=0):

        if isinstance(element, list):
            return {x: copy.deepcopy(element) for x in self.getEdgesBetweenRouters()}
        elif isinstance(element, dict):
            return {x: copy.deepcopy(element) for x in self.getEdgesBetweenRouters()}
        elif isinstance(element, set):
            return {x: copy.deepcopy(element) for x in self.getEdgesBetweenRouters()}
        else:
            return {x: element for x in self.getEdgesBetweenRouters()}

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
            self.hostsIpMapping["ipToName"][(hosts[host]["%s-eth0"%(host)]["ip"]).split("/")[0]] = host
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

    def _hostsToNetworksMapping(self):
        self.hostsToNetworksMapping = {}
        hosts = self.getHosts()
        self.hostsToNetworksMapping['hostToNetwork'] = {}
        self.hostsToNetworksMapping['networkToHost'] = {}

        for host, host_data in hosts.iteritems():
            self.hostsToNetworksMapping['hostToNetwork'][host] = {}
            for iface, iface_data in host_data.iteritems():
                if iface != 'type':
                    iface_network = self.hostInterfaceNetwork(host, iface)
                    self.hostsToNetworksMapping['hostToNetwork'][host][iface] = iface_network

                    if iface_network not in self.hostsToNetworksMapping['networkToHost']:
                        self.hostsToNetworksMapping['networkToHost'][iface_network] = {}

                    if host not in self.hostsToNetworksMapping['networkToHost'][iface_network]:
                        self.hostsToNetworksMapping['networkToHost'][iface_network][host] = [iface]
                    else:
                        self.hostsToNetworksMapping['networkToHost'][iface_network][host].append(iface)

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
        e = [key for key in self.network.node[routername].keys() if key in ['edge', 'core', 'aggregation']]
        if e:
            return e[0]
        else:
            raise ValueError

    def getHostPod(self, hostname):
        """"""
        gw = self.getGatewayRouter(hostname)
        return self.getRouterPod(gw)

    def sortHostsByName(self, hostsList):
        pseudoList = [(hostName, hostName.split("_")[1], hostName.split("_")[2]) for hostName in hostsList]
        pseudoList = sorted(pseudoList,key=lambda host:(host[1],host[2]))
        return [x[0] for x in pseudoList]

    def getEdgesBetweenRouters(self):
        """
        Returns all edges between routers in the network
        :return:
        """
        allEdges = self.networkGraph.to_directed().edges(nbunch=self.getRouters())

        # Filter the edges and remove the ones that have switches connected
        return [x for x in allEdges if not (any("sw" in s for s in x) or any("ovs" in s for s in x))]

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
            isEdge = self.networkGraph.node[router].has_key("edge")
            isAggr = self.networkGraph.node[router].has_key("aggregation")
            # For every router interface
            routerId = self.routerid(router)
            for intfData in self.routersInterfaces[routerId].values():
                if isEdge:
                     # Using /proc/net/dev
                     if routersUsage[router]["out"].has_key(intfData["ifname"]):
                         link_loads[(router, self.network[router][intfData["ifname"]]["connectedTo"])] = round(routersUsage[router]["out"][intfData['ifname']], 3)

                         if self.network[self.network[router][intfData["ifname"]]["connectedTo"]]["type"] == "host":
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
                        # Avoid adding fibbing controller here
                        if self.network[router][intfData["ifname"]]['connectedTo'] != 'c1':
                            link_loads[(router, self.network[router][intfData["ifname"]]["connectedTo"])] = round(routersUsage[router]["out"][intfData['ifname']], 3)

    def isEdgeRouter(self, router):
        """
        Check if router is edge router

        :param router: router name
        :return: boolean
        """
        return self.networkGraph.node[router].has_key("edge")

    def isAggregationRouter(self, router):
        return self.networkGraph.node[router].has_key("aggregation")

    def isCoreRouter(self, router):
        return self.networkGraph.node[router].has_key("core")

    def isHost(self, node_name):
        return self.network[node_name]['type'] == 'host'

    def hostInterfaceIP(self, host, interface):
        return self._interface(host, interface)['ip'].split("/")[0]

    def hostInterfaceNetwork(self, host, interface):
        return ipaddress.ip_interface(self._interface(host, interface)['ip']).network.compressed

    def _getInitialNetworkPrefixes(self):
        initial_prefixes = []
        hosts = self.getHosts()
        for host, host_data in hosts.iteritems():
            for iface, iface_data in host_data.iteritems():
                if iface != 'type':
                    iface_network = self.hostInterfaceNetwork(host, iface)
                    initial_prefixes.append(iface_network)

        return initial_prefixes

    def getInitialNetworkPrefixes(self):
        return self.initialPrefixes

    def getGatewayRouterFromNetworkPrefix(self, prefix):
        host = self.hostsToNetworksMapping['networkToHost'][prefix].keys()[0]
        return self.getGatewayRouter(host)

    def getBisectionEdges(self):
        """
        Returns all edges between routers in the network
        :return:
        """
        import ipdb; ipdb.set_trace()

    def getHopsBetweenHosts(self, src, dst):
        """
        Returns the number of devices in
        the path from src host to dst host
        """
        if src not in self.network:
            raise ValueError("{0} does not exist".format(src))

        if dst not in self.network:
            raise ValueError("{0} does not exist".format(dst))

        if src == dst:
            raise ValueError("src == dst")

        return len(nx.shortest_path(self.networkGraph, src, dst)) - 2

    def areNeighbors(self, node1, node2):
        return node1 in self.networkGraph.adj[node2]

    def getFatTreePositions(self, k=4):
        # assume that g is already the reduced graph
        # assume that we named the nodes using the fat tree "structure"
        # assume that we know k

        positions = {}

        hostStartPos = (0, 0)
        edgeStartPos = (1, 1)
        aggStartPos = (1, 2)

        hostBaseName = "h_{0}_{1}"
        edgeBaseName = "r_{0}_e{1}"
        aggBaseName = "r_{0}_a{1}"
        coreBaseName = "r_c{0}"

        # allocate hosts
        for pod in range(k):
            host_index = 0
            for edge_index in range(k / 2):
                for sub_edge_index in range(k / 2):
                    # Compute current host index
                    positions[hostBaseName.format(pod, host_index)] = hostStartPos

                    # Compute next host position
                    hostStartPos = (hostStartPos[0] + 2, hostStartPos[1])

                    # Increase host index
                    host_index += 1

                # Set position of first host in next edge
                hostStartPos = (hostStartPos[0] - 1, hostStartPos[1])

            # Set position of first host in next pod
            hostStartPos = (hostStartPos[0] + 2.5, hostStartPos[1])

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

#        print positions
        return positions

class NamesToIps(dict):
    def __init__(self, db_path='/tmp/topo.db'):
        super(NamesToIps, self).__init__()

        self['nameToIp'] = {}
        self['ipToName'] = {}

        with open(db_path, 'r') as f:
            network = json.load(f)

        for n, data in network.iteritems():
            is_host = 'h' in n
            is_router = 'r' in n
            if is_host:
                h_ip = self.getPrimaryIp(network, n)
                self['nameToIp'][n] = h_ip
                self['ipToName'][h_ip] = n
            if is_router:
                pass


    def getPrimaryIp(self, network, h):
        data = network[h]
        for k, v in data.iteritems():
            if isinstance(v, dict):
                iface_ip = ipaddress.ip_interface(v['ip'])
                return iface_ip.ip.compressed

if __name__ == "__main__":
    topology = TopologyGraph(getIfindexes=True, db=os.path.join(tmp_files, db_topo))
    import ipdb;ipdb.set_trace()
    topology.getHostsBehindRouter('r_0_e0')
