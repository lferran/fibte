#!/usr/bin/python
from fibbingnode.misc.mininetlib.ipnet import TopologyDB
import os
import json
from fibte.monitoring.snmplib import SnmpIfDescr
from fibte.monitoring.countersDev import IfDescr
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
#from lb.customExceptions import ValueError, InvalidIP

from fibte import CFG

tmp_files = CFG.get("DEFAULT", "tmp_files")
db_topo = CFG.get("DEFAULT", "db_topo")


class NetworkGraph(object):
    def __init__(self, topologyDB):

        # TopoDB object
        self.topologyDB = topologyDB

        # networkx graph
        self.graph = self.loadGraphFromDB(self.topologyDB)

    def loadGraphFromDB(self, topologyDB):

        g = nx.Graph()

        for node in topologyDB.network:

            import ipdb; ipdb.set_trace()

            if node not in g.nodes() and (node not in ["default_controller", "sw-mon", "ryu_controller"]):
                g.add_node(node)
                g.node[node]['type'] = topologyDB.type(node)

                # TODO IMPORTANT, here we will differenciate a type of routers. The ones that contain the letter e, will be
                # classified as edge routers. Edge routers are used to compute all the possible paths within the
                # topology
                if 'e' in node:
                    g.node[node]['edge'] = True

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

    def numberofPathsBetweenEdges(self):

        total_paths = 0
        edgeRouters = self.getEdgeRouters()
        for router in edgeRouters:
            for router_pair in edgeRouters:
                if router == router_pair:
                    continue
                npaths = sum(1 for _ in nx.all_shortest_paths(self.graph, router, router_pair))
                total_paths += npaths
        return total_paths

    def getAllPaths(self):

        paths = []
        edgeRouters = self.getEdgeRouters()
        for router in edgeRouters:
            for router_pair in edgeRouters:
                if router != router_pair:
                    paths += list(nx.all_shortest_paths(self.graph, router, router_pair))
        return paths

    def totalNumberOfPaths(self):

        """
        This function is very useful if the topology is unknown, however if we are using a fat tree, the number of paths is more or less
        (k/2**2) = number of paths from one node to another node that its not in the same pod
        number of nodes its k**3 / 4
        so number of paths is : (k/2**2) * (k**3)/4 - k**2/4)(this is number of nodes outside the pod) * total number of nodes
        here we should add the number of paths inside the pod
        + number of paths between hosts connected by the same router.
        :return:
        """

        total_paths = 0
        for host in self.getHosts():
            for host_pair in self.getHosts():
                if host == host_pair:
                    continue

                # compute the number of paths
                npaths = sum(1 for _ in nx.all_shortest_paths(self.graph, host, host_pair))
                total_paths += npaths

        return total_paths

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

    # returns all the edges between routers in the network
    def getEdgesBetweenRouters(self):

        allEdges = self.graph.to_directed().edges(nbunch=self.getRouters())

        # filter the edges and remove the ones that have switches connected
        return [x for x in allEdges if not (any("sw" in s for s in x) or any("ovs" in s for s in x))]


class TopologyGraph(TopologyDB):
    def __init__(self, getIfindexes=True, snmp=False, openFlowInformation=False, interfaceToRouterName=False, *args,
                 **kwargs):

        super(TopologyGraph, self).__init__(*args, **kwargs)

        # Indicates weather snmp is used or not
        self.snmp = snmp

        # Router interface name to interface snmp index mappings
        self.getIfIndexes = getIfindexes


        self.routersInterfaces = {}


        self.getIfNames()


        self.networkGraph = NetworkGraph(self)


        self.hostsIpMapping()

        if openFlowInformation:
            self.loadOpenFlowInformation()

        # used to find as fast as possible a router from one of its interface IP's
        if interfaceToRouterName:
            self.loadInterfaceToRouterName()
            # log.setLevel(logging.INFO)

        import ipdb; ipdb.set_trace()

    def loadInterfaceToRouterName(self):

        self.interfaceIPToRouterName = {}
        for name, interfaces in self.getRouters().iteritems():
            for interface, parameters in interfaces.iteritems():
                if isinstance(parameters, dict):
                    self.interfaceIPToRouterName[parameters["ip"].split("/")[0]] = name

    def getRouterFromInterfaceIp(self, ip):

        return self.interfaceIPToRouterName[ip]

    def inSameSubnetwork(self, srcHost, dstHost):

        """
        Checks if src host and dst host belong to the same subnet.
        The function assumes that every host has only one interface
        :param srcHost:
        :param dstHost:
        :return: Returns a boolean
        """

        # srcIp = self.network[srcHost]["{0}-eth0".format(srcHost)]["ip"]
        # dstIp = self.network[dstHost]["{0}-eth0".format(dstHost)]["ip"]

        if srcHost not in self.network:
            raise ValueError("{0} does not exist".format(srcHost))

        if dstHost not in self.network:
            raise ValueError("{0} does not exist".format(dstHost))

        return self.networkGraph.inSameSubnet(srcHost, dstHost)
        # return ip_interface(srcIp).network == ip_interface(dstIp).network

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

    def totalNumberOfPaths(self):

        """
        Returns the total number of paths between every host. This function is not really used, instead
        numberOfPathsBetweenEdges should be used.
        :return:
        """

        return self.networkGraph.totalNumberOfPaths()

    def getHostsBehindRouter(self, router):

        """
        Returns a list with all the hosts that have router as a gateway
        :param router:
        :return:
        """

        if router not in self.network:
            raise ValueError("{0} does not exist".format(router))

        return self.networkGraph.getHostsBehindRouter(router)

    def getHostsInOtherSubnetworks(self, host):

        """
        Returns all the hosts from other subnetworks. This is used to generate traffic only to "remote" hosts
        :param host:
        :return:
        """

        if host not in self.network:
            raise ValueError("{0} does not exist".format(host))

        # returns all the hosts that are in other subnetworks
        gatewayEdgeRouter = self.getGatewayRouter(host)
        edgeRouters = self.getEdgeRouters()

        otherHosts = []
        for edgeRouter in edgeRouters:
            if edgeRouter == gatewayEdgeRouter:
                continue
            otherHosts += self.getHostsBehindRouter(edgeRouter)

        return otherHosts

    def numberOfPathsBetweenEdges(self):

        """
        Return the number of paths between edge routers. In theory the returned number is the total paths we have to
        find in order to fill our header database
        :return:
        """

        return self.networkGraph.numberofPathsBetweenEdges()

    def numberDevicesInPath(self, src, dst):
        # returns the number of devices between two nodes in the network

        if src not in self.network:
            raise ValueError("{0} does not exist".format(src))

        if dst not in self.network:
            raise ValueError("{0} does not exist".format(dst))

        return len(nx.shortest_path(self.networkGraph.graph, src, dst))

    def getRouters(self):

        "Gets the routers from the topologyDB"

        return {node: self.network[node] for node in self.network if self.network[node]["type"] == "router"}

    def getEdgeRouters(self):

        return self.networkGraph.getEdgeRouters()

    def getHosts(self):

        "Gets the routers from the topologyDB"

        return {node: self.network[node] for node in self.network if self.network[node]["type"] == "host"}

    def getSwitches(self):

        return {node: self.network[node] for node in self.network if self.network[node]["type"] == "switch"}

    def isOVS(self, node):

        if node not in self.network:
            raise ValueError("{0} does not exist".format(node))

        return self.network[node]["type"] == "switch" and node[:3] == "ovs"

    def getEdgesUsageDictionary(self, element=0):

        if isinstance(element, list):
            return {x: copy.deepcopy(element) for x in self.networkGraph.getEdgesBetweenRouters()}
        elif isinstance(element, dict):
            return {x: copy.deepcopy(element) for x in self.networkGraph.getEdgesBetweenRouters()}
        elif isinstance(element, set):
            return {x: copy.deepcopy(element) for x in self.networkGraph.getEdgesBetweenRouters()}
        else:
            return {x: element for x in self.networkGraph.getEdgesBetweenRouters()}

    def getPathsUsageDictionary(self, element=0):

        # not implemented yet. with big k this dictionary would be to big...
        pass

    def getHostOVS(self, hostIP):

        """
        From the host ip we get the OVS name. This function is used to get the switch in which we want to install
        rules using the FLOW header.
        :param hostIP:
        :return:
        """

        # from the host ip it returns the ovs switch that belongs to that host
        node = self.getHostName(hostIP)

        if node not in self.network:
            raise ValueError("{0} does not exist".format(node))

        return self.network[node]["{0}-eth0".format(node)]["connectedTo"]

    def getOVSInformation(self):

        # getting bridge level information
        bridge_info = json.loads(
            subprocess.check_output("ovs-vsctl --format json --columns name,datapath_id list Bridge", shell=True))
        data = bridge_info["data"]
        headers = bridge_info["headings"]
        self.bridgeInfo = {}

        for bridge in data:
            for i, bridge_data in enumerate(bridge):
                if headers[i] == "name":
                    self.bridgeInfo[bridge[i]] = {}
                    name_tmp = bridge[i]
                else:

                    self.bridgeInfo[name_tmp][headers[i]] = bridge_data

        # same for interfaces information
        interfaces_info = json.loads(
            subprocess.check_output("ovs-vsctl --format json --columns name,ofport,ifindex list Interface", shell=True))
        data = interfaces_info["data"]
        headers = interfaces_info["headings"]
        self.interfacesInfo = {}

        for interface in data:
            for i, interface_data in enumerate(interface):
                if headers[i] == "name":
                    self.interfacesInfo[interface[i]] = {}
                    name_tmp = interface[i]
                else:

                    self.interfacesInfo[name_tmp][headers[i]] = interface_data

    def loadOpenFlowInformation(self):

        # we load the information from the ovsdb using the command line interface to access it
        self.getOVSInformation()

        for switch in self.getSwitches():
            if self.isOVS(switch):
                self.network[switch]["openFlow"] = {}

                # adding the datapath id
                self.network[switch]["openFlow"].update(self.bridgeInfo[switch])

                # adding interface information
                for interface in self.network[switch]:
                    # avoid this two entries of the dictionary
                    if interface in ["type", "openFlow"]:
                        continue

                    connected_to = self.network[switch][interface]["connectedTo"]

                    # if that interface is connected to a host we assign that port as the input
                    # port of the switch
                    if self.network[connected_to]["type"] == "host":

                        in_port = self.interfacesInfo[interface]["ofport"]

                    else:
                        out_port = self.interfacesInfo[interface]["ofport"]

                self.network[switch]["openFlow"].update({"in_port": in_port, "out_port": out_port})

    def getDatapathID(self, switch):

        """
        Returns datapath id of an OVS host
        :param switch:
        :return:
        """

        if switch not in self.network:
            raise ValueError("{0} does not exist".format(switch))

        if self.isOVS(switch):
            return self.network[switch]["openFlow"]["datapath_id"]

        print "%s is not an OVS switch" % switch

    def getInPort(self, switch):

        """
        Returns the in port index. The in port is considered the port where the host is connected.
        :param switch:
        :return:
        """

        if switch not in self.network:
            raise ValueError("{0} does not exist".format(switch))

        if self.isOVS(switch):
            return self.network[switch]["openFlow"]["in_port"]

        print "%s is not an OVS switch" % switch

    def getOutPort(self, switch):

        """
        Returns the out port index. The out port is considered the port that faces the network side.
        :param switch:
        :return:
        """

        if switch not in self.network:
            raise ValueError("{0} does not exist".format(switch))

        if self.isOVS(switch):
            return self.network[switch]["openFlow"]["out_port"]

        print "%s is not an OVS switch" % switch

    def getOneHostPerNetwork(self):

        """
        Returns a host name per subnetwork
        :return:
        """

        hosts = self.getHosts()

        seenNetworks = set([])
        hostPerNetwork = []
        for host in hosts:
            hostNetwork = ip_interface(hosts[host]["%s-eth0" % (host)]["ip"]).network
            if hostNetwork not in seenNetworks:
                seenNetworks.add(hostNetwork)
                hostPerNetwork.append(host)
        return hostPerNetwork

    def getNHostsPerNetwork(self, n):

        """
        Returns n host name per subnetwork
        :param n:
        :return:
        """

        hosts = self.getHosts()

        seenNetworks = {}
        hostsPerNetwork = []

        for host in hosts:
            hostNetwork = ip_interface(hosts[host]["%s-eth0" % (host)]["ip"]).network
            if hostNetwork not in seenNetworks:
                seenNetworks[hostNetwork] = 1
                hostsPerNetwork.append(host)
            elif seenNetworks[hostNetwork] < n:
                seenNetworks[hostNetwork] += 1
                hostsPerNetwork.append(host)

        return hostsPerNetwork

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

    def getHostName(self, ip):

        """
        Returns the host name of the host that has the ip address
        :param ip:
        :return:
        """

        if ip not in self.hostsIpMapping["ipToName"]:
            raise InvalidIP("Any host of the network has the ip {0}".format(ip))

        return self.hostsIpMapping["ipToName"][ip]

    def getHostIp(self, name):

        """
        returns the ip of host name
        :param name:
        :return:
        """

        if name not in self.hostsIpMapping["nameToIp"]:
            raise ValueError("Any host of the network has the ip {0}".format(name))

        return self.hostsIpMapping["nameToIp"][name]

    # Deprecated
    # def getSetOfInterfaces(self):
    #
    #     interfaces = self.getIfNames()
    #
    #     setOfInterfaces = set(sum([x.values() for x in interfaces.values()],[]))
    #
    #     return setOfInterfaces


    def loadIfNames(self):
        """
        Loads from the routers and using SNMP the interface index corresponding to each router interface.
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

            # if we want to use SNMP ifindexes
            if self.getIfIndexes:
                if self.snmp:
                    nameToIfindex = SnmpIfDescr(routerIp=routerId).getIfMapping()
                else:
                    path = "{1}ifDescr_{0}".format(router, tmp_files)
                    # subprocess.call("mx {0} ~/customized_load_balancing_sdn/complexTopology/src/ifDescrNamespace.py {1} &".format(router,path),shell=True)
                    try:
                        with open(path, "r") as f:
                            nameToIfindex = json.load(f)
                    except IOError:
                        raise KeyError("FERRIS") #log.debug("File {0} does not exist".format(path))
                tmp_dict = {}
                for interface in nameToIfindex:
                    mac = self.network[router][interface]['mac']
                    tmp_dict[mac] = {'ifname': interface, 'ifindex': nameToIfindex[interface]}

            # If we dont use snmp ifindexes
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

            # if its still empty, it means that snmp is not working
            if not (self.routersInterfaces):
                raise RuntimeError('Could not get SNMP data from the routers')

        return self.routersInterfaces

    def routerMacs(self, router):
        # Router is the router p..

        return self.routersInterfaces[router].keys()

    def getIfIndex(self, router, mac):
        if self.getIfIndexes:
            return self.routersInterfaces[router][mac]['ifindex']
        else:
            # since we don't know the index we return -1
            return -1

    def getIfName(self, router, mac):

        return self.routersInterfaces[router][mac]['ifname']

    def routerUsageToLinksLoad(self, routersUsage, link_loads):

        # for every royter
        for router in routersUsage:
            isEdge = self.networkGraph.graph.node[router].has_key("edge")
            # for every router interface
            routerId = self.routerid(router)
            for intfData in self.routersInterfaces[routerId].values():
                if isEdge:
                    # in case we are using ifindex for the loads
                    if self.snmp:
                        if routersUsage[router]["out"].has_key(intfData["ifindex"]):
                            # here we do the same
                            link_loads[(router, self.network[router][intfData["ifname"]]["connectedTo"])] = round(
                                routersUsage[router]["out"][intfData['ifindex']], 3)
                            # however if it connect to a switch (sw, or ovs) we get the input value of that ifindex and compute the link cost (switch/ovs -> router)

                            if self.network[self.network[router][intfData["ifname"]]["connectedTo"]][
                                "type"] == "switch":
                                link_loads[(self.network[router][intfData["ifname"]]["connectedTo"], router)] = round(
                                    routersUsage[router]["in"][intfData['ifindex']], 3)

                    # using /proc/net/dev
                    else:

                        if routersUsage[router]["out"].has_key(intfData["ifname"]):

                            link_loads[(router, self.network[router][intfData["ifname"]]["connectedTo"])] = round(
                                routersUsage[router]["out"][intfData['ifname']], 3)
                            # however if it connect to a switch (sw, or ovs) we get the input value of that ifindex and compute the link cost (switch/ovs -> router)

                            if self.network[self.network[router][intfData["ifname"]]["connectedTo"]][
                                "type"] == "switch":
                                link_loads[(self.network[router][intfData["ifname"]]["connectedTo"], router)] = round(
                                    routersUsage[router]["in"][intfData['ifname']], 3)

                else:
                    if self.snmp:
                        if routersUsage[router].has_key(intfData["ifindex"]):
                            link_loads[(router, self.network[router][intfData["ifname"]]["connectedTo"])] = round(
                                routersUsage[router][intfData['ifindex']], 3)
                    else:
                        # only for countersDev class
                        if routersUsage[router]["out"].has_key(intfData["ifname"]):
                            link_loads[(router, self.network[router][intfData["ifname"]]["connectedTo"])] = round(
                                routersUsage[router]["out"][intfData['ifname']], 3)


if __name__ == "__main__":

#    import ipdb; ipdb.set_trace()
    topology = TopologyGraph(getIfindexes=True, snmp=False, openFlowInformation=True,
                             db=os.path.join(tmp_files, db_topo))
    g = topology.networkGraph


    # topology.loadIfNames()
    # print

    # routers = topology.getIfNames()

    # roprint