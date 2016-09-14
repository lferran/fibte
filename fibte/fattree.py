from fibbingnode.misc.mininetlib.iptopo import IPTopo

from mininet.nodelib import LinuxBridge
from mininet.util import custom
from mininet.link import TCIntf

import fibte.res.config as cfg

class FatTree(IPTopo):
    def __init__(self, k=4, sflow=False, ovs_switches=True, *args, **kwargs):
        # k must be multiple of 2
        self.k = k

        # datapath-id for the switches, if you dont specify mininet assign them duplicated
        self.dpid = 1

        # Using sflow?
        self.sflow = sflow

        # Using host-edge intermediate switches?
        self.ovs_switches = ovs_switches

        super(FatTree,self).__init__(*args,**kwargs)

    def dpidToStr(self):
        strDpid = format(self.dpid,'x')
        if len(strDpid) < 16:
            return "0"*(16-len(strDpid)) + strDpid
        return strDpid

    def build(self, *args, **kwargs):
        # Build the topology

        # Create the pods first
        aggregationRouters = self.addPods()

        # Create the core routers
        coreRouters = self.addCoreRouters()

        # Connect them together
        self.connectCoreAggregation(aggregationRouters, coreRouters)

        # Add Fibbing Controller to one of the core routers
        fibbingController = self.addController(cfg.C1, cfg_path=cfg.C1_cfg)
        self.addLink(fibbingController, coreRouters[0], cost=10000)

    def connectCoreAggregation(self, aggregationRouters, coreRouters):
        # Connect every aggregation router with k/2 core routers
        for i, aggregationRouter in enumerate(aggregationRouters):
            # Position inside the pod
            position = i % (self.k/2)

            # Connect with core routers
            for coreRouter in coreRouters[(position*(self.k/2)):((position+1)*(self.k/2))]:
                self.addLink(aggregationRouter, coreRouter)

    def addOVSHost(self, podNum, index):
        """
        Creates a host/switch pair. Every host in the fat tree topology is based on a
        normal mininet host + an ovs switch that allows flow modifications.
        :param index: Host number in the topology
        :return:returns a tuple of the form (h, sw)
        """
        h = self.addHost("h_%d_%d" % (podNum,index))
        #sw = self.addSwitch("ovs_%d_%d" % (podNum,index), sflow=self.sflow, cls=RyuSwitch,dpid = self.dpidToStr())
        sw = self.addSwitch("ovs_%d_%d"%(podNum,index), cls=LinuxBridge)
        self.dpid +=1
        self.addLink(h, sw)#,bw=self.bw)
        return {"host": h, "ovs": sw}

    def addNormalHost(self, podNum, index):
        """Adds a host with the corresponding pod number and index"""
        h = self.addHost("h_%d_%d" % (podNum,index))
        self.dpid +=1
        return h

    def addHostsGrup(self, podNum, startIndex):
        """
        Given the pod number,
        """
        # Contains the name of the switches or hosts, depending on wheather we use intermediate ovs switches or not
        to_connect_to_edge = []

        for i in range(self.k/2):
            if self.ovs_switches:
                ovsHosts = self.addOVSHost(podNum, startIndex+i)
                to_connect_to_edge.append(ovsHosts["ovs"])
            else:
                normalHosts = self.addNormalHost(podNum, startIndex+i)
                to_connect_to_edge.append(normalHosts)

        return to_connect_to_edge

    def addPods(self, extraSwitch=True):
        aggregationRouters = []

        # Add k pods and store the aggregation Routers in aggregationRouters
        for i in range(self.k):
            aggregationRouters += (self.addPod(i))

        return aggregationRouters

    def addPod(self, podNum):
        """Creates a pod and returns the corresponding aggregation routers
        """
        edgeRouters = []
        aggregationRouters = []

        # Add Aggregation and Edge routers
        for i in range(self.k/2):
            edgeRouters.append(self.addRouter("r_%d_e%d" % (podNum, i)))
            aggregationRouters.append(self.addRouter("r_%d_a%d" % (podNum, i)))

        # Connect Aggregation layer with Edge layer
        for edge_router in edgeRouters:
            for aggregation_router in aggregationRouters:
                self.addLink(edge_router, aggregation_router)#,bw = self.bw)

        # Add hosts to the edge layer, each edge router should be connected to k/2 hosts
        startIndex  = 0
        for edge_router in edgeRouters:
            # Create hosts and switches and connect them
            to_connect_to_edge = self.addHostsGrup(podNum, startIndex)

            # Connect switch/switches with edge router
            for node in to_connect_to_edge:
                self.addLink(edge_router, node)

            startIndex += self.k/2

        # Only aggregation Routers are needed to connect with the core layer
        return aggregationRouters

    def addCoreRouters(self):
        """
        :return:
        """
        coreRouters = []

        #create (k/2)^2 core routers. Each one will be connected to every pod through one aggregation router
        for i in range((self.k/2)**2):
            coreRouters.append(self.addRouter("r_c%d" % i))
        return coreRouters


class FatTreeOOB(FatTree):
    """
    Build Fat Tree topology with an Out-Of-Band management/monitoring network:

    Each router has an interface connected to a monitoring switch sw-mon.
    """
    def build(self, *args, **kwargs):
        super(FatTreeOOB, self).build(*args,**kwargs)
        self.buildOOBNetwork()

    def buildOOBNetwork(self):
        sw = self.addSwitch("sw-mon", cls=LinuxBridge, dpid=self.dpidToStr())

        # Connect each router to the monitoring switch
        routers = self.routers()
        for router in routers:
            self.addLink(router, sw, intf=TCIntf, intfName1="%s-mon"%router, cost=-1)

