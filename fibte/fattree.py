from fibbingnode.misc.mininetlib.iptopo import IPTopo

from mininet.nodelib import LinuxBridge
from mininet.util import custom
from mininet.link import TCIntf
import fibte.res.config as cfg

class FatTree(IPTopo):
    def __init__(self, k=4, sflow=False, extraSwitch=False, bw = 10, *args, **kwargs):
        # k must be multiple of 2
        self.k = k

        # datapath-id for the switches, if you dont specify mininet assign them duplicated
        self.dpid = 1
        self.extraSwitch = extraSwitch
        self.sflow = sflow

        # Not used yet, when I do have time i should implement the part to choose the ips
        self.ip_bindings = {}
        self.ipBase = "10.0.0.0/8"
        self.max_alloc_prefixlen=24

        # For the switch bandwidth
        self.bw = bw
        if not(bw):
            self.extraSwitchBandwidth = None
        else:
            self.extraSwitchBandwidth = bw * (k/2 + 1)

        super(FatTree,self).__init__(*args,**kwargs)

    def dpidToStr(self):
        strDpid = format(self.dpid,'x')
        if len(strDpid) < 16:
            return "0"*(16-len(strDpid)) + strDpid
        return strDpid

    def build(self, *args, **kwargs):
        # Build the topology

        # Create the pods first
        aggregationRouters = self.addPods(self.extraSwitch)
        # Create the core routers
        coreRouters = self.addCoreRouters()
        # Connect them together
        self.connectCoreAggregation(aggregationRouters, coreRouters)

        # Add Fibbing Controller to one of the core routers
        fibbingController = self.addController(cfg.C1, cfg_path=cfg.C1_cfg)
        self.addLink(fibbingController, coreRouters[0], cost=10000)

        # Add Fibbing Load Balancer host to core router 0
        fibingLB = self.addHost('c2', fibte=True)
        self.addLink(fibingLB, coreRouters[0])

    def connectCoreAggregation(self, aggregationRouters, coreRouters):
        # Connect every aggregation router with k/2 core routers
        for i, aggregationRouter in enumerate(aggregationRouters):
            # Position inside the pod
            position = i % (self.k/2)

            # Connect with core routers
            for coreRouter in coreRouters[(position*(self.k/2)):((position+1)*(self.k/2))]:
                self.addLink(aggregationRouter, coreRouter)#,bw = self.bw)

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

    def addHostsGrup(self, podNum, startIndex, extraSwitch=True):
        """
        :param podNum:
        :param startIndex:
        :param extraSwitch:
        :return:
        """
        # Contains the name of the switches. They will be used to connect the hosts to the higher layers
        switches = []

        if extraSwitch:
            # First a switch to grup all the hosts is created
            switch_index = startIndex/(self.k/2)
            sw = self.addSwitch("sw_%d_%d" % (podNum, switch_index),cls=LinuxBridge, dpid = self.dpidToStr())
            #sw = self.addSwitch("sw_%d_%d" % (podNum, switch_index),cls=LinuxBridge)
            self.dpid +=1

            #creatres k/2 OVSHosts
            for i in range(self.k/2):
                ovsHosts = self.addOVSHost(podNum, startIndex+i)
                #add link between the ovs switch and the normal switch
                self.addLink(ovsHosts["ovs"], sw)#,bw = self.bw)
            #we add sw in switches list. Switches list is used as a return value,
            #  and will be used later to know what needs to be connected with the edge router
            switches.append(sw)
        # Case in which all the hosts are directly connected to the edge router
        else:
            for i in range(self.k/2):
                ovsHosts = self.addOVSHost(podNum, startIndex+i)
                switches.append(ovsHosts["ovs"])

        return switches

    def addPods(self, extraSwitch=True):
        aggregationRouters = []

        # Add k pods and store the aggregation Routers in aggregationRouters
        for i in range(self.k):
            aggregationRouters += (self.addPod(i, extraSwitch))

        return aggregationRouters

    def addPod(self, podNum, extraSwitch=True):
        """
        :param podNum:
        :param extraSwitch:
        :return:
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
            # Create hosts and switches
            switches = self.addHostsGrup(podNum, startIndex, extraSwitch)

            # Connect switch/switches with edge router
            for switch in switches:
                if extraSwitch:
                    self.addLink(edge_router, switch, intf = custom(TCIntf, bw=self.extraSwitchBandwidth))
                else:
                    self.addLink(edge_router, switch)#,bw = self.bw)
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

