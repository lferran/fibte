import argparse
from fibbingnode import CFG

import fibbingnode.misc.mininetlib as _lib
from fibbingnode.misc.mininetlib.cli import FibbingCLI
from fibbingnode.misc.mininetlib.ipnet import IPNet, TopologyDB
from fibbingnode.misc.mininetlib.iptopo import IPTopo
from fattree import FatTree, FatTreeOOB

from fibbingnode.algorithms.southbound_interface import SouthboundManager
from fibbingnode.algorithms.ospf_simple import OSPFSimple

from mininet.clean import cleanup, sh
from mininet.util import custom
from mininet.link import TCIntf
from fibte.misc.DCTCInterface import DCTCIntf
import fibte.res.config as cfg
import signal

from fibte import flowServer_path

from fibte.trafficgen import setup_alias

def signal_term_handler(signal, frame):
    import sys
    sys.exit(0)

class TestTopo(IPTopo):
    def build(self, *args, **kwargs):
        """
        """
        r1 = self.addRouter('r1')
        r2 = self.addRouter('r2')
        r3 = self.addRouter('r3')
        r4 = self.addRouter('r4')
        r5 = self.addRouter('r5')

        # Short path
        self.addLink(r1, r3)
        self.addLink(r3, r5)
        # Long path
        self.addLink(r1, r2)
        self.addLink(r2, r4)
        self.addLink(r4, r5)

        s1 = self.addHost('s1')
        d1 = self.addHost('d1')
        
        self.addLink(s1, r1)
        self.addLink(d1, r5)
        
        # Adding Fibbing Controller
        c1 = self.addController(cfg.C1, cfg_path=cfg.C1_cfg)
        self.addLink(c1, r1, cost = 1000)

def launch_network(k=4, bw=10, ip_alias=True):
    signal.signal(signal.SIGTERM, signal_term_handler)

    # Cleanup the network
    cleanup()
    sh("killall snmpd ospfd zebra pmacctd getLoads.py")

    # Topology
    #topo = FatTree(k=k, sflow=False, extraSwitch=False, bw=bw)

    #topo = FatTreeOOB(k=k, sflow=False, extraSwitch=False, bw=bw)
    topo = TestTopo()
    
    # Interfaces
    intf = custom(TCIntf, bw=bw)  # , max_queue_size=1000)
    #intf = custom(DCTCIntf, bw=bw, ip_alias=True)

    # Network
    net = IPNet(topo=topo, debug=_lib.DEBUG_FLAG, intf=intf)

    # Save the TopoDB object
    TopologyDB(net=net).save(cfg.DB_path)

    # Start the network
    net.start()

    print('*** Starting Flow Servers in virtualized hosts')
    for h in net.hosts:
        # Start flowServers
        h.cmd(flowServer_path + " {0} &".format(h.name))
        print(h.name)

    if ip_alias == True:
        print('*** Setting up ip alias for elephant traffic - alias identifier: .222')
        for h in net.hosts:
            # Setup alias at host h
            setup_alias(h)
            
    # Start the Fibbing CLI
    FibbingCLI(net)

    net.stop()

def launch_controller():
    CFG.read(cfg.C1_cfg)
    db = TopologyDB(db=cfg.DB_path)
    manager = SouthboundManager(optimizer=OSPFSimple())
    #manager.simple_path_requirement(db.subnet(R3, D1), [db.routerid(r)
    #                                                    for r in (R1, R2, R3)])
    #manager.simple_path_requirement(db.subnet(R3, D2), [db.routerid(r)
    #                                                 for r in (R1, R4, R3)])
    try:
        manager.run()
    except KeyboardInterrupt:
        manager.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--controller',
                       help='Start the controller',
                       action='store_true',
                       default=False)
    group.add_argument('-n', '--net',
                       help='Start the Mininet topology',
                       action='store_true',
                       default=True)
    parser.add_argument('-d', '--debug',
                        help='Set log levels to debug',
                        action='store_true',
                        default=False)

    parser.add_argument('-k', help='Launch k-ary fat-tree network',
                        type=int, default=4)

    args = parser.parse_args()
    if args.debug:
        _lib.DEBUG_FLAG = True
        from mininet.log import lg
        from fibbingnode import log
        import logging
        log.setLevel(logging.DEBUG)
        lg.setLogLevel('debug')
    if args.controller:
        launch_controller()
    elif args.net:
        launch_network(k=args.k)
        
