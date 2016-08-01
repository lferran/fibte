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
from fibte.res.mycustomhost import MyCustomHost
import fibte.res.config as cfg
import signal

from fibte import flowServer_path

def signal_term_handler(signal, frame):
    import sys
    sys.exit(0)

def launch_network(k = 4, bw=10):
    signal.signal(signal.SIGTERM, signal_term_handler)

    # Cleanup the network
    cleanup()
    sh("killall snmpd ospfd zebra pmacctd")

    # Topology
    topo = FatTreeOOB(k=4, sflow=False, extraSwitch=False, bw=bw)

    # Interfaces
    intf = custom(TCIntf, bw=bw)#, max_queue_size=1000)

    # Network
    net = IPNet(topo = topo, debug = _lib.DEBUG_FLAG, intf = intf, host = MyCustomHost)

    # Save the TopoDB object
    TopologyDB(net=net).save(cfg.DB_path)
    
    # Start the network
    net.start()

    print('*** Starting Flow Servers in virtualized hosts')
    #import ipdb; ipdb.set_trace()
    for h in net.hosts:
        # start flowServer
        h.cmd(flowServer_path + " {0} &".format(h.name))

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
    #                                                    for r in (R1, R4, R3)])
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
        launch_network()
