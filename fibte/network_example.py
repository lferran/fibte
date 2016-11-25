import argparse
import signal
import subprocess

from fibbingnode import CFG
from fibbingnode.misc.mininetlib.cli import FibbingCLI
from fibbingnode.misc.mininetlib.ipnet import IPNet, TopologyDB
from fibbingnode.misc.mininetlib.iptopo import IPTopo
from fibbingnode.algorithms.southbound_interface import SouthboundManager
from fibbingnode.algorithms.ospf_simple import OSPFSimple
import fibbingnode.misc.mininetlib as _lib
from fibbingnode.misc.utils import read_pid, del_file

from mininet.clean import cleanup, sh
from mininet.util import custom
from mininet.link import TCIntf

from fibte.fattree import FatTree, FatTreeOOB
from fibte.misc.DCTCInterface import DCTCIntf
from fibte.misc.ipalias import setup_alias
import fibte.res.config as cfg
from fibte import counterCollector_path, ifDescrNamespace_path, tmp_files

def signal_term_handler(signal, frame):
    import sys
    sys.exit(0)

class TestTopo1(IPTopo):
    def build(self, *args, **kwargs):
        """
        Used to test Fibbing in longer prefixes
                 ___ r3______
               /             \
        s1-- r1               r5 --d1
              \____r2____r4__/
        """
        r1 = self.addRouter('r1')
        r2 = self.addRouter('r2')
        r3 = self.addRouter('r3')
        r4 = self.addRouter('r4')
        r5 = self.addRouter('r5')

        # Short (default) path
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

class TestTopo2(IPTopo):
    def build(self, *args, **kwargs):
        """Used to test traffic shaping and packet drops
        """
        r1 = self.addRouter('r1')
        r2 = self.addRouter('r2')

        # Short path
        self.addLink(r1, r2)

        s1 = self.addHost('s1')
        s2 = self.addHost('s2')
        d1 = self.addHost('d1')
        d2 = self.addHost('d2')

        self.addLink(s1, r1)
        self.addLink(s2, r1)

        self.addLink(d1, r2)
        self.addLink(d2, r2)

        # Adding Fibbing Controller
        c1 = self.addController(cfg.C1, cfg_path=cfg.C1_cfg)
        self.addLink(c1, r1, cost=1000)

class TestTopo3(IPTopo):
    def build(self, *args, **kwargs):
        """
        Used to test Fibbing in longer prefixes
                     r6
                      | \
                      |  \______
                 ___ r3___ ___  |
               /             \  |
        s1-- r1---------------r5 --d1
              \____r2____r4__/
        """
        r1 = self.addRouter('r1')
        r2 = self.addRouter('r2')
        r3 = self.addRouter('r3')
        r4 = self.addRouter('r4')
        r5 = self.addRouter('r5')
        r6 = self.addRouter('r6')

        # Add links to make ECMP in the three paths
        self.addLink(r1, r3, cost=3)
        self.addLink(r3, r5, cost=3)

        self.addLink(r1, r5, cost=6)

        self.addLink(r1, r2, cost=2)
        self.addLink(r2, r4, cost=2)
        self.addLink(r4, r5, cost=2)

        self.addLink(r6, r3, cost=2)
        self.addLink(r6, r5, cost=2)

        s1 = self.addHost('s1')
        d1 = self.addHost('d1')

        self.addLink(s1, r1)
        self.addLink(d1, r5)

        # Adding Fibbing Controller
        c1 = self.addController(cfg.C1, cfg_path=cfg.C1_cfg)
        self.addLink(c1, r1, cost=1000)

def startCounterCollectors(topo, interval=1):
    """"""
    print("*** Starting counterCollectors")

    cc_cmd = "mx {0} {2} -n {0} -t {1} &"
    if_cmd = "mx {0} {1} {2} &"
    for rid in topo.routers():
        # Fetch interface information first
        itfDescr_path = "{1}ifDescr_{0}".format(rid, tmp_files)
        subprocess.call(if_cmd.format(rid, ifDescrNamespace_path, itfDescr_path), shell=True)

        # Start counter collector
        subprocess.call(cc_cmd.format(rid, interval, counterCollector_path), shell=True)

def stopCounterCollectors(topo):
    """"""
    print("*** Stopping counterCollectors")

    for rid in topo.routers():
        # Stop counters process
        pid = read_pid("/tmp/load_{0}.pid".format(rid))

        if pid:
            subprocess.call(['kill', '-9', pid])

        # Erase load writen by the process
        del_file("/tmp/load_{0}.pid".format(rid))
        del_file("/tmp/load_{0}".format(rid))
        del_file("/tmp/load_{0}_tmp".format(rid))

        # Delete ifDescript file
        path = "/tmp/ifDescr_{0}".format(rid)
        del_file(path)

def setupSecondaryIps(net):
    print('*** Setting up secondary ips at hosts')
    for host in net.hosts:
        setup_alias(host)

def launch_network(k=4, bw=10, ip_alias=True):
    signal.signal(signal.SIGTERM, signal_term_handler)

    # Cleanup the network
    cleanup()
    sh("killall ospfd zebra getLoads.py nc")

    # Remove tmp files and old namespaces
    subprocess.call(['rm', '/tmp/*.log', '/tmp/*.pid', '/tmp/mice*'])
    subprocess.call(['rm', '/var/run/netns/*'])

    # Flush root namespace mangle table
    subprocess.call(["iptables", "-t", "mangle" ,"-F"])
    subprocess.call(["iptables", "-t", "mangle" ,"-X"])

    # Topology
    topo = FatTree(k=k, sflow=False, ovs_switches=False)

    # Interfaces
    intf = custom(DCTCIntf, bw=bw)
    #intf = custom(TCIntf, bw=bw)

    # Network
    net = IPNet(topo=topo, debug=_lib.DEBUG_FLAG, intf=intf)

    # Save the TopoDB object
    TopologyDB(net=net).save(cfg.DB_path)

    # Start the network
    net.start()

    startCounterCollectors(topo, interval=1)

    if ip_alias:
        setupSecondaryIps(net)

    # Start the Fibbing CLI
    FibbingCLI(net)

    net.stop()

    stopCounterCollectors(topo)

def launch_controller():
    CFG.read(cfg.C1_cfg)
    db = TopologyDB(db=cfg.DB_path)
    manager = SouthboundManager(optimizer=OSPFSimple())
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

    parser.add_argument('--ip_alias', help='Configure ip alias if argument is present',
                        action="store_true", default=True)

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
        launch_network(k=args.k, ip_alias=args.ip_alias)
