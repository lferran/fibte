from mininet.link import TCIntf, Intf
from mininet.log import info, error, debug
from fibte import iptables_path

class TCIntf2(TCIntf):

    #@time_func
    def config(self, *args,**kwargs):
        super(TCIntf2,self).config(*args,**kwargs)

class DCTCIntf(TCIntf):
    """
    Customized TC class for the Datacenter network. It will have a special TC tree
    designed for our purposes
    """
    def __init__(self, *args,**kwargs):
        super(DCTCIntf,self).__init__(*args,**kwargs)

    def bwCmds(self, bw=None, speedup=0, use_hfsc=False, use_tbf=False,
               latency_ms=None, enable_ecn=False, enable_red=False):
        "Return tc commands to set bandwidth"

        cmds, parent = [], ' root '

        if bw and (bw < 0 or bw > self.bwParamMax):
            error('Bandwidth limit', bw, 'is outside supported range 0..%d'
                  % self.bwParamMax, '- ignoring\n')

        elif bw is not None:
            # BL: this seems a bit brittle...
            if (speedup > 0 and self.node.name[0:1] == 's'):
                bw = speedup
            # This may not be correct - we should look more closely
            # at the semantics of burst (and cburst) to make sure we
            # are specifying the correct sizes. For now I have used
            # the same settings we had in the mininet-hifi code.

            # Own interface implementation starts here --------------------------------------------

            # Description:
            #    - Root qdisc: hierarchical token bucket with 3 classes (one for each priority)
            #    - Root class 1:1
            #    - Class 1:10 for priority 1 traffic (ICMP): is reserved 20% of the bw
            #    - Class 1:20 for priority 2 traffic (OSPF, SYN + ACK): is reserved 10% of the bw
            #    - Class 1:30 for other traffic: is reserved 70% of the bw
            #    - At the end of class 1: a Packet-length-measured FIFO
            #    - At the end of classes 2 and 3: a SFQ is attached

            # Add htb qdisc as a root and class
            cmds += ['%s qdisc add dev %s root handle 1: htb default 30',
                     '%s class add dev %s parent 1: classid 1:1 htb ' +
                     'rate %fMbit burst 15k' % bw]

            parent = ' parent 1:1'

            # We add 2 htb subclasses
            prio1_bw = 0.2 * bw
            prio2_bw = 0.1 * bw
            prio3_bw = 0.7 * bw

            cmds += ['%s class add dev %s'+ parent +
                     ' classid 1:10 htb ' + 'rate %fMbit ceil %fmbit prio 1 burst 15k' % (prio1_bw, bw),
                     '%s class add dev %s' + parent +
                     ' classid 1:20 htb ' + 'rate %fMbit ceil %fmbit prio 2 burst 15k' % (prio2_bw, bw),
                     '%s class add dev %s' + parent +
                     ' classid 1:30 htb ' +'rate %fMbit ceil %fmbit prio 3 burst 15k' % (prio3_bw, bw)]

            parent1 = 'parent 1:10'
            parent2 = 'parent 1:20'
            parent3 = 'parent 1:30'

            # We add the final SFQ leaf-queues here
            cmds +=['%s qdisc add dev %s '+parent1 +
                    ' handle 10: pfifo limit 1000',
                    '%s qdisc add dev %s '+parent2 +
                    ' handle 20: sfq perturb 10 limit 64 quantum 10000',
                    '%s qdisc add dev %s '+parent3 +
                    ' handle 30: sfq perturb 10 limit 64 quantum 10000']

            parent = ''

        return cmds, parent

    @staticmethod
    def delayCmds(parent, delay=None, jitter=None, loss=None, max_queue_size=None):
        "Internal method: return tc commands for delay and loss"
        cmds = []

        # we can not do anything since we attach classless qdiscs in our previous command
        return cmds, parent

    @staticmethod
    def filterCmds(parent="parent 1:0", prio=1, handle=10, flowid="1:10"):
        """Command to filter and classify packets according to priorities"""
        return ['%s filter add dev %s protocol ip '+parent+' prio %d handle %d fw flowid %s'%(prio, handle, flowid)]

    def tc(self, cmd, tc='tc'):
        "Execute tc command for our interface"
        c = cmd % (tc, self)  # Add in tc command and our name
        debug(" *** executing command: %s\n" % c)
        return self.cmd(c)

    def config(self, bw=None, delay=None, jitter=None, loss=None,
               disable_gro=True, speedup=0, use_hfsc=False, use_tbf=False,
               latency_ms=None, enable_ecn=False, enable_red=False,
               max_queue_size=None, **params):
        """Configure the port and set its properties."""
        result = Intf.config(self, **params)

        # Disable GRO
        if disable_gro:
            self.cmd('ethtool -K %s gro off' % self)

        # Optimization: return if nothing else to configure
        # Question: what happens if we want to reset things?
        if (bw is None and not delay and not loss
            and max_queue_size is None):
            return

        # Clear existing configuration
        tcoutput = self.tc('%s qdisc show dev %s')
        if "priomap" not in tcoutput and "noqueue" not in tcoutput:
            cmds = ['%s qdisc del dev %s root']
        else:
            cmds = []

        # Bandwidth limits via various methods
        bwcmds, parent = self.bwCmds(bw=bw, speedup=speedup,
                                     use_hfsc=use_hfsc, use_tbf=use_tbf,
                                     latency_ms=latency_ms,
                                     enable_ecn=enable_ecn,
                                     enable_red=enable_red)
        cmds += bwcmds

        # Delay/jitter/loss/max_queue_size using netem
        delaycmds, parent = self.delayCmds(delay=delay, jitter=jitter,
                                           loss=loss,
                                           max_queue_size=max_queue_size,
                                           parent=parent)
        cmds += delaycmds

        # We add the filter for the HTB class
        filtercmds = self.filterCmds(parent="parent 1:0", prio=1, handle=10, flowid="1:10")
        cmds += filtercmds

        filtercmds = self.filterCmds(parent="parent 1:0", prio=2, handle=20, flowid="1:20")
        cmds += filtercmds

        # Ugly but functional: display configuration info
        stuff = ((['%.2fMbit' % bw] if bw is not None else []) +
                 (['%s delay' % delay] if delay is not None else []) +
                 (['%s jitter' % jitter] if jitter is not None else []) +
                 (['%.5f%% loss' % loss] if loss is not None else []) +
                 (['ECN'] if enable_ecn else ['RED']
                 if enable_red else []))
        info('(' + ' '.join(stuff) + ') ')

        # Execute all the commands in our node
        debug("at map stage w/cmds: %s\n" % cmds)
        # for cmd in cmds:
        #     print cmd
        tcoutputs = [self.tc(cmd) for cmd in cmds]
        # if "r_0_e0" in self.name:
        #     for cmd in cmds:
        #         print cmd

        for output in tcoutputs:
            if output != '':
                error("*** Error: %s" % output)
        debug("cmds:", cmds, '\n')
        debug("outputs:", tcoutputs, '\n')

        # add marks to packets through IPTABLES RULES
        # OSPF: marks ospf packets with priority 0x14 (20)
        #self.cmd("iptables -w -t mangle -A POSTROUTING -o %s -p 89 -j MARK --set-mark 20" % self.name)

        # ICMP: marks icmp packets with priority 0xa (10)
        #self.cmd("iptables -w -t mangle -A POSTROUTING -o %s -p 1 -j MARK --set-mark 10" % self.name)

        # Traceroute: marks pakets with TTL < 10 that are not OSPF with priority 0xa (10)
        #self.cmd("iptables -w -t mangle -A POSTROUTING -o %s -m ttl --ttl-lt 10 ! -p 89 -j MARK --set-mark 10" % self.name)

        # TCP flags: Gives priority 20 to TCP SYN and ACK packets
        #self.cmd("iptables -w -t mangle -A POSTROUTING -o %s -m ttl --ttl-gt 10 -p tcp --tcp-flags ACK ACK -m length --length :64 -j MARK --set-mark 20"% self.name)
        #self.cmd("iptables -w -t mangle -A POSTROUTING -o %s -m ttl --ttl-gt 10 -p tcp --tcp-flags SYN SYN -m length --length :64 -j MARK --set-mark 20" % self.name)

        self.cmd(iptables_path + " {0} &".format(self.name))

        result['tcoutputs'] = tcoutputs
        result['parent'] = parent

        return result