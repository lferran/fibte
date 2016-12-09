import random
import time
import os
import argparse
import abc
import json
import scipy.stats as stats
import numpy as np
import copy
try:
    import cPickle as pickle
except:
    import pickle
import ast

from fibte.misc.topology_graph import TopologyGraph
from fibte.misc.unixSockets import UnixClient, UnixClientTCP
from fibte import tmp_files, db_topo, CFG, LINK_BANDWIDTH, MICE_SIZE_RANGE, ELEPHANT_SIZE_RANGE, MICE_SIZE_STEP, ELEPHANT_SIZE_STEP
from fibte.logger import log
from fibte.trafficgen.flow import Flow, Base
from fibte.trafficgen import nonNICCongestionTest

import logging

controllerServer = CFG.get("DEFAULT","controller_UDS_name")

MIN_PORT = 1001
MIN_MICE_PORT = MIN_PORT
MAX_MICE_PORT = MIN_MICE_PORT + 1000
MAX_PORT = 2**16 -1
MiceRangePorts = xrange(MIN_PORT, MAX_MICE_PORT)
ElephantRangePorts = xrange(MAX_MICE_PORT, MAX_PORT)

SAVED_TRAFFIC_DIR = os.path.join(os.path.dirname(__file__), 'saved_traffic/')

def time_func(function):
    def wrapper(*args,**kwargs):
        t = time.time()
        res = function(*args,**kwargs)
        log.info("{0} took {1}s to execute".format(function.func_name, time.time()-t))
        return res
    return wrapper

def read_pid(n):
    """
    Extract a pid from a file
    :param n: path to a file
    :return: pid as a string
    """
    with open(n, 'r') as f:
        return str(f.read()).strip(' \n\t')

def del_file(f):
    os.remove(f)

class TGParser(object):
    def __init__(self):
        # Start parser
        self.parser = argparse.ArgumentParser()

    def loadParser(self):
        self.parser.add_argument('--addtc', help='Add flows to current traffic --it doesnt reset the controller', action="store_true", default=False)
        self.parser.add_argument('--pattern', help='Communication pattern', choices=['random','staggered','bijection','stride'], type=str, default='random')
        self.parser.add_argument('--pattern_args', help='Communication pattern arguments', type=str, default="{}")
        self.parser.add_argument('--mice_avg', help="Specifiy the average for the poisson arrival process for mice flows", type=float, default=0.0)
        self.parser.add_argument('-t', '--time', help='Duration of the traffic generator', type=int, default=120)
        self.parser.add_argument('-s', '--time_step', help="Granularity at which we inspect the generated traffic so that the rates are kept", type=int, default=1)
        self.parser.add_argument('--save_traffic', help='Saves traffic in a file so it can be repeated',action="store_true")
        self.parser.add_argument('--file_name', help='Specify the filename for the traffic file', action="store_true", default=False)
        self.parser.add_argument('--load_traffic', help='Load traffic from a file so it can be repeated', default="")
        self.parser.add_argument('--flows_file', help="Schedule the flows specified in file", default=False)
        self.parser.add_argument('--terminate', help='Terminate any ongoing traffic', action='store_true')

    def parseArgs(self):
        # Parse the arguments and return them
        args = self.parser.parse_args()
        return args

    @time_func
    def runArguments(self, tgf, args):
        """Given a traffic generator object and the
         arguments, act accordingly
        """
        # Start counting time
        if args.terminate:
            print "Terminating ongoing traffic!"
            tgf.terminateTrafficAtHosts()
            time.sleep(2)
            tgf.resetController()

        else:
            # Check if flow file has been given
            if not args.flows_file:
                # If traffic must be loaded
                if args.load_traffic:
                    elephants_per_host, mice_bijections, mice_average = tgf.loadTraffic(args.load_traffic)

                else:
                    print("Generating elephant traffic ...")
                    # Generate traffic
                    elephants_per_host = tgf.plan_elephant_flows()

                    mice_average = args.mice_avg

                    if args.mice_avg != 0.0:
                        print("Generating mice traffic ...")
                        mice_bijections = tgf.calculate_mice_port_bijections()
                    else:
                        mice_bijections = None

                    # If it must be saved
                    if args.save_traffic:
                        if not args.file_name:
                            filename = tgf.get_filename()
                        else:
                            filename = args.file_name

                        # Save it
                        tgf.saveTraffic(filename, elephants_per_host, mice_bijections, mice_average)

                # Orchestrate the traffic (either loaded or generated)
                print("Scheduling traffic...")
                tgf.schedule(elephants_per_host, mice_bijections, mice_average)

            # Flow file has been given
            else:
                print("Scheduling elephant flows specified in TEST file: {0}".format(args.flows_file))
                # Generate traffic from flows file
                elephants_per_host = tgf.plan_from_flows_file(args.flows_file)

                # Schedule it
                tgf.schedule(elephants_per_host)

    def printArgs(self, args):
        import ipdb; ipdb.set_trace()

class udpTrafficGeneratorBase(Base):
    def __init__(self, pattern='random', pattern_args="{}", mice_avg=1, totalTime=100, timeStep=1, *args, **kwargs):
        super(udpTrafficGeneratorBase, self).__init__(*args, **kwargs)

        # Fodler where we store the traffic files
        self.saved_traffic_dir = SAVED_TRAFFIC_DIR

        # Link bandwidth
        self.linkBandwidth = LINK_BANDWIDTH

        # Get attributes
        self.pattern = pattern
        self.pattern_args = ast.literal_eval(pattern_args)

        # Get mice average
        self.mice_avg = mice_avg

        self.totalTime = totalTime
        self.timeStep = timeStep

        # Load topology
        self.topology = TopologyGraph(getIfindexes=False, interfaceToRouterName=False, db=os.path.join(tmp_files, db_topo))

        # Used to communicate with flowServers at the hosts.
        self.unixClient = UnixClientTCP(tmp_files+"flowServer_{0}")

        # Used to communicate with LoadBalancer Controller
        self.ControllerClient = UnixClientTCP(os.path.join(tmp_files, controllerServer))

        # Get sender hosts
        self.senders = self.topology.getHosts().keys()
        # Shuffle them
        random.shuffle(self.senders)

        self.possible_destinations = self._createPossibleDestinations()

        log.setLevel(logging.INFO)

    @staticmethod
    def get_poisson_times(average, totalTime):
        """
        Returns a list of Poisson arrival process times ranging
        from zero to totalTime with a certain average

        :param average: indicates how often new flows are started at the host.
        :param totalTime: total time of the simulation
        :return: list of flow start times drawn from a poisson arrival process
        """
        absolute_times = []
        time_index = 0
        while (time_index < totalTime):
            # Generate starting time for next flow from Poisson distribution
            next_flow_time = random.expovariate(average) + time_index

            # Stop generating more flow times if we reached the end of the simulation
            if next_flow_time >= totalTime:
                break

            else:
                # Append it to the absolute times
                absolute_times.append(next_flow_time)

                # Update time_index
                time_index = next_flow_time

        return absolute_times

    @staticmethod
    def weighted_choice(choices):
        """
        *args example: [("e",weight_e),('m',weight_m)]
        """
        total = sum(w for c, w in choices)
        r = random.uniform(0, total)
        upto = 0
        for c, w in choices:
            if upto + w >= r:
                return c
            upto += w
        assert False, "Shouldn't get here"

    @abc.abstractmethod
    def get_filename(self):
        """Returns the filename to save the traffic to"""

    def get_pattern_args_filename(self):
        if self.pattern == 'random':
            return None

        elif self.pattern == 'staggered':
            sameEdge = self.pattern_args.get('sameEdge')
            samePod = self.pattern_args.get('samePod')
            return "se{0}sp{1}".format(sameEdge, samePod)

        elif self.pattern == 'bijection':
            return None

        elif self.pattern == 'stride':
            i = self.pattern_args.get('i')
            return "i{0}".format(i)

    @staticmethod
    def get_flow_duration(flow_type):
        """
        Makes a choice on the flow duration, depending on the flow_type
        :param flow_type: 'm' (mice) or 'e' (elephant)

        :return: integer representing the flow duration in seconds
        """
        # Flow duration ranges
        min_len_elephant = 20
        max_len_elephant = 50#500
        min_len_mice = 0.2
        max_len_mice = 6

        # flow is elephant
        if flow_type == 'e':
            # Draw flow duration
            return random.uniform(min_len_elephant, max_len_elephant)

        # flow is mice
        elif flow_type == 'm':
            # Draw flow duration
            #return random.uniform(min_len_mice, max_len_mice)
            mean = 4.0
            ep = random.expovariate(1/mean)
            while ep > max_len_mice:
                ep = random.expovariate(1 / mean)
            return ep
        else:
            raise ValueError("Unknown flow type: {0}".format(flow_type))

    @staticmethod
    def get_flow_size(flow_type, distribution='exponential'):
        """
        :param flow_type:
        :return:
        """
        if flow_type == 'e':
            if distribution == 'uniform':
                return random.randrange(ELEPHANT_SIZE_RANGE[0], ELEPHANT_SIZE_RANGE[1] + ELEPHANT_SIZE_STEP, ELEPHANT_SIZE_STEP)

            elif distribution == 'exponential':
                size_range = range(int(ELEPHANT_SIZE_RANGE[0]), int(ELEPHANT_SIZE_RANGE[1] + ELEPHANT_SIZE_STEP), int(ELEPHANT_SIZE_STEP))
                # Draw exponential sample
                point = stats.expon.rvs(scale=2, size=1)
                point_index = int(round(point))
                point_index = max(0, point_index)
                point_index = min(point_index, len(size_range) - 1)
                return size_range[point_index]

            elif distribution == 'constant':
                return int(ELEPHANT_SIZE_RANGE[0])

        elif flow_type == 'm':
            #return random.randrange(MICE_SIZE_RANGE[0], MICE_SIZE_RANGE[1]+MICE_SIZE_STEP, MICE_SIZE_STEP)
            return random.uniform(MICE_SIZE_RANGE[0], MICE_SIZE_RANGE[1])
        else:
            raise ValueError("Unknown flow type: {0}".format(flow_type))

    def get_mice_size(self, proto='tcp'):
        if proto.lower() == 'tcp':
            size = random.expovariate(1 / 12.0) * 10e6
            rate = LINK_BANDWIDTH
            duration = size / float(rate)
            while duration > 20:
                size = random.expovariate(1 / 12.0) * 10e6
                rate = LINK_BANDWIDTH
                duration = size / float(rate)
            return size
        elif proto.lower() == 'udp':
            return random.uniform(MICE_SIZE_RANGE[0], MICE_SIZE_RANGE[1])

    def get_elephant_size(self, proto='tcp'):
        if proto.lower() == 'tcp':
            size = random.expovariate(1 / 12.0) * 10e6
            rate = LINK_BANDWIDTH
            duration = size / float(rate)
            while duration < 20:
                size = random.expovariate(1 / 12.0) * 10e6
                rate = LINK_BANDWIDTH
                duration = size / float(rate)
            return size

        elif proto.lower() == 'udp':
            size_range = range(int(ELEPHANT_SIZE_RANGE[0]), int(ELEPHANT_SIZE_RANGE[1] + ELEPHANT_SIZE_STEP),
                               int(ELEPHANT_SIZE_STEP))
            # Draw exponential sample
            point = stats.expon.rvs(scale=2, size=1)
            point_index = int(round(point))
            point_index = max(0, point_index)
            point_index = min(point_index, len(size_range) - 1)
            return size_range[point_index]

    @staticmethod
    def _getFlowEndTime(flow):
        if flow['proto'].lower()  == 'udp':
            return flow.get('startTime') + flow.get('duration')

        elif flow['proto'].lower() == 'tcp':
            duration = flow.get('size') / float(flow.get('rate'))
            return flow.get('startTime') + duration
        else:
            raise ValueError("Wrong flow type")

    def getTCPFlowRemainingTime(self, flow):
        """
        Assumes current rate as the stable rate until flow finishes
        """
        if flow['proto'].lower() == 'tcp':
            remaining_data = flow.get('remaining', flow.get('size'))
            remaining_time = remaining_data / float(flow.get('rate'))
            return remaining_time
        else:
            raise ValueError("Wrong flow type")

    def _createPossibleDestinations(self):
        """"""
        pos_dst = {}
        rcvs = self.senders[:]
        if self.pattern == 'random':
            for sender in self.senders:
                # Filter sender and hosts with same network
                rcvs_tmp = [r for r in rcvs if r != sender and not self.topology.inSameSubnetwork(r, sender)]
                pos_dst[sender] = rcvs_tmp

        elif self.pattern == 'staggered':
            for sender in self.senders:
                # Same edge
                rcvs_edge = [r for r in rcvs if r != sender and self.topology.inSameSubnetwork(r, sender)]

                # Get host pod
                hpod = self.topology.getHostPod(sender)

                # Get hosts behind pod
                rcvs_pod = self.topology.getHostsBehindPod(hpod)

                # Filter sender and hosts with same network
                rcvs_pod = [r for r in rcvs_pod if r != sender and r not in rcvs_edge]

                # Compute receivers outside pod
                rcvs_outside = [r for r in rcvs if self.topology.getHostPod(r) != hpod]

                pos_dst[sender] = {'sameEdge': rcvs_edge, 'samePod': rcvs_pod, 'otherPod': rcvs_outside}

        elif self.pattern == 'bijection':
            # Create random pairs
            random_hosts_list = rcvs[:]
            random.shuffle(random_hosts_list)
            for index, h in enumerate(random_hosts_list):
                r_index = (index + 1) % len(random_hosts_list)
                r = random_hosts_list[r_index]
                pos_dst[h] = r

        elif self.pattern == 'stride':
            # Get pattern args
            stride_i = self.pattern_args.get('i', 4)

            # Get ordered list of hosts
            orderedHosts = self.topology.sortHostsByName(rcvs)

            for index, h in enumerate(orderedHosts):
                r_index = (index + stride_i) % len(orderedHosts)
                r = orderedHosts[r_index]
                pos_dst[h] = r

        return pos_dst

    def get_possible_destinations(self, sender):
        return self.possible_destinations[sender]

    def get_flow_destination(self, sender, exclude=[]):
        """
        This method abstracts the choice of a receiver. It chooses a
        receiver depending on the communication pattern.

        :param sender:
        :return: chosen receiver
        """
        # Get the pre-computed possible destinations for the sender
        receivers = self.get_possible_destinations(sender)

        if nonNICCongestionTest:
            if isinstance(receivers, list):
                receivers = [r for r in receivers if '0' in r[-1] or '2' in r[-1]]

        if self.pattern == 'random':
            receivers = list(set(receivers) - set(exclude))
            if receivers:
                # Pick one at random
                return random.choice(receivers)
            else:
                return None

        elif self.pattern == 'stride':
            receivers = list({receivers} - set(exclude))
            if len(receivers) == 1:
                return receivers[0]
            elif len(receivers) == 0:
                return None
            else:
                print("ERROR: Stride can only return 1 receiver")
                import ipdb; ipdb.set_trace()

        elif self.pattern == 'staggered':
            # Extract probabilites from pattern params
            sameEdge_p = self.pattern_args.get('ep', 0.1)
            samePod_p = self.pattern_args.get('pp', 0.2)
            otherPod_p = max(0, 1 - sameEdge_p - samePod_p)

            # Make a weighted choice
            choice = self.weighted_choice([('sameEdge', sameEdge_p),
                                           ('samePod', samePod_p),
                                           ('otherPod', otherPod_p)])

            # Choose host at random within chosen group
            # Remove excluded
            receivers_choice = receivers[choice]
            receivers_choice = list(set(receivers_choice) - set(exclude))
            if receivers_choice != []:
                return random.choice(receivers_choice)
            else:
                return None

        elif self.pattern == 'bijection':
            receivers = list({receivers} - set(exclude))
            if len(receivers) == 1:
                return receivers[0]
            elif len(receivers) == 0:
                return None
            else:
                print("ERROR: Bijection can only return 1 receiver")
                import ipdb; ipdb.set_trace()

    def get_destination(self, pattern, sender, exclude=[]):
        if pattern == 'random':
            allReceivers = set(self.topology.getHosts())
            receivers = list(set(allReceivers) - set(exclude) - set(sender))
            if receivers:
                # Pick one at random
                return random.choice(receivers)
            else:
                return None

        elif pattern == 'stride':
            rcvs = self.senders[:]

            # Get pattern args
            stride_i = self.pattern_args.get('i', 4)

            # Get ordered list of hosts
            orderedHosts = self.topology.sortHostsByName(rcvs)

            for index, h in enumerate(orderedHosts):
                if h == sender:
                    r_index = (index + stride_i) % len(orderedHosts)
                    r = orderedHosts[r_index]
                    return r

    @abc.abstractmethod
    def plan_elephant_flows(self):
        """
        Generates a traffic plan for specified traffic pattern, during a certain
        amount of time.

        :return: dictionary host -> flowlist
        """

    def saveTraffic(self, filename, elephants_per_host, mice_bijections=None, mice_average=0.0):
        # Log it
        print("Saving traffic to file -> {0}".format(filename))

        # Convert current ip's to hostnames
        elephants_per_host = self.changeTrafficIpsToHostnames(elephants_per_host)
        mice_bijections = self.changeMiceIpsToHostnames(mice_bijections)
        traffic_to_save = {'elephant': elephants_per_host,
                           'mice': {'bijections': mice_bijections, 'average': mice_average},
                           'totalTime': self.totalTime}

        with open(filename, "w") as f:
            pickle.dump(traffic_to_save, f)

    def loadTraffic(self, filename):
        print("Loading traffic from file <- {0}".format(filename))

        # Fetch traffic from file
        saved_traffic = pickle.load(open(filename, "r"))
        elephants_per_host = saved_traffic.get('elephant')
        mice = saved_traffic.get('mice')
        mice_bijections = mice.get('bijections')
        mice_average = mice.get('average')
        self.mice_avg = mice_average
        totalTime = saved_traffic.get('totalTime')
        self.totalTime = totalTime

        # Convert hostnames to current ips
        elephants_per_host = self.changeTrafficHostnamesToIps(elephants_per_host)
        mice_bijections = self.changeMiceHostnamesToIps(mice_bijections)
        return elephants_per_host, mice_bijections, mice_average

    def resetController(self):
        # Reset controller first
        log.info("Sending RESET to controller")
        try:
            self.ControllerClient.send(json.dumps({"type": "reset"}), "")
        except:
            log.error("Controller is not connected/present")

    def sendStartTime(self, starttime, hosts):
        # Send traffic starting time
        log.info("Sending STARTTIMEs to hosts")
        for host in hosts:
            try:
                self.unixClient.send(json.dumps({"type": "starttime", "data": starttime}), host)

            except Exception as e:
                log.error("Host {0} could not be informed about starttime.".format(host))

    def sendFlowLists(self, traffic_per_host):
        # Schedule all the flows
        log.info("Sending FLOWLISTs to hosts")
        for sender, flowlist in traffic_per_host.iteritems():
            try:
                # Sends flowlist to the sender's server
                self.unixClient.send(json.dumps({"type": "flowlist", "data": flowlist}), sender)

            except Exception as e:
                log.error("Host {0} could not be informed about flowlist.".format(sender))

    def sendReceiveLists(self, receive_per_host):
        # Schedule all the flows
        log.info("Sending RECEIVE_LISTs to hosts")
        for receiver, receivelist in receive_per_host.iteritems():
            try:
                # Sends flowlist to the sender's server
                self.unixClient.send(json.dumps({"type": "receivelist", "data": receivelist}), receiver)

            except Exception as e:
                log.error("flowServer_{0} could not be informed about receive list.".format(receiver))

    def computeReceiveList(self, traffic_per_host):
        receivelist_per_host = {}

        for sender, flowlist in traffic_per_host.iteritems():
            for flow in flowlist:
                if flow['proto'].lower() == 'tcp':
                    dst_name = self.topology.getHostName(flow['dst'])
                    dport = flow['dport']
                    start_time = flow['start_time']
                    if dst_name not in receivelist_per_host.keys():
                        receivelist_per_host[dst_name] = [(start_time, dport)]
                    else:
                        receivelist_per_host[dst_name].append((start_time, dport))

        return {rcv: sorted(rcvlist, key=lambda x: x[0]) for rcv, rcvlist in receivelist_per_host.iteritems()}

    def addInitialDelay(self, traffic_per_host, delay):
        new_traffic_per_host = {}
        for host, flowlist in traffic_per_host.iteritems():
            new_traffic_per_host[host] = []
            for flow in flowlist:
                flow['start_time'] += delay
                new_traffic_per_host[host].append(flow)

        return new_traffic_per_host

    def resetCurrentTraffic(self):
        log.info("Resetting current traffic")

        # Terminate traffic at the hosts
        self.terminateTrafficAtHosts()

        # Send reset to controller
        self.resetController()

    def schedule(self, elephants_per_host, mice_bijections=None, mice_avg=None):
        """
        """
        if elephants_per_host:
            self.scheduleElephants(elephants_per_host)

        if mice_bijections:
            self.scheduleMices(mice_bijections, mice_avg)

    def scheduleElephants(self, elephants_per_host):
        """
        Sends the flowlists to their respective senders, together with the
        scheduling starting time.

        :param traffic_per_host:
        :return:
        """
        # Compute flows for which we need to start TCP server
        receive_per_host = self.computeReceiveList(elephants_per_host)

        # Compute average
        sums = [len(r) for s, r in receive_per_host.iteritems()]
        if sums:
            receive_avg = np.asarray(sums)
            receive_avg = float(receive_avg.mean())
            waiting_time = receive_avg/10.0

            # Add initial delay to all flows
            INITIAL_DELAY = waiting_time + 1
            elephants_per_host = self.addInitialDelay(elephants_per_host, INITIAL_DELAY)

            # Send receiveLists first
            self.sendReceiveLists(receive_per_host)

            log.info('Waiting {0}s before sending flowlists...'.format(waiting_time))
            time.sleep(waiting_time)

            # Send flowLists to flowServers!
            self.sendFlowLists(elephants_per_host)

    def scheduleMices(self, port_bijections, mice_avg):
        # Schedule all the flows
        log.info("Sending MICE bijections to hosts")
        for h, bijections in port_bijections.iteritems():
            try:
                # Sends flowlist to the sender's server
                data = {'bijections': bijections, 'average': mice_avg, 'totalTime': self.totalTime}
                self.unixClient.send(json.dumps({"type": "mice_bijections", "data": data}), h)

            except Exception as e:
                log.error("Host {0} could not be informed about flowlist.".format(h))

    def plan_from_flows_file(self, flows_file):
        """Opens the flows file and schedules the specified flows
        """
        # Open the file
        ff = open(flows_file)

        # One flow at each line
        flow_lines = ff.readlines()

        # Remove first explanatory line
        flow_lines = flow_lines[1:]

        # Flows to schedule
        traffic_per_host = {}

        # Parse file
        for flowline in flow_lines:
            # Try parsing flow data
            fields = flowline.split('\t')

            if len(fields) == 9 and '#' not in fields[0]:
                # Remove the \n from the last element
                fields[-1] = fields[-1].strip('\n')

                # Extract fields
                (src, sport, dst, dport, proto, start_time, size, rate, duration) = fields

                # Add entry for source if it's not there
                if src not in traffic_per_host.keys():
                    traffic_per_host[src] = []

                # Modify nones
                if size == 'None':
                    size = None
                if rate == 'None':
                    rate = None
                if duration == 'None':
                    duration = None

                try:
                    # Get hosts ip addresses
                    srcIp = self.topology.getHostIp(src)
                    dstIp = self.topology.getHostIp(dst)

                    # Create the flow object
                    flow = Flow(src=srcIp,
                                dst=dstIp,
                                sport=sport,
                                dport=dport,
                                proto=proto,
                                start_time=start_time,
                                size=size,
                                rate=rate,
                                duration=duration)

                    # Append it
                    traffic_per_host[src].append(flow.toDICT())

                except Exception as e:
                    print("ERROR: Flow object could not be created: {0}".format(e))
            else:
                continue
                #print("WARNING: Wrong flow format - line skipped: {0}".format(flowline))

        return traffic_per_host

    def terminateTrafficAtHosts(self):
        """
        Sends a reset command to the controller and terminate traffic command
        to all flowServers of the network
        """
        for host in self.topology.getHosts():
            try:
                self.unixClient.send(json.dumps({"type": "terminate"}), host)
            except Exception as e:
                log.error("Couldn't terminate traffic at flowServer_{0}. Exception: {1}".format(host, e))

    def changeTrafficHostnamesToIps(self, traffic):
        """
        Changes all hostnames in traffic dictionary to ips
        :param traffic:
        :return:
        """
        traffic_copy = {}
        for sender, flowlist in traffic.iteritems():
            flowlist_copy = []
            for flow in flowlist:
                flow_copy = flow.copy()
                flow_copy['src'] = self.topology.getHostIp(flow['src'])
                flow_copy['dst'] = self.topology.getHostIp(flow['dst'])
                flowlist_copy.append(flow_copy)

            traffic_copy[sender] = flowlist_copy

        return traffic_copy

    def changeTrafficIpsToHostnames(self, traffic):
        """
        Searches in host->flowlist traffic dictionary and changes all ips for
        hostnames, so that we save the traffic regardles of the current ip assigned
        to the hosts

        :param traffic:
        :return: dict: host->flowlist
        """
        traffic_copy = {}
        for sender, flowlist in traffic.iteritems():
            flowlist_copy = []
            for fw in flowlist:
                flow_copy = fw.copy()
                flow_copy['src'] = self.topology.getHostName(fw['src'])
                flow_copy['dst'] = self.topology.getHostName(fw['dst'])
                flowlist_copy.append(flow_copy)

            traffic_copy[sender] = flowlist_copy

        return traffic_copy

    def changeMiceHostnamesToIps(self, bijections):
        if bijections:
            bijections_c = copy.deepcopy(bijections)
            for host in bijections.iterkeys():
                for index, conn in enumerate(bijections[host]['toSend']):
                    dstname = conn['dst']
                    dstip = self.topology.getHostIp(dstname)
                    bijections_c[host]['toSend'][index]['dst'] = dstip
            return bijections_c
        else:
            return bijections

    def changeMiceIpsToHostnames(self, bijections):
        if bijections:
            bijections_c = copy.deepcopy(bijections)
            for host in bijections.iterkeys():
                for index, conn in enumerate(bijections[host]['toSend']):
                    dstip = conn['dst']
                    dstname = self.topology.getHostName(dstip)
                    bijections_c[host]['toSend'][index]['dst'] = dstname
            return bijections_c
        else:
            return bijections

    def choose_correct_src_dst_ports(self, flows_per_sender):
        """
        Chooses non colliding source and destiantion ports and re-writes ip addresses
        """
        # Keep track of available ports here
        availablePorts = {s: set(ElephantRangePorts) for s in flows_per_sender.keys()}

        # Return results here
        new_flows_per_sender = {}

        # Iterate all senders
        for sender, flowlist in flows_per_sender.iteritems():
            # Final flowlist
            new_flows_per_sender[sender] = []

            # Iterate flowlist
            for flow_tmp in flowlist:
                # Get available source ports for sender
                avSrcPorts = availablePorts[sender]

                # Get flow's destination
                dstHost = flow_tmp['dst']

                # Get available destination ports
                avDstPorts = availablePorts[dstHost]

                # Make a choice
                sport = random.choice(list(avSrcPorts))
                dport = random.choice(list(avDstPorts))


                # Update used ports
                availablePorts[sender] -= {sport}
                availablePorts[dstHost] -= {dport}

                # Get ips
                srcIp = self.topology.getHostIp(flow_tmp['src'])
                dstIp = self.topology.getHostIp(flow_tmp['dst'])

                # Create the flow object
                flow = Flow(src=srcIp, dst=dstIp, sport=sport, dport=dport,
                            proto=flow_tmp['proto'], start_time=flow_tmp['startTime'],
                            size=flow_tmp['size'], rate=flow_tmp['rate'], duration=flow_tmp['duration'])

                # Append it
                flowdict = flow.toDICT()
                flowdict.update({'non-blocking-ct': flow_tmp['non-blocking-ct']})
                new_flows_per_sender[sender].append(flowdict)

        # Return flowlist
        return new_flows_per_sender

    def calculate_mice_port_bijections(self):
        """
        - Each host has N TCP connections limultaneously open
        - We have to make sure that there are no port collisions
        :return:
        """
        N_CONNECTIONS = 100

        # Keep track of available ports here
        availablePorts = {s: set(MiceRangePorts) for s in self.senders}

        # Randomly shuffle hosts
        shuffled_hosts = self.senders[:]

        if nonNICCongestionTest:
            shuffled_hosts = [h for h in shuffled_hosts if '1' in h[-1] or '3' in h[-1]]

        random.shuffle(shuffled_hosts)

        # Accumulate result here
        port_bijections = {s: {'toReceive': [], 'toSend': []} for s in self.senders}

        # Iterate hosts
        for host in shuffled_hosts:
            other_hosts = list(set(shuffled_hosts) - {host})
            for conn in range(N_CONNECTIONS):
                # Choose source port
                avSrcPorts = list(availablePorts[host])
                sport = random.choice(avSrcPorts)

                # Choose random destination
                dst = random.choice(other_hosts)

                # Get destination ip
                dstIp = self.topology.getHostIp(dst)

                # Choose random dport from its available
                avDstPorts = list(availablePorts[dst])
                dport = random.choice(avDstPorts)

                # Update bijectoin mappings
                port_bijections[host]['toSend'].append({'sport': sport, 'dst': dstIp, 'dport': dport})
                port_bijections[dst]['toReceive'].append(dport)

                # Update availablePorts
                availablePorts[host] -= {sport}
                availablePorts[dst] -= {dport}
                # Return it

        test = False#True
        if test:
            for h in self.senders:
                if h != 'h_0_0':
                    port_bijections[h]['toSend'] = []

        return port_bijections


if __name__ == '__main__':
    tf = udpTrafficGeneratorBase()
    bijections = tf.calculate_mice_port_bijections()
    import ipdb; ipdb.set_trace()







