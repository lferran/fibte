from flow import Flow, Base
from fibte.misc.topology_graph import TopologyGraph
import random
import time
import os
import argparse
import abc
import json

try:
    import cPickle as pickle
except:
    import pickle

import json
import scipy.stats as stats

from fibte.misc.unixSockets import UnixClient, UnixClientTCP
from fibte import CFG, LINK_BANDWIDTH, MICE_SIZE_RANGE, ELEPHANT_SIZE_RANGE, MICE_SIZE_STEP, ELEPHANT_SIZE_STEP

import logging
from fibte.logger import log

tmp_files = CFG.get("DEFAULT","tmp_files")
db_topo = CFG.get("DEFAULT","db_topo")
controllerServer = CFG.get("DEFAULT","controller_UDS_name")

MIN_PORT = 1
MAX_PORT = 2**16 -1
RangePorts = xrange(MIN_PORT,MAX_PORT)

SAVED_TRAFFIC_DIR = os.path.join(os.path.dirname(__file__), 'saved_traffic/')

def time_func(function):
    def wrapper(*args,**kwargs):
        t = time.time()
        res = function(*args,**kwargs)
        log.debug("{0} took {1}s to execute".format(function.func_name, time.time()-t))
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
        self.parser.add_argument('--pattern', help='Communication pattern', choices=['random','staggered','bijection','stride'], type=str, default='random')
        self.parser.add_argument('--pattern_args', help='Communication pattern arguments', type=json.loads, default='{}')
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

    def printArgs(self, args):
        import ipdb; ipdb.set_trace()

class udpTrafficGeneratorBase(Base):
    def __init__(self, pattern='random', pattern_args={}, totalTime=100, timeStep=1, *args, **kwargs):
        super(udpTrafficGeneratorBase, self).__init__(*args, **kwargs)

        # Fodler where we store the traffic files
        self.saved_traffic_dir = SAVED_TRAFFIC_DIR

        # Link bandwidth
        self.linkBandwidth = LINK_BANDWIDTH

        # Get attributes
        self.pattern = pattern
        self.pattern_args = pattern_args

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
        self.possible_destinations = self._createPossibleDestinations()

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
        min_len_mice = 2
        max_len_mice = 10

        # flow is elephant
        if flow_type == 'e':
            # Draw flow duration
            return random.randint(min_len_elephant, max_len_elephant)

        # flow is mice
        elif flow_type == 'm':
            # Draw flow duration
            return random.randint(min_len_mice, max_len_mice)

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
            return random.randrange(MICE_SIZE_RANGE[0], MICE_SIZE_RANGE[1]+MICE_SIZE_STEP, MICE_SIZE_STEP)

        else:
            raise ValueError("Unknown flow type: {0}".format(flow_type))

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

    @abc.abstractmethod
    def plan_flows(self):
        """
        Generates a traffic plan for specified traffic pattern, during a certain
        amount of time.

        :return: dictionary host -> flowlist
        """

    def schedule(self, traffic_per_host):
        """
        Sends the flowlists to their respective senders, together with the
        scheduling starting time.

        :param traffic_per_host:
        :return:
        """
        try:
            # Reset controller first
            self.ControllerClient.send(json.dumps({"type": "reset"}), "")
        except:
            log.error("Controller is not connected/present")

        # Wait a bit
        time.sleep(0.5)

        # Set sync. delay
        SYNC_DELAY = 2

        # Schedule all the flows
        for sender, flowlist in traffic_per_host.iteritems():
            try:
                # Sends flowlist to the sender's server
                self.unixClient.send(json.dumps({"type": "flowlist", "data": flowlist}), sender)

            except Exception as e:
                log.error("Host {0} could not be informed about flowlist.".format(sender))

        # Set traffic start time -- same time for everyone!
        traffic_start_time = time.time() + SYNC_DELAY

        for sender in traffic_per_host.keys():
            try:
                # Send traffic starting time
                self.unixClient.send(json.dumps({"type": "starttime", "data": traffic_start_time}), sender)

            except Exception as e:
                log.error("Host {0} could not be informed about starttime.".format(sender))

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
            if len(fields) == 7 and '#' not in fields[0]:
                # Remove the \n from the last element
                fields[-1] = fields[-1].strip('\n')

                # Extract fields
                (src, sport, dst, dport, start_time, size, duration) = fields

                # Add entry for source if it's not there
                if src not in traffic_per_host.keys(): traffic_per_host[src] = []

                try:
                    # Get hosts ip addresses
                    srcIp = self.topology.getHostIp(src)
                    dstIp = self.topology.getHostIp(dst)

                    # Create the flow object
                    flow = Flow(src=srcIp, dst=dstIp, sport=sport, dport=dport,
                                size=size, start_time=start_time, duration=duration)

                    # Append it
                    traffic_per_host[src].append(flow.toDICT())

                except Exception as e:
                    print("ERROR: Flow object could not be created: {0}".format(e))
            else:
                continue
                #print("WARNING: Wrong flow format - line skipped: {0}".format(flowline))

        return traffic_per_host

    def terminateTraffic(self):
        """
        Sends a reset command to the controller and terminate traffic command
        to all flowServers of the network
        """
        for host in self.topology.getHosts():
            try:
                self.unixClient.send(json.dumps({"type": "terminate"}), host)
            except Exception as e:
                log.debug("FlowServer of {0} did not receive terminate command. Exception: {1}".format(host, e))
                pass
        try:
            # Send reset to the controller
            self.ControllerClient.send(json.dumps({"type": "reset"}), "")
        except Exception as e:
            log.debug("Controller is not connected/present. Exception: {0}".format(e))
            pass

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
            for flow in flowlist:
                flow_copy = flow.copy()
                flow_copy['src'] = self.topology.getHostName(flow['src'])
                flow_copy['dst'] = self.topology.getHostName(flow['dst'])
                flowlist_copy.append(flow_copy)

            traffic_copy[sender] = flowlist_copy

        return traffic_copy

    def choose_correct_ports(self, flowlist_tmp):
        """
        Given a temporal flow list, the ports of the flows have to be randomly
        choosen in such a way that no two active outgoing flows from the sender
        use the same source port.

        :param flowlist_tmp: temporal flowlist
        :return: final flowlist
        """
        # Final flowlist
        flowlist = []

        allPorts = set(RangePorts)

        for index, flow_tmp in enumerate(flowlist_tmp):

            # Current flow start time
            start_time  = flow_tmp['startTime']

            # Filter out the flows that are not active anymore
            active_flows = [v_flow for v_flow in flowlist_tmp[:index] if v_flow['startTime'] + v_flow['duration'] + 1 >= start_time]

            # Collect used port numbers
            usedPorts = set([a_flow['sport'] for a_flow in active_flows])

            # Calculate available ports
            availablePorts = allPorts - usedPorts

            # Choose random source port from the available
            sport = random.choice(list(availablePorts))

            # Choose also random destination port: no restrictions here
            dport = random.choice(RangePorts)

            # Get hosts ip addresses
            srcIp = self.topology.getHostIp(flow_tmp['srcHost'])
            dstIp = self.topology.getHostIp(flow_tmp['dstHost'])

            # Create the flow object
            flow = Flow(src=srcIp, dst=dstIp, sport=sport, dport=dport, size=flow_tmp['size'],
                        start_time=flow_tmp['startTime'], duration=flow_tmp['duration'])

            # Append it to the list -- must be converted to dictionary to be serializable
            flowlist.append(flow.toDICT())

        # Return flowlist
        return flowlist
