from flow import Flow, Base
from fibte.misc.topology_graph import TopologyGraph
import random
import time
import os

try:
    import cPickle as pickle
except:
    import pickle

import json

from fibte.logger import log
from fibte.trafficgen.flowGenerator import isElephant
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

saved_traffic_folder = os.path.join(os.path.dirname(__file__), 'saved_traffic/')

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

class TrafficGenerator(Base):
    def __init__(self, pMice = 0.9, pElephant = 0.1, *args, **kwargs):
        super(TrafficGenerator, self).__init__(*args,**kwargs)
        # Set debug level
        log.setLevel(logging.DEBUG)

        # Set traffic mice/elephant flow percent
        self.pMice = pMice
        self.pElephant = pElephant

        self.topology = TopologyGraph(getIfindexes=False,
                                      interfaceToRouterName=False,
                                      db=os.path.join(tmp_files,db_topo))

        self.linkBandwidth = LINK_BANDWIDTH

        # Used to communicate with flowServers at the hosts.
        # {0} is because it will be filled with whichever server we want to talk to!
        self.unixClient = UnixClientTCP(tmp_files+"flowServer_{0}")

        # Used to communicate with LoadBalancer Controller
        self.ControllerClient = UnixClient(os.path.join(tmp_files, controllerServer))

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
    def weighted_choice(weight_m, weight_e):
        """
        Makes a choice between
        :param weight_m:
        :param weight_e:
        :return:
        """
        choices = [("e", weight_e), ('m', weight_m)]
        total = sum(w for c, w in choices)
        r = random.uniform(0, total)
        upto = 0
        for c, w in choices:
            if upto + w >= r:
                return c
            upto += w
        assert False, "Shouldn't get here"

    def get_flow_type(self):
        return  self.weighted_choice(self.pMice, self.pElephant)

    def get_flow_duration(self, flow_type):
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

    def get_flow_size(self, flow_type):
        """
        :param flow_type:
        :return:
        """
        if flow_type == 'e':
            return random.randrange(ELEPHANT_SIZE_RANGE[0], ELEPHANT_SIZE_RANGE[1] + ELEPHANT_SIZE_STEP, ELEPHANT_SIZE_STEP)
        elif flow_type == 'm':
            return random.randrange(MICE_SIZE_RANGE[0], MICE_SIZE_RANGE[1]+MICE_SIZE_STEP, MICE_SIZE_STEP)
        else:
            raise ValueError("Unknown flow type: {0}".format(flow_type))

    def get_flow_destination(self, sender, receivers):
        """
        This method abstracts the choice of a receiver. It chooses a
        receiver uniformly at random from the list of receivers.

        I will be potentially subclassed by other instances of
        TrafficGenerator that generate other kinds of traffic patterns.

        :param receivers: list of possible receivers
        :return: chosen receiver
        """
        # Exclude receivers with the same edge router
        receivers = [r for r in receivers if not self.topology.inSameSubnetwork(sender, r)]

        # Pick one at random
        return random.choice(receivers)

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

    def _get_count(self, active_flows):
        """
        Count the number of mice and elephant flows
        """
        n_mice = sum([1 for flow in active_flows.values() if not isElephant(flow)])
        return (n_mice, len(active_flows) - n_mice)

    def _get_ratios(self, active_flows):
        """
        Given the current list of active flows, returns the percentages of mice and elephants

        :param active_flows: []: [{'type': 'mice', 'src': ...}, {flow2}]
        returns: tuple of ratio: (mice_rate, eleph_rate)
        """
        total_flows = len(active_flows)
        # Avoid zero division error
        if total_flows == 0:
            return (0, 0)

        else:
            # Get current mice&eleph count
            (n_mice, n_elephant) = self._get_count(active_flows)

            # Return ratios
            return map(lambda x: x/float(total_flows), (n_mice, n_elephant))

    def _get_active_flows(self, all_flows, period):
        """
        Returns a dict of the current flows in a certain traffic simulation period

        :param flows_per_sender: dict of all flows keyed by flow_id
        :param period: current time period
        :return: dict of active flows in the current time step keyed by flow_id
        """
        # Filter out the ones that haven't been visited yet
        visited_flows = {f_id: flow for f_id, flow in all_flows.iteritems() if flow.get("dstHost") != None}

        # Filter out the ones that finished before the start of the current period
        active_flows = {f_id: flow for f_id, flow in visited_flows.iteritems() if flow.get('startTime') + flow.get('duration') > period}

        # Return the active flows for the period
        return active_flows

    def _get_starting_flows(self, all_flows, period):
        """
        Returns a dictionary of all the flows that are starting in the current time period.
        Dictionary is keyed by flow id
        """

        # Filter out the visited ones
        non_visited_flows = {f_id: flow for f_id, flow in all_flows.iteritems() if flow.get("type") == None}

        # Filter out the ones that do not start in the period
        starting_flows = {f_id: flow for f_id, flow in non_visited_flows.iteritems()
                          if flow.get('startTime') >= period and flow.get('startTime') < period + self.time_step}

        return starting_flows

    def _get_best_mice_number(self, xm, n, x):
        results = []
        xe = x - xm
        for i in range(0, n+1):
            nm = i
            ne = n - nm
            new_pMice = ((xm + nm)/(float(x+n)))
            new_pElephant = ((xe + ne) / (float(x + n)))
            dm = abs(self.pMice - new_pMice)
            de = abs(self.pElephant - new_pElephant)
            sumde = dm + de
            results.append((i, sumde))

        optimal_mice = min(results, key=lambda x: x[1])

        return optimal_mice[0]

    def plan_flows(self, senders, receivers, flowRate, totalTime, timeStep):
        """
        Given a sender list and a list of possible receivers, together with the
        total simulation time, this function generates random flows from each
        sender to the possible receivers.

        The number of flows that each sender generates and their starting time
        is given by a Poisson arrival process with a certain average.

        The objective is that, at any point in the simulation time, a certain
        fraction of flows are elephant and the rest are mice.

        For each flow:
          - A receiver is chosen uniformly at random among the receivers list

          - Weather the flow is mice or elephant is chosen with certain proba
            bility depending on the object initialization.

          - Durtaion is chosen uniformly at random within certain pre-defined ranges

          - Size is chosen from a normal distribution relative to the total link capacity

        :param sender: sending hosts list
        :param receivers: list of possible receiver hosts
        :param totalTime: total simulation time

        :return: dictionary keyed by sender associated with the flowlist
                 of ordered flows for each sender
        """

        # The resulting traffic is stored here
        flows_per_sender = {}

        # We first obtain the flow start times for each host
        flows_per_sender_tmp = {sender: self.get_poisson_times(flowRate, totalTime) for sender in senders}

        # Initial flow id
        next_id = 0

        # Convert start times list into dict
        for sender, time_list in flows_per_sender_tmp.iteritems():
            # Convert it in a list of dicts with unique id for each flow
            list_of_dicts = [{'startTime': starttime, 'id': next_id+index, 'srcHost': sender, 'type': None} for index, starttime in enumerate(time_list)]
            next_id += len(time_list)
            flows_per_sender[sender] = list_of_dicts

        # Create dict indexed by flow id {}: id -> flow
        all_flows = {flow['id']: flow for sender, flowlist in flows_per_sender.iteritems() for flow in flowlist}

        # Iterate simulation time trying to keep the desired elephant and mice ratios
        if timeStep == 0:
            self.time_step = totalTime/10
        else:
            self.time_step = timeStep

        for period in range(0, totalTime, self.time_step):

            # Get active flows in current period
            active_flows = self._get_active_flows(all_flows, period)

            # Get next starting flows
            starting_flows = self._get_starting_flows(all_flows, period)

            # If not enough active flows yet - choose e/m with weighted prob.
            if len(active_flows) < 10:
                # Flow ids of the flows that will be mice
                starting_flows_mice = []

                # Randomly draw if elephant or mice for each new active_flow
                for f_id, flow in starting_flows.iteritems():
                    # Draw the choice
                    flow_type = self.weighted_choice(self.pMice, self.pElephant)
                    if flow_type == 'm': starting_flows_mice.append(f_id)

            # Calculate how many should be mice to keep good ratio otherwise
            else:
                # Get current counts
                (current_nMice, current_nElephant) = self._get_count(active_flows)

                # Compute what's the best allocation
                n_mice = self._get_best_mice_number(xm=current_nMice, n=len(starting_flows), x=len(active_flows))

                # Choose randomly which flow ids will be mice
                msflows = starting_flows.keys()
                random.shuffle(msflows)
                starting_flows_mice = msflows[:n_mice]

            # Update starting flows data
            for f_id, flow in starting_flows.iteritems():
                # Mice
                if f_id in starting_flows_mice:
                    flow_type = 'm'

                # Elephant
                else:
                    flow_type = 'e'

                # Get flow duration
                flow_duration = self.get_flow_duration(flow_type)

                # Get flow size
                flow_size = self.get_flow_size(flow_type)

                # Sender
                snd = flow['srcHost']

                # Choose receiver
                receiver = self.get_flow_destination(snd, receivers)

                # Update flow
                all_flows[f_id] = {'srcHost': flow['srcHost'],
                                   'dstHost': receiver,
                                   'size': flow_size,
                                   'startTime': flow['startTime'],
                                   'duration': flow_duration,
                                   'sport': -1, 'dport': -1}

        # Update the flows_per_sender dict
        new_flows_per_sender = {}
        for sender, flowlist in flows_per_sender.iteritems():
            new_flows_per_sender[sender] = []
            for flow in flowlist:
                new_flows_per_sender[sender].append(all_flows[flow['id']])

        # Re-write correct source and destination ports per each sender
        flows_per_sender = {}
        for sender, flowlist in new_flows_per_sender.iteritems():
            flowlist = self.choose_correct_ports(flowlist)
            flows_per_sender[sender] = flowlist

        return flows_per_sender

    def parse_communication_parties(self, senders, receivers):
        """
        Given lists of possible senders and receivers, returns the lists
        of sender and receiver hosts.

        Senders and receivers are two lists of strings specifying pods,
        edge routers or hosts, indistinctively.

        :param senders: []-> {pod_X | r_X_eY | h_X_Y}
        :param receivers: []-> {pod_X | r_X_eY | h_X_Y}
        :return: Two lists: (sender_hosts, receiver_hosts)
        """
        if senders != ['all']:
            # Get all hosts behind pod, edge router or hostname
            host_senders_lists = []
            for sender in senders:
                # Pod specified
                if 'pod' in sender:
                    host_senders_lists.append(self.topology.getHostsBehindPod(sender))

                # Router edge was specified
                elif 'r_' in sender and self.topology.isEdgeRouter(sender):
                    host_senders_lists.append(self.topology.getHostsBehindRouter(sender))

                # Host was directly specified
                elif 'h_' in sender:
                    host_senders_lists.append([sender])

                # Raise error
                else:
                    raise ValueError("Communicatoin pattern was wrongly specified. Error due to: {0}".format(sender))

            # Convert it to list of unique senders
            host_senders = list({host for host_list in host_senders_lists for host in host_list})

        else:
            # Get all of them
            host_senders = self.topology.getHosts().keys()

        if receivers != ['all']:
            # Get all hosts behind pod, edge router or hostname
            host_receivers_lists = []
            for receiver in receivers:
                # Pod specified
                if 'pod' in receiver:
                    host_receivers_lists.append(self.topology.getHostsBehindPod(receiver))

                # Router edge was specified
                elif 'r_' in receiver and self.topology.isEdgeRouter(receiver):
                    host_receivers_lists.append(self.topology.getHostsBehindRouter(receiver))

                # Host was directly specified
                elif 'h_' in receiver:
                    host_receivers_lists.append([receiver])

                # Raise error
                else:
                    raise ValueError("Communicatoin pattern was wrongly specified. Error due to: {0}".format(receiver))

            # Convert it to list of unique receivers
            host_receivers = list({host for host_list in host_receivers_lists for host in host_list})
        else:
            # Get all of them
            host_receivers = self.topology.getHosts().keys()

        return (host_senders, host_receivers)

    def trafficPlanner(self, senders=['pod_0'], receivers=['pod_3'], flowRate=0.25, totalTime=500, timeStep=0):
        """
        Generates a traffic plan for specified senders and receivers, with
        a given flow starting rate per each host for a specified simulation time

        :param senders: list of pods or edge routers
        :param receivers: list of pods or edge routers
        :param flowRate: floating number indicating flow stargin rate
        :param totalTime: total simulation time
        :return: dictionary host -> flowlist
        """
        # Parse communication parties
        host_senders, host_receivers = self.parse_communication_parties(senders, receivers)

        # Holds {}: host -> flowlist
        traffic_per_host = self.plan_flows(host_senders, host_receivers, flowRate, totalTime, timeStep)

        # Return all generated traffic
        return traffic_per_host

    def schedule(self, traffic_per_host):
        """
        Sends the flowlists to their respective senders, together with the
        scheduling starting time.

        :param traffic_per_host:
        :return:
        """
        try:
            # Reset controller first
            self.ControllerClient.send(json.dumps({"type":"reset"}),"")
        except:
            log.error("Controller is not connected/present")

        # Wait a bit
        time.sleep(1)

        # Set sync. delay
        SYNC_DELAY = 5

        # Schedule all the flows
        try:
            for sender, flowlist in traffic_per_host.iteritems():
                # Sends flowlist to the sender's server
                self.unixClient.send(json.dumps({"type": "flowlist", "data": flowlist}), sender)

        except Exception as e:
            log.error("Host {0} could not be informed about flowlist. Error: {1}".format(sender, e))
            #raise Exception("Host {0} could not be informed about flowlist.\n\tException: {1}".format(sender, e))

        try:
            # Set traffic start time -- same time for everyone!
            traffic_start_time = time.time() + SYNC_DELAY
            for sender in traffic_per_host.keys():
                # Send traffic starting time
                self.unixClient.send(json.dumps({"type": "starttime", "data": traffic_start_time}), sender)

        except Exception as e:

            log.error("Host {0} could not be informed about starttime. Error: {1}".format(sender, e))
            #raise Exception("Host {0} could not be informed about flowlist.\n\tException: {1}".format(sender, e))

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
            if len(fields) == 7:
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
        try:
            for host in self.topology.networkGraph.getHosts():
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

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    #group = parser.add_mutually_exclusive_group()

    parser.add_argument('--terminate', help='Terminate any ongoing traffic', action='store_true')
    parser.add_argument('-t', '--time',
                           help='Duration of the traffic generator',
                           type=int,
                           default=400)

    parser.add_argument('-r', '--flow_rate',
                           help="Rate at which a host starts new flows (flows/second)",
                           type=float,
                           default=0.25)

    parser.add_argument('-e', '--elephant',
                           help='Percentage of elephant flows',
                           type=float,
                           default=0.1)

    parser.add_argument('-m', '--mice',
                           help='Percentage of mice flows',
                           type=float,
                           default=0.9)

    parser.add_argument('-s', '--time_step',
                           help="Granularity at which we inspect the generated traffic so that the rates are kept",
                           type=int,
                           default=1)

    parser.add_argument('--senders',
                           help='List of switch edges or pods that can send traffic',
                           default="all")

    parser.add_argument('--receivers',
                           help='List of switch edges or pods that can receive traffic',
                           default="all")

    parser.add_argument('--save_traffic',
                           help='saves traffic in a file so it can be repeated',
                           action="store_true")

    parser.add_argument('--load_traffic',
                        help='load traffic from a file so it can be repeated',
                           default="")

    parser.add_argument('--flows_file',
                        help="Schedule the flows specified in file",
                        default=False)

    # Parse the arguments
    args = parser.parse_args()

    # Prepare senders and receivers
    senders = args.senders.split(",")
    receivers = args.receivers.split(",")

    # Start the TG object
    tg = TrafficGenerator(pMice=args.mice, pElephant=args.elephant)

    # Start counting time
    t = time.time()
    if args.terminate:
        print "Terminating ongoing traffic!"
        tg.terminateTraffic()

    else:

        # Check if flow file has been given (for testing purposes)
        if not args.flows_file:
            # If traffic must be loaded
            if args.load_traffic:
                msg = "Loading traffic from file <- {0}"
                print msg.format(args.load_traffic)

                # Fetch traffic from file
                traffic = pickle.load(open(args.load_traffic,"r"))

                # Convert hostnames to current ips
                traffic = tg.changeTrafficHostnamesToIps(traffic)

            else:
                # Generate traffic
                traffic = tg.trafficPlanner(senders=senders,receivers=receivers,
                                            flowRate=args.flow_rate,totalTime=args.time,
                                            timeStep=args.time_step)

                msg = "Generating traffic:\n\tSenders: {0}\n\tReceivers: {1}\n\tFlow rate: {2}\n\t"
                msg += "Total time: {3}\n\tTime step: {4}"
                print msg.format(args.senders, args.receivers, args.flow_rate, args.time, args.time_step)

                # If it must be saved
                if args.save_traffic:
                    msg = "Saving traffic file -> {0}"
                    filename = '{0}'.format(saved_traffic_folder)
                    filename += "{0}_to_{1}_m{2}e{3}_fr{4}_t{5}_ts{6}.traffic".format(','.join(senders), ','.join(receivers),
                                                                                      str(args.mice).replace('.', ','),
                                                                                      str(args.elephant).replace('.', ','),
                                                                                      str(args.flow_rate).replace('.', ','),
                                                                                      args.time, args.time_step)

                    # Convert current ip's to hostnames
                    traffic_to_save = tg.changeTrafficIpsToHostnames(traffic)
                    print msg.format(filename)
                    with open(filename,"w") as f:
                        pickle.dump(traffic_to_save,f)

            # Orchestrate the traffic (either loaded or generated)
            print "Scheduling traffic..."
            tg.schedule(traffic)

        # Flow file has been given
        else:
            print "Scheduling flows specified in {0}".format(args.flows_file)
            traffic = tg.plan_from_flows_file(args.flows_file)
            #import ipdb; ipdb.set_trace()
            tg.schedule(traffic)

    print "Elapsed time ", time.time()-t


# Example commandline call:
# python trafficGenerator.py --senders pod_0,pod_1 --receivers pod_2,pod_3 --mice 0.8 --elephant 0.2 --flow_rate 0.25 --time 300 --save_traffic
# python trafficGenerator.py --terminate
# python trafficGenerator.py --load_traffic saved_traffic/

