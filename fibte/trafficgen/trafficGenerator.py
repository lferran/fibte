from flow import Flow, Base
from fibte.misc.topologyGraph import TopologyGraph
import random
import time
import os

try:
    import cPickle as pickle
except:
    import pickle

import json

from fibte.logger import log

from fibte.misc.unixSockets import UnixClient, UnixClientTCP
from fibte import CFG, LINK_BANDWIDTH, MICE_SIZE_RANGE, ELEPHANT_SIZE_RANGE, MICE_SIZE_STEP, ELEPHANT_SIZE_STEP

tmp_files = CFG.get("DEFAULT","tmp_files")
db_topo = CFG.get("DEFAULT","db_topo")
controllerServer = CFG.get("DEFAULT","controller_UDS_name")

MIN_PORT = 1
MAX_PORT = 2**16 -1
RangePorts = xrange(MIN_PORT,MAX_PORT)


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

        # Set traffic mice/elephant flow percent
        self.pMice = pMice
        self.pElephant = pElephant

        self.topology = TopologyGraph(getIfindexes=False,
                                      snmp=False,
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
        max_len_elephant = 500
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

    def get_flow_destination(self, receivers):
        """
        This method abstracts the choice of a receiver. It chooses a
        receiver uniformly at random from the list of receivers.

        I will be potentially subclassed by other instances of
        TrafficGenerator that generate other kinds of traffic patterns.

        :param receivers: list of possible receivers
        :return: chosen receiver
        """
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

    def plan_flows(self, sender, receivers, flowRate, totalTime):
        """
        Given a sender and a list of possible receivers, together with the
        total simulation time, this function generates random flows from
        sender to the receiver.

        The number of flows that sender generates and their starting time
        is given by a Poisson arrival process with a certain average.

        For each flow:
          - A receiver is chosen uniformly at random among the receivers list

          - Weather the flow is mice or elephant is chosen with certain proba
            bility depending on the object initialization.

          - Durtaion is chosen uniformly at random within certain pre-defined ranges

          - Size is chosen from a normal distribution relative to the total link capacity

        :param sender: sending host
        :param receivers: list of possible receiver hosts
        :param totalTime: total simulation time

        :return: flowlist of ordered flows for a certain host
        """

        # List of flows planned for the sender
        flowlist = []

        # Generate flow starting times
        flow_times = self.get_poisson_times(average=flowRate, totalTime=totalTime)

        # Iterate each flow
        for flow_time in flow_times:
            # Is flow mice or elephant?
            flow_type = self.get_flow_type()

            # Get flow duration
            flow_duration = self.get_flow_duration(flow_type)

            # Get flow size
            flow_size = self.get_flow_size(flow_type)

            # Choose receiver
            receiver = self.get_flow_destination(receivers)

            # Create temporal flow
            flow_tmp = {'srcHost':sender, 'dstHost':receiver, 'size':flow_size, 'startTime':flow_time, 'duration':flow_duration, 'sport': -1, 'dport': -1}

            # Append it to the list
            flowlist.append(flow_tmp)

        # Re-write correct source and destination ports
        flowlist = self.choose_correct_ports(flowlist)

        return flowlist

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

    def trafficPlanner(self, senders=['pod_0'], receivers=['pod_3'], flowRate=0.25, totalTime=500):
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
        traffic_per_host = {}

        for sender in host_senders:
            # Generate flowlist
            flowlist = self.plan_flows(sender, host_receivers, flowRate, totalTime)

            # Save it
            traffic_per_host[sender] = flowlist

        # Return all generated traffic
        return traffic_per_host

    def schedule(self, traffic_per_host):

        try:
            # Reset controller
            #self.ControllerClient.send(json.dumps({"type":"reset"}),"")
            pass
        except:
            # log.debug("Controller is not connected/present")
            pass

        # Set sync. delay
        SYNC_DELAY = 10

        # Set traffic start time
        traffic_start_time = time.time() + SYNC_DELAY

        # Schedule all the flows
        try:
            for sender, flowlist in traffic_per_host.iteritems():

                # Sends flowlist to the sender's server
                self.unixClient.send(json.dumps({"type": "flowlist", "data": flowlist}), sender)

                # Send traffic starting time
                self.unixClient.send(json.dumps({"type": "starttime", "data": traffic_start_time}), sender)

        except Exception as e:
            #log.info("Host {0} could not be informed about flowlist/starttime".format(sender))
            raise Exception("Host {0} could not be informed about flowlist/starttime.\n\tException: {1}".format(sender, e))

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


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    #group = parser.add_mutually_exclusive_group()
    parser.add_argument('--terminate', help='Terminate ongoing traffic', action='store_true')

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


    parser.add_argument('--senders',
                           help='List of switch edges or pods that can send traffic',
                           default="r_0_e0")


    parser.add_argument('--receivers',
                           help='List of switch edges or pods that can receive traffic',
                           default="r_1_e0")

    parser.add_argument('--save_traffic',
                           help='saves traffic in a file so it can be repeated',
                           action="store_true")

    parser.add_argument('--load_traffic',
                           help='load traffic from a file so it can be repeated',
                           default="")

    args = parser.parse_args()

    senders = args.senders.split(",")
    receivers = args.receivers.split(",")

    # Start the TG object
    tg = TrafficGenerator(pMice=args.mice, pElephant=args.elephant)

    t = time.time()

    if args.terminate:
        tg.terminateTraffic()

    else:
        # If traffic must be loaded
        if args.load_traffic:
            # Fetch traffic from file
            traffic = pickle.load(open(args.load_traffic,"r"))
        else:
            # Generate traffic
            traffic = tg.trafficPlanner(senders=senders,
                                        receivers=receivers,
                                        flowRate=args.flow_rate,
                                        totalTime=args.time)

        # If it must be saved
        if args.save_traffic:
            filename = "{0}_to_{1}_m{2}e{3}_fr{4}_t{5}.traffic".format(','.join(senders),
                                                               ','.join(receivers),
                                                               str(args.mice).replace('.', ''),
                                                               str(args.elephant).replace('.', ''),
                                                               str(args.flow_rate).replace('.', ''),
                                                               args.time)
            with open(args.save_traffic,"w") as f:
                pickle.dump(traffic,f)

        # Orchestrate the traffic (either loaded or generated)
        tg.schedule(traffic)

    print "elapsed time ", time.time()-t


# Example commandline call:
# python trafficGenerator.py --senders pod_0,pod_1 --receivers pod_2,pod_3 --mice 0.8 --elephant 0.2 --flow_rate 0.25 --time 300 --save_traffic pod01_to_pod02_m08e02_fr025_t300.traffic

# python trafficGenerator.py --terminate

