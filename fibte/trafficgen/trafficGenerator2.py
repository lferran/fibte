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
import scipy.stats as stats
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

from trafficGenerator import TrafficGeneratorBase

class TrafficGenerator(TrafficGeneratorBase):
    def __init__(self, mice_rate=1.5, elephant_rate=0.1, target_link_load=0.5, *args, **kwargs):
        super(TrafficGenerator, self).__init__(*args, **kwargs)
        self.target_link_load = LINK_BANDWIDTH*target_link_load
        self.elephant_rate = elephant_rate
        self.mice_rate = mice_rate

    def traffic_planner(self, senders=['pod_0'], receivers=['pod_3'], totalTime=500, timeStep=0):
        # Parse communication parties
        host_senders, host_receivers = self.parse_communication_parties(senders, receivers)

        # Holds {}: host -> flowlist
        traffic_per_host = self.plan_the_flows(host_senders, host_receivers, totalTime, timeStep)

        # Return all generated traffic
        return traffic_per_host

    def plan_the_flows(self, senders, receivers, totalTime, timeStep):
        """
        :return:
        """
        # The resulting traffic is stored here
        flows_per_sender = {}

        # Initial flow id
        next_id = 0

        # We first obtain the flow start times for elephants at each host
        elephants_times = {sender: self.get_poisson_times(self.elephant_rate, totalTime) for sender in senders}
        # Generate elephants times first
        for sender, time_list in elephants_times.iteritems():
            list_of_dicts = []
            for index, starttime in enumerate(time_list):
                # Convert it in a list of dicts with unique id for each flow
                size = self.get_flow_size(flow_type='e')
                duration = self.get_flow_duration(flow_type='e')
                destination = self.get_flow_destination(sender, receivers, flow_type='e')
                list_of_dicts.append({'startTime': starttime, 'id': next_id + index,
                                      'srcHost': sender, 'dstHost': destination,
                                      'type': 'e', 'size': size, 'duration': duration})
            flows_per_sender[sender] = list_of_dicts
            next_id += len(time_list)

        # Create dict indexed by flow id {}: id -> flow
        all_flows = {flow['id']: flow for sender, flowlist in flows_per_sender.iteritems() for flow in flowlist}

        # Now we generate start times for mice flows
        mice_times = {sender: self.get_poisson_times(self.mice_rate, totalTime) for sender in senders}
        for sender, time_list in mice_times.iteritems():
            list_of_dicts = []
            for index, starttime in enumerate(time_list):
                # Convert it in a list of dicts with unique id for each flow
                size = self.get_flow_size(flow_type='m')
                duration = self.get_flow_duration(flow_type='m')
                destination = self.get_flow_destination(sender, receivers, flow_type='m')
                ff = {'startTime': starttime, 'id': next_id + index, 'srcHost': sender, 'dstHost': destination, 'type': 'm', 'size': size, 'duration': duration}
                # Add it to all_flows
                all_flows[ff['id']] = ff

                # Append it to list of dicts (flows)
                list_of_dicts.append(ff)

            # Append mice flows in sender dict
            flows_per_sender[sender] += list_of_dicts
            # Sort lists of dicts per starting time
            final_list = flows_per_sender[sender]
            flows_per_sender[sender] = sorted(final_list, key=lambda x: x.get('startTime'))
            next_id += len(time_list)

        # Iterate simulation time trying to keep the desired elephant and mice ratios
        if timeStep == 0: self.time_step = totalTime/10
        else: self.time_step = timeStep

        print "Initial number of flows: {0}".format(len(all_flows))

        # Check first transmitted load from each host
        for period in range(0, totalTime, self.time_step):
            # Get active and starting flows at that period of time
            active_flows = self._get_active_flows_2(all_flows, period)
            starting_flows = self._get_starting_flows_2(all_flows, period)

            # Iterate senders
            for sender in senders:
                # Get active load tx at sender
                active_load = sum([all_flows[id]['size'] for id in active_flows.keys() if all_flows[id]['srcHost'] == sender])

                #Get new load at sender
                starting_load = sum([all_flows[id]['size'] for id in starting_flows.keys() if all_flows[id]['srcHost'] == sender])

                # Virtual load
                vload = active_load + starting_load

                # Get all starting flow idsand shuffle them
                sf_ids = starting_flows.keys()[:]
                random.shuffle(sf_ids)

                while vload > LINK_BANDWIDTH and sf_ids != []:
                    # Remove one new flow at random and recompute
                    id_to_remove = sf_ids[0]
                    sf_ids.remove(id_to_remove)
                    all_flows.pop(id_to_remove)
                    starting_flows.pop(id_to_remove)

                    # Recompute
                    starting_load = sum([all_flows[id]['size'] for id in starting_flows.keys() if all_flows[id]['srcHost'] == sender])

                    # Virtual load
                    vload = active_load + starting_load

        for i in range(2):
            # Check then received loads at each host
            for period in range(0, totalTime, self.time_step):

                # Get active and starting flows at that period of time
                active_flows = self._get_active_flows_2(all_flows, period)
                starting_flows = self._get_starting_flows_2(all_flows, period)

                # Iterate receivers
                for rcv in receivers:
                    # Get active load rx at receiver
                    active_load = sum([all_flows[id]['size'] for id in active_flows.keys() if all_flows[id]['dstHost'] == rcv])

                    # Get new load rx at receiver
                    starting_load = sum([all_flows[id]['size'] for id in starting_flows.keys() if all_flows[id]['dstHost'] == rcv])

                    # Virtual load
                    vload = active_load + starting_load

                    # If new starting flows don't fit in receiver link: try to reallocate them
                    if vload > LINK_BANDWIDTH:

                        # Get all starting flow idsand shuffle them
                        sf_ids = starting_flows.keys()[:]
                        random.shuffle(sf_ids)

                        while vload > LINK_BANDWIDTH and sf_ids != []:
                            # Remove one new flow at random and recompute
                            id_to_reallocate = sf_ids[0]
                            sf_ids.remove(id_to_reallocate)
                            starting_flows.pop(id_to_reallocate)

                            # Choose another destination
                            # Compute all other possible receivers
                            possible_receivers = list(set(receivers) - {rcv})
                            sender = all_flows[id_to_reallocate]['srcHost']
                            flow_type = all_flows[id_to_reallocate]['type']
                            new_dst = self.get_flow_destination(sender, possible_receivers, flow_type)

                            loads_new_dst = sum([all_flows[id]['size'] for id in active_flows.keys() if
                                                 all_flows[id]['dstHost'] == new_dst])

                            if loads_new_dst + all_flows[id_to_reallocate]['size'] > LINK_BANDWIDTH:

                                while loads_new_dst + all_flows[id_to_reallocate]['size'] > LINK_BANDWIDTH and new_dst != None:
                                    # Choose new dst
                                    possible_receivers = list(set(possible_receivers) - {new_dst})
                                    new_dst = self.get_flow_destination(sender, possible_receivers, flow_type)

                                    # Recompute loads new dst
                                    loads_new_dst = sum([all_flows[id]['size'] for id in active_flows.keys() if
                                                         all_flows[id]['dstHost'] == new_dst])

                                # If we flow could not be fit anywhere:
                                if new_dst == None:
                                    # Remove flow forever
                                    all_flows.pop(id_to_reallocate)

                                else:
                                    # Reallocate
                                    all_flows[id_to_reallocate]['dstHost'] = new_dst

                            else:
                                # Reallocate
                                all_flows[id_to_reallocate]['dstHost'] = new_dst


                            # Recompute
                            starting_load = sum([all_flows[id]['size'] for id in starting_flows.keys() if all_flows[id]['dstHost'] == rcv])

                            # Virtual load
                            vload = active_load + starting_load

        print "After-reallocation/removal number of flows: {0}".format(len(all_flows))

        # Update the flows_per_sender dict
        new_flows_per_sender = {}
        for sender, flowlist in flows_per_sender.iteritems():
            new_flows_per_sender[sender] = []
            for flow in flowlist:
                if flow['id'] in all_flows.keys():
                    floww = all_flows[flow['id']]
                    floww['sport'] = -1
                    floww['dport'] = -1
                    new_flows_per_sender[sender].append(floww)

        # Re-write correct source and destination ports per each sender
        print "Choosing correct ports..."
        flows_per_sender = {}
        for sender, flowlist in new_flows_per_sender.iteritems():
            flowlist = self.choose_correct_ports(flowlist)
            flows_per_sender[sender] = flowlist

        return flows_per_sender


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
                           default=0.2)

    parser.add_argument('-m', '--mice',
                           help='Percentage of mice flows',
                           type=float,
                           default=0.8)

    parser.add_argument('--elephant_rate',
                           help='Percentage of elephant flows',
                           type=float,
                           default=0.1)

    parser.add_argument('--mice_rate',
                           help='Percentage of mice flows',
                           type=float,
                           default=1.5)


    parser.add_argument('--target_load',
                           help='Target average link load per host',
                           type=float,
                           default=0.5)

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
    tg = TrafficGenerator(mice_rate=args.mice_rate, elephant_rate=args.elephant_rate,
                          target_link_load=args.target_load, pMice=args.mice, pElephant=args.elephant)

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
                traffic = tg.traffic_planner(senders=senders,receivers=receivers, totalTime=args.time, timeStep=args.time_step)

                msg = "Generating traffic:\n\tSenders: {0}\n\tReceivers: {1}\n\t"
                msg += "Total time: {2}\n\tTime step: {3}\n\tMice start rate: {4}\n\tElephant start rate: {5}\n\tTarget load: {6}"
                print msg.format(args.senders, args.receivers, args.time, args.time_step, args.mice_rate, args.elephant_rate, args.target_load)

                # If it must be saved
                if args.save_traffic:
                    msg = "Saving traffic file -> {0}"
                    filename = '{0}'.format(saved_traffic_folder)
                    filename += "tg2_{0}_to_{1}_m{2}e{3}_fre{4}_frm{5}_tl{6}_t{7}_ts{8}.traffic".format(','.join(senders), ','.join(receivers),
                                                                                      str(args.mice).replace('.', ','),
                                                                                      str(args.elephant).replace('.', ','),
                                                                                      str(args.elephant_rate).replace('.', ','),
                                                                                      str(args.mice_rate).replace('.', ','),
                                                                                      str(args.target_load).replace('.', ','),
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
# python trafficGenerator.py --flows_file file.txt
