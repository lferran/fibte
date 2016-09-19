from udpTrafficGenerator import *

class TG1Parser(TGParser):
    def __init__(self):
        super(TG1Parser, self).__init__()

    def loadParser(self):
        # Load base arguments into the parser
        super(TG1Parser, self).loadParser()

        # Load additional arguments
        self.parser.add_argument('-r', '--flow_rate',
                                 help="Rate at which a host starts new flows (flows/second)",
                                 type=float,
                                 default=0.25)

        self.parser.add_argument('-e', '--elephant',
                                 help='Percentage of elephant flows',
                                 type=float,
                                 default=0.1)

        self.parser.add_argument('-m', '--mice',
                                 help='Percentage of mice flows',
                                 type=float,
                                 default=0.9)

class udpTrafficGenerator1(udpTrafficGeneratorBase):
    def __init__(self, flowRate=2, pMice = 0.9, pElephant = 0.1, *args, **kwargs):
        super(udpTrafficGenerator1, self).__init__(*args, **kwargs)

        # Set debug level
        log.setLevel(logging.DEBUG)

        # Set flow rate
        self.flowRate = flowRate

        # Set traffic mice/elephant flow percent
        self.pMice = pMice
        self.pElephant = pElephant


    def get_filename(self):
        """Return filename sample pattern"""
        filename = '{0}'.format(self.saved_traffic_dir)
        anames = ['tg1', '{2}', 'm{3}e{4}', 'fr{5}', 't{6}', 'ts{7}']
        filename += '_'.join(anames)
        filename.format(args.pattern,
                        str(args.mice).replace('.', ','),
                        str(args.elephant).replace('.', ','),
                        str(args.flow_rate).replace('.', ','),
                        args.time, args.time_step)
        filename += '.traffic'
        return filename

    def get_random_flow_type(self):
        return self.weighted_choice([('m', self.pMice), ('e', self.pElephant)])

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

    def restrict_receiver_load(self, all_flows, starting_flows, active_flows, all_receivers):

        # Get receivers of starting flows
        receivers = list({all_flows[id]['dstHost'] for id in starting_flows.keys()})

        # Iterate receivers
        for rcv in receivers:
            # Get starting flows to receiver
            receiver_starting_flows = {id: all_flows[id] for id in starting_flows.keys() if all_flows[id]['dstHost'] == rcv}

            # Check how much are receiving rigth now
            current_load = sum([all_flows[id]['size'] for id in active_flows.keys() if all_flows[id]['dstHost'] == rcv])

            # Get new loads
            new_load = sum([flow['size'] for (fid, flow) in receiver_starting_flows.iteritems()])

            # Check if overload link
            if new_load + current_load > LINK_BANDWIDTH:

                # Starting flow ids
                starting_flow_ids = receiver_starting_flows.keys()[:]
                random.shuffle(starting_flow_ids)

                while new_load + current_load > LINK_BANDWIDTH and starting_flow_ids != []:
                    #print "New load + Current load to {0} still bigger than LINK BANDWIDTH".format(rcv)

                    # Pick one of the flows at random
                    flow_change_dst = starting_flow_ids[0]
                    #print "Length of starting flows ids: {0} ".format(len(starting_flow_ids))
                    #print "Changing destination for one flow {0}".format(flow_change_dst)

                    starting_flow_ids.remove(flow_change_dst)

                    # Compute all other possible receivers
                    all_receivers_t = list(set(all_receivers) - {rcv})

                    # Get new destination host
                    new_dst = self.get_flow_destination(all_flows[flow_change_dst]['srcHost'], all_receivers_t, all_flows[flow_change_dst]['type'])
                    #print "Picked new destination for flow {0}: {1}".format(flow_change_dst, new_dst)

                    # Get loads of destination host
                    loads_new_dst = sum([all_flows[id]['size'] for id in active_flows.keys() if all_flows[id]['dstHost'] == new_dst])

                    # Check if it fits though
                    if loads_new_dst + all_flows[flow_change_dst]['size'] > LINK_BANDWIDTH:
                        #print "Flow doesn't fit in other destination either..."

                        # Remove new chosen one from all possible receivers
                        all_receivers_t = list(set(all_receivers_t) - {new_dst})

                        while loads_new_dst + all_flows[flow_change_dst]['size'] > LINK_BANDWIDTH and all_receivers_t != [] and new_dst != None:
                            # Pick another destination
                            #print "Length of all_receivers_t: {0}".format(len(all_receivers_t))

                            # Get new destination host
                            new_dst = self.get_flow_destination(all_flows[flow_change_dst]['srcHost'], all_receivers_t, all_flows[flow_change_dst]['type'])

                            #print "Choosing another one... {0}".format(new_dst)
                            all_receivers_t = list(set(all_receivers_t) - {new_dst})

                            # Get loads of destination host
                            loads_new_dst = sum([all_flows[id]['size'] for id in active_flows.keys() if all_flows[id]['dstHost'] == new_dst])

                        if all_receivers_t != [] and new_dst != None:
                            # Found one receiver
                            #print "A receiver was found for flow {0}: {1}".format(flow_change_dst, new_dst)
                            # Change destination in all data structures
                            all_flows[flow_change_dst]['dstHost'] = new_dst
                            receiver_starting_flows.pop(flow_change_dst)

                        else:
                            # Receiver not found
                            #print "Flow {0} could not be found any destination that would fit him, so we remove it".format(flow_change_dst)
                            # Remove flow then
                            all_flows.pop(flow_change_dst)
                            starting_flows.pop(flow_change_dst)
                            receiver_starting_flows.pop(flow_change_dst)

                    else:
                        #print "Flow fits in new chosen destination!"
                        #print "A receiver was found for flow {0}: {1}".format(flow_change_dst, new_dst)
                        # Change destination in all data structures
                        all_flows[flow_change_dst]['dstHost'] = new_dst
                        receiver_starting_flows.pop(flow_change_dst)

                    # Re-compute starting flows to receiver
                    receiver_starting_flows = {id: all_flows[id] for id in starting_flows.keys() if all_flows[id]['dstHost'] == rcv}
                    new_load = sum([flow['size'] for (fid, flow) in receiver_starting_flows.iteritems()])

                if starting_flow_ids != []:
                    # Flows allocated so that the load for that receiver is ok
                    #print "Some flows to {0} were re-allocated/removed! and now load restriction is preserved".format(rcv)
                    pass
                else:
                    # Problem
                    #print "Receiver can't hold any more flows: all were removed/reallocated"
                    pass
            else:
                #print "Loads fit in - NOP"
                pass

        return all_flows

    def restrict_sizes(self, all_flows, starting_flows, active_flows):
        """All flows: Dictionary of flows keyed by id"""
        # Get new senders
        senders = list({all_flows[id]['srcHost'] for id in starting_flows.keys()})

        # Assign flows to senders
        sender_starting_flows = {sender: {id: all_flows[id] for id in starting_flows.keys() if all_flows[id]['srcHost'] == sender} for sender in senders}

        # Accumulate how much they are sending right now
        current_traffic = {}
        for sender in senders:
            current_load = sum([all_flows[id]['size'] for id in active_flows.keys() if all_flows[id]['srcHost'] == sender])
            current_traffic[sender] = current_load

        # Make a copy that we iterate
        sender_starting_flows_copy = sender_starting_flows.copy()

        for sender, sfws in sender_starting_flows_copy.iteritems():

            # Give initial sizes to mice flows first
            initial_sizes = [(fid, self.get_flow_size(sflow['type'])) for (fid, sflow) in sender_starting_flows[sender].iteritems()]

            # Update them in all_flows dict
            for (fid, size) in initial_sizes:
                all_flows[fid]['size'] = size

            # Recompute current load of the sender
            current_load = sum([all_flows[id]['size'] for id in active_flows.keys()+starting_flows.keys() if all_flows[id]['srcHost'] == sender])

            # Fetch elephant id's from the starting ones
            elephant_ids = [id for id in starting_flows.keys() if all_flows[id]['type'] == 'e' and all_flows[id]['srcHost'] == sender]

            # While the sum of the elephants exceed...
            while current_load > LINK_BANDWIDTH and elephant_ids != []:
                # Decrease sizes for each elephant flow starting
                for eid in elephant_ids:
                    current_size = all_flows[eid]['size']
                    new_size = current_size * 0.9
                    # Check if it's still an elephant flow
                    if new_size >= ELEPHANT_SIZE_RANGE[0] and new_size <= ELEPHANT_SIZE_RANGE[1]:
                        all_flows[eid]['size'] = new_size
                    else:
                        #print "Epa! We had to remove elephant!"
                        # Need to remove elephant flow from starting flows and all flows
                        all_flows.pop(eid)
                        starting_flows.pop(eid)
                        sender_starting_flows[sender].pop(eid)

                # Recompute load and elephant ids
                current_load = sum([all_flows[id]['size'] for id in active_flows.keys() + starting_flows.keys() if all_flows[id]['srcHost'] == sender])
                elephant_ids = [id for id in starting_flows.keys() if all_flows[id]['type'] == 'e' and all_flows[id]['srcHost'] == sender]

        return all_flows

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

    def _get_active_flows_2(self, all_flows, period):
        return {fid: flow for fid, flow in all_flows.iteritems() if flow['startTime'] <= period and flow['startTime'] + flow['duration'] >= period}

    def _get_starting_flows(self, all_flows, period):
        """
        Returns a dictionary of all the flows that are starting in the current time period.
        Dictionary is keyed by flow id
        """

        # Filter out the visited ones
        non_visited_flows = {f_id: flow for f_id, flow in all_flows.iteritems() if flow.get("type") == None}

        # Filter out the ones that do not start in the period
        starting_flows = {f_id: flow for f_id, flow in non_visited_flows.iteritems()
                          if flow.get('startTime') >= period and flow.get('startTime') < period + self.timeStep}

        return starting_flows

    def _get_starting_flows_2(self, all_flows, period):
        return {f_id: flow for f_id, flow in all_flows.iteritems() if flow.get('startTime') >= period and flow.get('startTime') < period + self.timeStep}

    def plan_flows(self):
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
        flows_per_sender_tmp = {sender: self.get_poisson_times(self.flowRate, self.totalTime) for sender in senders}

        # Initial flow id
        next_id = 0

        # Convert start times list into dict
        for sender, time_list in flows_per_sender_tmp.iteritems():
            # Convert it in a list of dicts with unique id for each flow
            list_of_dicts = [{'startTime': starttime, 'id': next_id + index, 'srcHost': sender, 'type': None} for
                             index, starttime in enumerate(time_list)]
            next_id += len(time_list)
            flows_per_sender[sender] = list_of_dicts

        # Create dict indexed by flow id {}: id -> flow
        all_flows = {flow['id']: flow for sender, flowlist in flows_per_sender.iteritems() for flow in flowlist}

        for period in range(0, self.totalTime, self.timeStep):

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
                    flow_type = self.get_random_flow_type()
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

                # Set flow type
                flow['type'] = flow_type

                # Get flow duration
                flow_duration = self.get_flow_duration(flow_type)

                # Put flow temporary size
                # flow_size = self.get_flow_size(flow_type)
                flow_size = -1

                # Sender
                snd = flow['srcHost']

                # Choose receiver
                receiver = self.get_flow_destination(snd)

                # Update flow
                all_flows[f_id] = {'srcHost': flow['srcHost'],
                                   'dstHost': receiver,
                                   'size': flow_size,
                                   'type': flow_type,
                                   'startTime': flow['startTime'],
                                   'duration': flow_duration,
                                   'sport': -1, 'dport': -1}

            # Restrict flow sizes so that any host sends more traffic than link capacity
            all_flows = self.restrict_sizes(all_flows, starting_flows, active_flows)

            # Restriction: any host receives more traffic than LINK_BANDWIDTH
            # all_flows = self.restrict_receiver_load(all_flows, starting_flows, active_flows, all_receivers=receivers)

        # Final check on receiving load restrictions
        for period in range(0, self.totalTime, self.timeStep):

            # Get active flows in current period
            active_flows = self._get_active_flows_2(all_flows, period)

            # Get next starting flows
            starting_flows = self._get_starting_flows_2(all_flows, period)

            # Iterate hosts
            for rcv in receivers:
                # Compute current load of the link
                rcv_current_load = sum(
                    [all_flows[id]['size'] for id in active_flows.keys() if all_flows[id]['dstHost'] == rcv])

                # Compute new flows to receiver
                rcv_new_flows = {id: all_flows[id] for id in starting_flows.keys() if all_flows[id]['dstHost'] == rcv}
                rcv_new_flows_load = sum([flow['size'] for (fid, flow) in rcv_new_flows.iteritems()])

                if rcv_current_load + rcv_new_flows_load > LINK_BANDWIDTH:
                    rcv_new_em_flows = [fid for fid, flow in rcv_new_flows.iteritems()]
                    random.shuffle(rcv_new_em_flows)

                    while rcv_current_load + rcv_new_flows_load > LINK_BANDWIDTH and rcv_new_em_flows != []:
                        # Pick one elephant flow
                        flow_to_remove_id = rcv_new_em_flows[0]
                        rcv_new_em_flows.remove(flow_to_remove_id)

                        # Remove it from starting flows and all flows
                        starting_flows.pop(flow_to_remove_id)
                        all_flows.pop(flow_to_remove_id)

                        # Recompute new flow load
                        rcv_new_flows = {id: all_flows[id] for id in starting_flows.keys() if
                                         all_flows[id]['dstHost'] == rcv}
                        rcv_new_flows_load = sum([flow['size'] for (fid, flow) in rcv_new_flows.iteritems()])

                    if rcv_new_em_flows == []:
                        print "All starting flows had to be removed!"
                else:
                    continue

        # Update the flows_per_sender dict
        new_flows_per_sender = {}
        for sender, flowlist in flows_per_sender.iteritems():
            new_flows_per_sender[sender] = []
            for flow in flowlist:
                if flow['id'] in all_flows.keys():
                    new_flows_per_sender[sender].append(all_flows[flow['id']])

        # Re-write correct source and destination ports per each sender
        flows_per_sender = {}
        for sender, flowlist in new_flows_per_sender.iteritems():
            flowlist = self.choose_correct_ports(flowlist)
            flows_per_sender[sender] = flowlist

        return flows_per_sender

if __name__ == "__main__":

    # Get the TGParser
    tg1parser = TG1Parser()
    tg1parser.loadParser()

    args = tg1parser.parseArgs()

    # Start the TG object
    tg1 = udpTrafficGenerator1(flowRate=args.flow_rate, pMice=args.mice, pElephant=args.elephant,
                              pattern=args.pattern, totalTime=args.time, timeStep=args.time_step)
    # Start counting time
    t = time.time()
    if args.terminate:
        print "Terminating ongoing traffic!"
        tg1.terminateTraffic()

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
                traffic = tg1.changeTrafficHostnamesToIps(traffic)

            else:
                # Prepare senders and receivers
                senders = args.senders.split(",")
                receivers = args.receivers.split(",")

                # Generate traffic
                traffic = tg1.plan_flows()

                msg = "Generating traffic:\n\t"
                msg += "Pattern: {0}\n\tFlow rate: {1}\n\t"
                msg += "Mice(%): {2}\n\tElephant(%): {3}\n\t"
                msg += "Total time: {2}\n\tTime step: {5}"
                print msg.format(args.pattern, args.flow_rate, args.mice, args.elephants, args.time, args.time_step)

                # If it must be saved
                if args.save_traffic:
                    msg = "Saving traffic file -> {0}"
                    filename = tg1.get_filename()
                    print msg.format(filename)

                    # Convert current ip's to hostnames
                    traffic_to_save = tg1.changeTrafficIpsToHostnames(traffic)

                    with open(filename,"w") as f:
                        pickle.dump(traffic_to_save, f)

            # Orchestrate the traffic (either loaded or generated)
            print "Scheduling traffic..."
            tg1.schedule(traffic)

        # Flow file has been given
        else:
            print "Scheduling flows specified in {0}".format(args.flows_file)
            traffic = tg1.plan_from_flows_file(args.flows_file)
            tg1.schedule(traffic)

    print "Elapsed time ", time.time()-t


# Example commandline call:
# python trafficGenerator1.py --senders pod_0,pod_1 --receivers pod_2,pod_3 --mice 0.8 --elephant 0.2 --flow_rate 0.25 --randomized --time 300 --save_traffic
# python trafficGenerator1.py --terminate
# python trafficGenerator1.py --load_traffic saved_traffic/
# python trafficGenerator1.py --flows_file file.txt

