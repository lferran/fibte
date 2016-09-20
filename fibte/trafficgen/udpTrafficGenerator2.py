from udpTrafficGeneratorBase import *

class TG2Parser(TGParser):
    def __init__(self):
        super(TG2Parser, self).__init__()

    def loadParser(self):
        # Load base arguments into the parser
        super(TG2Parser, self).loadParser()

        # Load additional arguments
        self.parser.add_argument('--elephant_rate', help='Rate at which elephant flows start at any host (flows/s)', type=float, default=0.1)
        self.parser.add_argument('--mice_rate', help='Rate at which mice flows start at any host (flows/s)', type=float, default=1.5)
        self.parser.add_argument('--target_load', help='Target average link load per host', type=float,default=0.5)

class udpTrafficGenerator2(udpTrafficGeneratorBase):
    def __init__(self, mice_rate=1.5, elephant_rate=0.1, target_link_load=0.5, *args, **kwargs):
        super(udpTrafficGenerator2, self).__init__(*args, **kwargs)

        # Set target link load
        self.target_relative_load = target_link_load
        self.target_link_load = LINK_BANDWIDTH*target_link_load

        # Set specific flow starting rates
        self.elephant_rate = elephant_rate
        self.mice_rate = mice_rate

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

    def get_filename(self):
        """Return filename sample pattern"""
        pattern_args_fn = self.get_pattern_args_filename()
        filename = '{0}'.format(self.saved_traffic_dir)
        anames = ['tg2', '{0}', pattern_args_fn, 'fm{1}', 'fe{2}', 'tl{3}', 't{4}', 'ts{5}']
        filename += '_'.join([a for a in anames if a != None])

        filename = filename.format(self.pattern,
                        str(self.mice_rate).replace('.', ','),
                        str(self.elephant_rate).replace('.', ','),
                        str(self.target_relative_load).replace('.', ','),
                        self.totalTime, self.timeStep)

        filename += '.traffic'
        return filename

    def _get_active_flows(self, all_flows, period):
        return {fid: flow for fid, flow in all_flows.iteritems() if flow['startTime'] <= period and flow['startTime'] + flow['duration'] >= period}

    def _get_starting_flows(self, all_flows, period):
        return {f_id: flow for f_id, flow in all_flows.iteritems() if flow.get('startTime') >= period and flow.get('startTime') < period + self.timeStep}

    def plan_flows(self):
        """
        * Creates elephant and mice flows at certain specified rates for a poisson process
        * Then it checks that any host is never sending more than its link capacity
        * Then it checks that any host is never receiving more than its link capacity
        * Chooses correct ports so that they are not repeated at any point in time

        :return:
        """
        # The resulting traffic is stored here
        flows_per_sender = {}

        # Initial flow id
        next_id = 0

        # We first obtain the flow start times for elephants at each host
        elephants_times = {sender: self.get_poisson_times(self.elephant_rate, self.totalTime) for sender in self.senders}
        # Generate elephants times first
        for sender, time_list in elephants_times.iteritems():
            list_of_dicts = []
            for index, starttime in enumerate(time_list):
                # Convert it in a list of dicts with unique id for each flow
                size = self.get_flow_size(flow_type='e')
                duration = self.get_flow_duration(flow_type='e')
                destination = self.get_flow_destination(sender)
                list_of_dicts.append({'startTime': starttime, 'id': next_id + index,
                                      'srcHost': sender, 'dstHost': destination,
                                      'type': 'e', 'size': size, 'duration': duration})
            flows_per_sender[sender] = list_of_dicts
            next_id += len(time_list)

        # Create dict indexed by flow id {}: id -> flow
        all_flows = {flow['id']: flow for sender, flowlist in flows_per_sender.iteritems() for flow in flowlist}

        # Now we generate start times for mice flows
        mice_times = {sender: self.get_poisson_times(self.mice_rate, self.totalTime) for sender in self.senders}
        for sender, time_list in mice_times.iteritems():
            list_of_dicts = []
            for index, starttime in enumerate(time_list):
                # Convert it in a list of dicts with unique id for each flow
                size = self.get_flow_size(flow_type='m')
                duration = self.get_flow_duration(flow_type='m')
                destination = self.get_flow_destination(sender)
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

        print "Initial number of flows: {0}".format(len(all_flows))

        # Check first transmitted load from each host
        for period in range(0, self.totalTime, self.timeStep):
            # Get active and starting flows at that period of time
            active_flows = self._get_active_flows(all_flows, period)
            starting_flows = self._get_starting_flows(all_flows, period)

            # Iterate senders
            for sender in self.senders:
                # Get active load tx at sender
                active_load = sum([all_flows[id]['size'] for id in active_flows.keys() if all_flows[id]['srcHost'] == sender])

                #Get new load at sender
                starting_load = sum([all_flows[id]['size'] for id in starting_flows.keys() if all_flows[id]['srcHost'] == sender])

                # Virtual load
                vload = active_load + starting_load

                # Get all starting flow ids and shuffle them
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

        # Make a copy of senders
        receivers = self.senders[:]

        for i in range(2):
            # Check then received loads at each host
            for period in range(0, self.totalTime, self.timeStep):
                # Get active and starting flows at that period of time
                active_flows = self._get_active_flows(all_flows, period)
                starting_flows = self._get_starting_flows(all_flows, period)

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

                        # Get all starting flow ids and shuffle them
                        sf_ids = starting_flows.keys()[:]
                        random.shuffle(sf_ids)

                        while vload > LINK_BANDWIDTH and sf_ids != []:
                            # Remove one new flow at random and recompute
                            id_to_reallocate = sf_ids[0]
                            sf_ids.remove(id_to_reallocate)

                            # Choose another destination
                            to_exclude = [rcv]
                            sender = all_flows[id_to_reallocate]['srcHost']
                            new_dst = self.get_flow_destination(sender, exclude=to_exclude)

                            if new_dst != None:
                                loads_new_dst = sum([all_flows[id]['size'] for id in active_flows.keys() if
                                                     all_flows[id]['dstHost'] == new_dst])

                                if loads_new_dst + all_flows[id_to_reallocate]['size'] > LINK_BANDWIDTH:
                                    while loads_new_dst + all_flows[id_to_reallocate]['size'] > LINK_BANDWIDTH and new_dst != None:
                                        # Add dest to to_exclude
                                        to_exclude += [new_dst]

                                        # Choose new dst
                                        new_dst = self.get_flow_destination(sender, exclude=to_exclude)

                                        # Recompute loads new dst
                                        loads_new_dst = sum([all_flows[id]['size'] for id in active_flows.keys() if
                                                             all_flows[id]['dstHost'] == new_dst])

                                    # If we flow could not be fit anywhere:
                                    if new_dst == None:
                                        # Remove flow forever
                                        all_flows.pop(id_to_reallocate)
                                        starting_flows.pop(id_to_reallocate)

                                    else:
                                        # Reallocate
                                        all_flows[id_to_reallocate]['dstHost'] = new_dst
                                else:
                                    # Reallocate
                                    all_flows[id_to_reallocate]['dstHost'] = new_dst
                            else:
                                # Remove flow forever
                                all_flows.pop(id_to_reallocate)
                                starting_flows.pop(id_to_reallocate)

                            # Recompute
                            try:
                                starting_load = sum([all_flows[id]['size'] for id in starting_flows.keys() if all_flows[id]['dstHost'] == rcv])
                            except:
                                import ipdb; ipdb.set_trace()

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

    # Get the TGParser
    tg2parser = TG2Parser()
    tg2parser.loadParser()

    # Parse args
    args = tg2parser.parseArgs()

    # Start the TG object
    tg2 = udpTrafficGenerator2(mice_rate=args.mice_rate,
                               elephant_rate=args.elephant_rate,
                               target_link_load=args.target_load,
                               pattern=args.pattern,
                               pattern_args=args.pattern_args,
                               totalTime=args.time,
                               timeStep=args.time_step)

    # Start counting time
    t = time.time()
    if args.terminate:
        print "Terminating ongoing traffic!"
        tg2.terminateTraffic()

    else:
        # Check if flow file has been given
        if not args.flows_file:

            # If traffic must be loaded
            if args.load_traffic:
                msg = "Loading traffic from file <- {0}"
                print msg.format(args.load_traffic)

                # Fetch traffic from file
                traffic = pickle.load(open(args.load_traffic,"r"))

                import ipdb; ipdb.set_trace()

                # Convert hostnames to current ips
                traffic = tg2.changeTrafficHostnamesToIps(traffic)

            else:
                # Generate traffic
                traffic = tg2.plan_flows()

                msg = "Generating traffic\n\tArguments: {0}\n\t"
                print msg.format(args)

                # If it must be saved
                if args.save_traffic:
                    msg = "Saving traffic file -> {0}"
                    filename = tg2.get_filename()
                    print msg.format(filename)

                    # Convert current ip's to hostnames
                    traffic_to_save = tg2.changeTrafficIpsToHostnames(traffic)

                    with open(filename,"w") as f:
                        pickle.dump(traffic_to_save,f)

            # Orchestrate the traffic (either loaded or generated)
            print "Scheduling traffic..."
            tg2.schedule(traffic)

        # Flow file has been given
        else:
            print "Scheduling flows specified in {0}".format(args.flows_file)
            # Generate traffic from flows file
            traffic = tg2.plan_from_flows_file(args.flows_file)

            # Schedule it
            tg2.schedule(traffic)

    print "Elapsed time ", time.time()-t


# Example commandline call:
# python udpTrafficGenerator2.py --pattern bijection --mice_rate 0.25 --elephant_rate 1 --time 50 --save_traffic
# python udpTrafficGenerator2.py --terminate
# python udpTrafficGenerator2.py --load_traffic saved_traffic/
# python udpTrafficGenerator2.py --flows_file file.txt
