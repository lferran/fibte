from udpTrafficGeneratorBase import *

class TGFillerParser(TGParser):
    def __init__(self):
        super(TGFillerParser, self).__init__()

    def loadParser(self):
        # Load base arguments into the parser
        super(TGFillerParser, self).loadParser()

        # Load additional arguments
        self.parser.add_argument('--elephant_load', help='Level of elephant load', type=float, default=0.8)
        self.parser.add_argument('--mice_load', help='Level of mice load', type=float, default=0.2)

class udpTrafficGeneratorFiller(udpTrafficGeneratorBase):
    def __init__(self, elephant_load=0.8, mice_load=0.2, *args, **kwargs):
        super(udpTrafficGeneratorFiller, self).__init__(*args, **kwargs)

        # Set target link load
        self.elephant_load = elephant_load
        self.mice_load = mice_load

    def get_filename(self):
        """Return filename sample pattern"""
        pattern_args_fn = self.get_pattern_args_filename()
        filename = '{0}'.format(self.saved_traffic_dir)
        anames = ['tgf', '{0}', pattern_args_fn, 'el{1}', 'ml{2}', 't{2}', 'ts{3}']
        filename += '_'.join([a for a in anames if a != None])

        filename = filename.format(self.pattern,
                        str(self.elephant_load).replace('.', ','),
                        str(self.mice_load).replace('.', ','),
                        self.totalTime, self.timeStep)

        filename += '.traffic'
        return filename

    def _get_active_flows(self, all_flows, period, to_host=None, from_host=None):
        if not to_host and not from_host:
            return {fid: flow for fid, flow in all_flows.iteritems() if flow['startTime'] <= period and flow['startTime'] + flow['duration'] >= period}
        elif to_host and not from_host:
            return {fid: flow for fid, flow in all_flows.iteritems() if flow['dstHost'] == to_host and
                    flow['startTime'] <= period and flow['startTime'] + flow['duration'] >= period}
        elif from_host and not to_host:
            return {fid: flow for fid, flow in all_flows.iteritems() if flow['srcHost'] == from_host and
                    flow['startTime'] <= period and flow['startTime'] + flow['duration'] >= period}
        else:
            raise ValueError

    def _get_starting_flows(self, all_flows, period, to_host=None, from_host=None):
        if not to_host and not from_host:
            return {f_id: flow for f_id, flow in all_flows.iteritems() if flow.get('startTime') >= period and flow.get('startTime') < period + self.timeStep}
        elif to_host and not from_host:
            return {f_id: flow for f_id, flow in all_flows.iteritems() if flow['dstHost'] == to_host and
                    flow.get('startTime') >= period and flow.get('startTime') < period + self.timeStep}
        elif from_host and not to_host:
            return {f_id: flow for f_id, flow in all_flows.iteritems() if flow['srcHost'] == from_host and
                    flow.get('startTime') >= period and flow.get('startTime') < period + self.timeStep}
        else:
            raise ValueError

    def can_send_more(self, all_flows, sender, period):
        afs_sender = self._get_active_flows(all_flows, period=period, from_host=sender)
        aload = sum([f['size'] for f in afs_sender.itervalues()])
        min_eleph_size = ELEPHANT_SIZE_RANGE[0]
        max_elephant_load = LINK_BANDWIDTH*self.elephant_load
        return aload < max_elephant_load - min_eleph_size

    def can_receive_more(self, all_flows, receiver, period):
        active_flows = self._get_active_flows(all_flows, period=period, to_host=receiver)
        starting_flows = self._get_starting_flows(all_flows, period=period, to_host=receiver)
        # Get loads
        aload = sum([f['size'] for f in active_flows.itervalues()])
        sload = sum([f['size'] for f in starting_flows.itervalues()])
        max_eleph_load = LINK_BANDWIDTH*self.elephant_load
        return aload + sload <= max_eleph_load

    def get_hosts_that_can_afford(self, all_flows, flow, period):
        can_afford = []
        for host in self.senders:
            if host != flow['dstHost'] and host != flow['srcHost']:
                active_flows = self._get_active_flows(all_flows, period=period, to_host=host)
                starting_flows = self._get_starting_flows(all_flows, period=period, to_host=host)

                # Get loads
                aload = sum([f['size'] for f in active_flows.itervalues()])
                sload = sum([f['size'] for f in starting_flows.itervalues()])

                max_eleph_load = LINK_BANDWIDTH*self.elephant_load

                if aload + sload + flow['size'] < max_eleph_load:
                    can_afford.append(host)

        return can_afford

    def plan_flows(self):
        """
        * Creates elephant and mice flows at certain specified rates for a poisson process
        * Then it checks that any host is never sending more than its link capacity
        * Then it checks that any host is never receiving more than its link capacity
        * Chooses correct ports so that they are not repeated at any point in time

        :return:
        """

        # The resulting traffic is stored here
        flows_per_sender = {s: [] for s in self.senders}

        # Here we store all flows by id
        all_elep_flows = {}

        # Initial flow id
        next_id = 0

        ## Schedule first elephant flows #############################
        if self.elephant_load > 0.0:

            # Allocate elephant flows
            for i in range(0, self.totalTime, self.timeStep):
                # Iterate all senders
                for sender in self.senders:
                    # Check if they can send some more elephant flows
                    if self.can_send_more(all_elep_flows, sender, period=i):
                        # Get a new elephant flow
                        size = self.get_flow_size(flow_type='e', distribution='constant')
                        duration = self.get_flow_duration(flow_type='e')
                        destination = self.get_flow_destination(sender)
                        fid = next_id
                        f = {'startTime': random.uniform(i, i+1), 'id': fid,
                             'srcHost': sender, 'dstHost': destination,
                             'type': 'e', 'size': size, 'duration': duration}
                        flows_per_sender[sender].append(f)
                        all_elep_flows[fid] = f
                        next_id += 1
                    else:
                        continue

            print "Initial number of elephant flows: {0}".format(len(all_elep_flows))

            # Check receivers!
            for i in range(0, self.totalTime, self.timeStep):
                for receiver in self.senders:
                    # Check if can receive all new starting flows towards him
                    if not self.can_receive_more(all_elep_flows, receiver, period=i):
                        # Get starting flows
                        starting_flows = self._get_starting_flows(all_elep_flows, period=i, to_host=receiver)

                        # Get flow ids
                        starting_ids = starting_flows.keys()

                        # Randomly shuffle them
                        random.shuffle(starting_ids)

                        while starting_ids != []:
                            # Get flow id to reallocate
                            fid_reallocate = starting_ids.pop()
                            flow = starting_flows[fid_reallocate]

                            # Get receivers that can afford it!
                            possible_receivers = self.get_hosts_that_can_afford(all_elep_flows, flow, period=i)

                            if not possible_receivers:
                                # Just remove flow forever
                                all_elep_flows.pop(fid_reallocate)
                                starting_flows.pop(fid_reallocate)

                            else:
                                # Choose one randomly
                                new_receiver = random.choice(possible_receivers)

                                # Change receiver
                                all_elep_flows[fid_reallocate]['dstHost'] = new_receiver

                            # Check if already fits
                            if self.can_receive_more(all_elep_flows, receiver, period=i):
                                # Alreaady fits!
                                break

                    else:
                        continue

            print "After-reallocation/removal number of elephant flows: {0}".format(len(all_elep_flows))

        # Update the flows_per_sender dict
        new_flows_per_sender = {}
        for sender, flowlist in flows_per_sender.iteritems():
            new_flows_per_sender[sender] = []
            for flow in flowlist:
                if flow['id'] in all_elep_flows.keys():
                    floww = all_elep_flows[flow['id']]
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
    tgfparser = TGFillerParser()
    tgfparser.loadParser()

    # Parse args
    args = tgfparser.parseArgs()

    # Start the TG object
    tgf = udpTrafficGeneratorFiller(elephant_load=args.elephant_load,
                                    mice_load=args.mice_load,
                                    pattern=args.pattern,
                                    pattern_args=args.pattern_args,
                                    totalTime=args.time,
                                    timeStep=args.time_step)
    
    # Start counting time
    t = time.time()
    if args.terminate:
        print "Terminating ongoing traffic!"
        tgf.terminateTraffic()

    else:
        # Check if flow file has been given
        if not args.flows_file:

            # If traffic must be loaded
            if args.load_traffic:
                msg = "Loading traffic from file <- {0}"
                print msg.format(args.load_traffic)

                # Fetch traffic from file
                traffic = pickle.load(open(args.load_traffic,"r"))

                # Convert hostnames to current ips
                traffic = tgf.changeTrafficHostnamesToIps(traffic)

            else:
                # Generate traffic
                traffic = tgf.plan_flows()

                msg = "Generating traffic\n\tArguments: {0}\n\t"
                print msg.format(args)

                # If it must be saved
                if args.save_traffic:
                    msg = "Saving traffic file -> {0}"
                    filename = tgf.get_filename()
                    print msg.format(filename)

                    # Convert current ip's to hostnames
                    traffic_to_save = tgf.changeTrafficIpsToHostnames(traffic)

                    with open(filename,"w") as f:
                        pickle.dump(traffic_to_save,f)

            # Orchestrate the traffic (either loaded or generated)
            print "Scheduling traffic..."
            tgf.schedule(traffic)

        # Flow file has been given
        else:
            print "Scheduling flows specified in {0}".format(args.flows_file)
            # Generate traffic from flows file
            traffic = tgf.plan_from_flows_file(args.flows_file)

            # Schedule it
            tgf.schedule(traffic)

    print "Elapsed time ", time.time()-t

# Example commandline call:
# python udpTrafficGenerator2.py --pattern bijection --mice_rate 0.25 --elephant_rate 1 --time 50 --save_traffic
# python udpTrafficGenerator2.py --terminate
# python udpTrafficGenerator2.py --load_traffic saved_traffic/
# python udpTrafficGenerator2.py --flows_file file.txt