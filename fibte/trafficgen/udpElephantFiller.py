from udpTrafficGeneratorBase import *

class TGFillerParser(TGParser):
    def __init__(self):
        super(TGFillerParser, self).__init__()

    def loadParser(self):
        # Load base arguments into the parser
        super(TGFillerParser, self).loadParser()

        # Load additional arguments
        self.parser.add_argument('--elephant_load', help='Level of elephant load', type=float, default=0.0)
        self.parser.add_argument('--n_elephants', help='Number of elephant flows to maintain', type=int, default=16)

class udpElephantFiller(udpTrafficGeneratorBase):
    def __init__(self, elephant_load=0.8, n_elephants=32, mice_load=0.2, n_mice=64, *args, **kwargs):
        super(udpElephantFiller, self).__init__(*args, **kwargs)

        # Set target link load
        self.elephant_load = elephant_load
        self.n_elephants = n_elephants
        self.mice_load = 0
        self.n_mice = 0

    def get_filename(self):
        """Return filename sample pattern"""
        pattern_args_fn = self.get_pattern_args_filename()
        filename = '{0}'.format(self.saved_traffic_dir)
        anames = ['tgf', '{0}', pattern_args_fn, 'eload{1}', 'nelep{2}', 'mload{3}', 'nmice{4}', 't{5}', 'ts{6}']
        filename += '_'.join([a for a in anames if a != None])

        filename = filename.format(self.pattern,
                                   str(self.elephant_load).replace('.', ','),
                                   str(self.n_elephants),
                                   str(self.mice_load).replace('.', ','),
                                   str(self.n_mice),
                                   self.totalTime, self.timeStep)

        filename += '.traffic'
        return filename

    def _get_active_flows(self, all_flows, period, to_host=None, from_host=None):
        return self._get_active_flows_interval(all_flows, start=period, end=period+self.timeStep, to_host=to_host, from_host=from_host)

    def _get_active_flows_interval(self, all_flows, start, end, to_host=None, from_host=None):
        # Flows that start in this period
        all_active_flows = {fid: flow for fid, flow in all_flows.iteritems() if flow.get('startTime') <= start and self._getFlowEndTime(flow) > end}

        if not to_host and not from_host:
            return all_active_flows

        elif to_host and not from_host:
            return {f_id: flow for f_id, flow in all_active_flows.iteritems() if flow['dstHost'] == to_host}

        elif from_host and not to_host:
            return {f_id: flow for f_id, flow in all_active_flows.iteritems() if flow['srcHost'] == from_host}

        else:
            raise ValueError

    def _get_starting_flows(self, all_flows, period, to_host=None, from_host=None):
        return self._get_starting_flows_interval(all_flows, start=period, end=period + self.timeStep, to_host=to_host, from_host=from_host)

    def _get_starting_flows_interval(self, all_flows, start, end, to_host=None, from_host=None):
        # Flows that start in this period
        all_starting_flows = {f_id: flow for f_id, flow in all_flows.iteritems() if
                              flow.get('startTime') >= start and flow.get('startTime') < end}

        if not to_host and not from_host:
            return all_starting_flows

        elif to_host and not from_host:
            return {f_id: flow for f_id, flow in all_starting_flows.iteritems() if flow['dstHost'] == to_host}

        elif from_host and not to_host:
            return {f_id: flow for f_id, flow in all_starting_flows.iteritems() if flow['srcHost'] == from_host}

        else:
            raise ValueError

    def _get_stopping_flows(self, all_flows, period):
        return {f_id: flow for f_id, flow in all_flows.iteritems()
                if self._getFlowEndTime(flow) >= period
                and self._getFlowEndTime(flow) < period + self.timeStep}

    def can_send_more(self, all_flows, sender, period):
        afs_sender = self._get_active_flows(all_flows, period=period, from_host=sender)
        aload = sum([f['size'] for f in afs_sender.itervalues()])
        min_eleph_size = ELEPHANT_SIZE_RANGE[0]
        max_elephant_load = LINK_BANDWIDTH * self.elephant_load
        return aload < max_elephant_load - min_eleph_size

    def can_receive_more(self, all_flows, receiver, period):
        active_flows = self._get_active_flows(all_flows, period=period, to_host=receiver)
        starting_flows = self._get_starting_flows(all_flows, period=period, to_host=receiver)

        # Get loads
        aload = sum([f['size'] for f in active_flows.itervalues()])
        sload = sum([f['size'] for f in starting_flows.itervalues()])
        max_eleph_load = LINK_BANDWIDTH * self.elephant_load
        can_receive_more = aload + sload < max_eleph_load
        if not can_receive_more:
            return False
        else:
            return True

    def receives_too_much_traffic(self, all_flows, receiver, period, type='elephant'):
        active_flows = self._get_active_flows(all_flows, period=period, to_host=receiver)
        starting_flows = self._get_starting_flows(all_flows, period=period, to_host=receiver)

        # Get loads
        aload = sum([f['size'] if f['proto'] == 'UDP' else f['rate'] for f in active_flows.itervalues()])
        sload = sum([f['size'] if f['proto'] == 'UDP' else f['rate'] for f in starting_flows.itervalues()])
        if type == 'elephant':
            max_load = LINK_BANDWIDTH * self.elephant_load
        else:
            max_load = LINK_BANDWIDTH * self.mice_load
        return aload + sload > max_load

    def get_hosts_that_can_afford(self, all_flows, flow, period, type='elephant'):
        can_afford = []
        for host in self.senders:
            if host != flow['dstHost'] and host != flow['srcHost']:
                active_flows = self._get_active_flows(all_flows, period=period, to_host=host)
                starting_flows = self._get_starting_flows(all_flows, period=period, to_host=host)

                # Get loads
                aload = sum([f['size'] for f in active_flows.itervalues()])
                sload = sum([f['size'] for f in starting_flows.itervalues()])

                if type == 'elephant':
                    max_load = LINK_BANDWIDTH * self.elephant_load
                else:
                    max_load = LINK_BANDWIDTH * self.mice_load

                if aload + sload + flow['size'] < max_load:
                    can_afford.append(host)

        return can_afford

    def get_terminating_flows(self, all_flows, period):
        tfs = self._get_stopping_flows(all_flows, period)
        return tfs

    def max_load_reached(self, all_flows, sender, init, end, type='elephant'):
        starting_fws = self._get_starting_flows_interval(all_flows, init, end, from_host=sender)
        total_load = sum([f['size'] if f['proto'] == 'UDP' else f['rate'] for fid, f in starting_fws.iteritems()])
        if type == 'elephant':
            max_load = LINK_BANDWIDTH * self.elephant_load
        else:
            max_load = LINK_BANDWIDTH * self.mice_load

        return total_load >= max_load

    def get_full_receivers(self, all_flows, start, end):
        full_receivers = []

        for receiver in self.senders:
            # Get active flows to receiver in this period of time
            active_flows = self._get_active_flows_interval(all_flows, start, end, to_host=receiver)

            # Calculate the sum
            total_load = sum([fw.get('size') for f, fw in active_flows.iteritems()])

            # Append it if full
            if total_load >= LINK_BANDWIDTH*self.elephant_load:
                full_receivers.append(receiver)

        return full_receivers

    def plan_flows(self):
        """
        """
        # The resulting traffic is stored here
        flows_per_sender = {s: [] for s in self.senders}

        # Here we store all flows by id
        all_flows = {}

        # Here we store only the elephants
        all_elep_flows = {}

        # Initial flow id
        next_id = 0

        ## Schedule first elephant flows #############################
        if self.elephant_load > 0.0:
            # Compute number of elephant flows and fixed sizes
            elep_per_host = int(self.n_elephants / float(len(self.senders)))
            elep_size = (self.elephant_load * LINK_BANDWIDTH) / elep_per_host

            # Start fws_per_host at each sender
            senders_t = self.senders[:]
            random.shuffle(senders_t)
            print("Number of senders: {0}".format(len(senders_t)))
            for sender in senders_t:
                for nf in range(elep_per_host):
                    # Get a new elephant flow
                    # size = self.get_flow_size(flow_type='e', distribution='uniform')
                    size = elep_size
                    duration = self.get_flow_duration(flow_type='e')
                    # duration = 20
                    full_receivers = self.get_full_receivers(all_elep_flows, start=0.5, end=14.9)
                    destination = self.get_flow_destination(sender, exclude=full_receivers)
                    if not destination:
                        print("{0} can't start more flows! No receivers available!".format(sender))
                        break

                    fid = next_id
                    f = {'id': fid,
                         'type': 'e',
                         'srcHost': sender,
                         'dstHost': destination,
                         'proto': 'UDP',
                         'startTime': random.uniform(5, 14.9),
                         'size': size,
                         'rate': size,
                         'duration': duration}
                    flows_per_sender[sender].append(f)
                    all_elep_flows[fid] = f
                    next_id += 1

            # Allocate elephant flows
            for i in range(15, self.totalTime, self.timeStep):
                if i < self.totalTime - self.timeStep:
                    # Get flows that are terminating
                    terminating_flows = self.get_terminating_flows(all_elep_flows, period=i)

                    # Start as many flows as the ones that finish
                    for tfk, tf in terminating_flows.iteritems():
                        # Get a new elephant flow with similar info
                        # size = self.get_flow_size(flow_type='e', distribution='uniform')
                        original_sender = tf['srcHost']
                        size = elep_size
                        duration = self.get_flow_duration(flow_type='e')
                        # duration = 20
                        old_endtime = int(tf.get('startTime') + tf.get('duration'))
                        new_starttime = random.uniform(old_endtime + 0.5, old_endtime + 2)
                        full_receivers = self.get_full_receivers(all_elep_flows, start=old_endtime, end=old_endtime+self.timeStep)
                        destination = self.get_flow_destination(original_sender, exclude=full_receivers)
                        if not destination:
                            print("{0} can't start more flows! No receivers available!".format(original_sender))
                            continue

                        fid = next_id
                        f = {'id': fid,
                             'type': 'e',
                             'srcHost': original_sender,
                             'dstHost': destination,
                             'proto': 'UDP',
                             'startTime': new_starttime,
                             'size': size,
                             'rate': size,
                             'duration': duration}
                        flows_per_sender[original_sender].append(f)
                        all_elep_flows[fid] = f
                        next_id += 1

            print "Initial number of elephant flows: {0}".format(len(all_elep_flows))

            # # Check receivers!
            # for i in range(0, self.totalTime, self.timeStep):
            #     for receiver in self.senders:
            #         # Check if can receive all new starting flows towards him
            #         if self.receives_too_much_traffic(all_elep_flows, receiver, period=i, type='elephant'):
            #             # Get starting flows
            #             starting_flows = self._get_starting_flows(all_elep_flows, period=i, to_host=receiver)
            #
            #             # Get flow ids
            #             starting_ids = starting_flows.keys()
            #
            #             # Randomly shuffle them
            #             random.shuffle(starting_ids)
            #
            #             while starting_ids != []:
            #                 # Get flow id to reallocate
            #                 fid_reallocate = starting_ids.pop()
            #                 flow = starting_flows[fid_reallocate]
            #
            #                 # Get receivers that can afford it!
            #                 possible_receivers = self.get_hosts_that_can_afford(all_elep_flows, flow, period=i,
            #                                                                     type='elephant')
            #
            #                 if not possible_receivers:
            #                     # Just remove flow forever
            #                     all_elep_flows.pop(fid_reallocate)
            #                     starting_flows.pop(fid_reallocate)
            #
            #                 else:
            #                     # Choose one randomly
            #                     new_receiver = random.choice(possible_receivers)
            #
            #                     # Change receiver
            #                     all_elep_flows[fid_reallocate]['dstHost'] = new_receiver
            #
            #                 # Check if already fits
            #                 if self.can_receive_more(all_elep_flows, receiver, period=i):
            #                     # Alreaady fits!
            #                     break
            #         else:
            #             continue

            #print "After-reallocation/removal number of elephant flows: {0}".format(len(all_elep_flows))

            # Update all flows dict
            all_flows.update(all_elep_flows)

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
        print("Choosing correct ports for flows...")
        flows_per_sender = self.choose_corrent_src_dst_ports(new_flows_per_sender)
        return flows_per_sender


if __name__ == "__main__":

    # Get the TGParser
    tgfparser = TGFillerParser()
    tgfparser.loadParser()

    # Parse args
    args = tgfparser.parseArgs()

    # Start the TG object
    tgf = udpElephantFiller(elephant_load=args.elephant_load,
                                    n_elephants=args.n_elephants,
                                    pattern=args.pattern,
                                    pattern_args=args.pattern_args,
                                    totalTime=args.time,
                                    timeStep=args.time_step)

    # Start counting time
    t = time.time()
    if args.terminate:
        print "Terminating ongoing traffic!"
        tgf.terminateTrafficAtHosts()

    else:
        # Check if flow file has been given
        if not args.flows_file:

            # If traffic must be loaded
            if args.load_traffic:
                msg = "Loading traffic from file <- {0}"
                print msg.format(args.load_traffic)

                # Fetch traffic from file
                traffic = pickle.load(open(args.load_traffic, "r"))

                # Convert hostnames to current ips
                traffic = tgf.changeTrafficHostnamesToIps(traffic)

            else:
                # Generate traffic
                traffic = tgf.plan_flows()

                print "Generating traffic"
                # msg = "Generating traffic\n\tArguments: {0}\n\t"
                # print msg.format(args)

                # If it must be saved
                if args.save_traffic:
                    if not args.file_name:
                        filename = tgf.get_filename()
                    else:
                        filename = args.file_name

                    # Log it
                    msg = "Saving traffic file -> {0}"
                    print msg.format(filename)

                    # Convert current ip's to hostnames
                    traffic_to_save = tgf.changeTrafficIpsToHostnames(traffic)

                    with open(filename, "w") as f:
                        pickle.dump(traffic_to_save, f)

            # Orchestrate the traffic (either loaded or generated)
            print "Scheduling traffic..."
            tgf.schedule(traffic, add=args.addtc)

        # Flow file has been given
        else:
            print "Scheduling flows specified in {0}".format(args.flows_file)
            # Generate traffic from flows file
            traffic = tgf.plan_from_flows_file(args.flows_file)

            # Schedule it
            tgf.schedule(traffic, add=args.addtc)

    print "Elapsed time ", time.time() - t

# Example commandline call:
# python udpTrafficGenerator2.py --pattern bijection --mice_rate 0.25 --elephant_rate 1 --time 50 --save_traffic
# python udpTrafficGenerator2.py --terminate
# python udpTrafficGenerator2.py --load_traffic saved_traffic/
# python udpTrafficGenerator2.py --flows_file file.txt
