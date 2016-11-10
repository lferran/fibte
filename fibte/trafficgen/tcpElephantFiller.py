from udpTrafficGeneratorBase import *

class TGFillerParser(TGParser):
    def __init__(self):
        super(TGFillerParser, self).__init__()

    def loadParser(self):
        # Load base arguments into the parser
        super(TGFillerParser, self).loadParser()

        # Load additional arguments
        self.parser.add_argument('--n_elephants', help='Number of elephant flows to maintain', type=int, default=16)

class tcpElephantFiller(udpTrafficGeneratorBase):
    def __init__(self, n_elephants=32, *args, **kwargs):
        super(tcpElephantFiller, self).__init__(*args, **kwargs)

        # Set target link load
        self.n_elephants = n_elephants
        self.mice_load = 0
        self.n_mice = 0

    def get_filename(self):
        """Return filename sample pattern"""
        pattern_args_fn = self.get_pattern_args_filename()
        filename = '{0}'.format(self.saved_traffic_dir)
        anames = ['tgf_tcp', '{0}', pattern_args_fn, 'nelep{1}', 't{2}', 'ts{3}']
        filename += '_'.join([a for a in anames if a != None])

        filename = filename.format(self.pattern,
                                   str(self.n_elephants),
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

    def _get_stopping_flows(self, all_flows, period):
        return {f_id: flow for f_id, flow in all_flows.iteritems()
                if flow.get('estimated_endtime') >= period
                and flow.get('estimated_endtime') < period + self.timeStep}

    def get_terminating_flows(self, all_flows, period):
        tfs = self._get_stopping_flows(all_flows, period)
        return tfs

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
        if self.n_elephants > 0:
            # Compute number of elephant flows and fixed sizes
            elep_per_host = int(self.n_elephants / float(len(self.senders)))

            # Start fws_per_host at each sender
            senders_t = self.senders[:]
            random.shuffle(senders_t)
            print("Number of senders: {0}".format(len(senders_t)))
            for sender in senders_t:
                for nf in range(elep_per_host):
                    # Get a new tcp elephant flow
                    min_duration = self.get_flow_duration(flow_type='e')
                    max_rate = LINK_BANDWIDTH
                    data_size = min_duration * max_rate

                    # Rate reduction
                    rate_reduction = random.uniform(0.7, 1)
                    estimated_rate = (LINK_BANDWIDTH/elep_per_host) * rate_reduction

                    # Random start time
                    new_starttime = random.uniform(5, 14.9)

                    # Compute estimated endtime
                    estimated_endtime = new_starttime + (data_size / (estimated_rate))

                    # Get a new destination
                    destination = self.get_flow_destination(sender)

                    fid = next_id
                    f = {'id': fid,
                         'type': 'e',
                         'srcHost': sender,
                         'dstHost': destination,
                         'proto': 'TCP',
                         'startTime': new_starttime,
                         'size': data_size,
                         'rate': LINK_BANDWIDTH,
                         'duration': None,
                         'estimated_endtime': estimated_endtime}

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
                        original_sender = tf['srcHost']

                        # Get a new tcp elephant flow
                        min_duration = self.get_flow_duration(flow_type='e')
                        max_rate = LINK_BANDWIDTH
                        data_size = min_duration * max_rate

                        # Rate reduction
                        rate_reduction = random.uniform(0.7, 1)
                        estimated_rate = (LINK_BANDWIDTH / elep_per_host) * rate_reduction

                        # Previous endtime
                        previous_endtime = tf.get('estimated_endtime')

                        # Random start time
                        new_starttime = random.uniform(previous_endtime + 0.5, previous_endtime + 2)

                        # Compute estimated endtime
                        estimated_endtime = new_starttime + (data_size / (estimated_rate))

                        # Get a new destination
                        destination = self.get_flow_destination(original_sender)

                        fid = next_id
                        f = {'id': fid,
                             'type': 'e',
                             'srcHost': original_sender,
                             'dstHost': destination,
                             'proto': 'TCP',
                             'startTime': new_starttime,
                             'size': data_size,
                             'rate': LINK_BANDWIDTH,
                             'duration': None,
                             'estimated_endtime': estimated_endtime}

                        flows_per_sender[original_sender].append(f)
                        all_elep_flows[fid] = f
                        next_id += 1

            print "Initial number of elephant flows: {0}".format(len(all_elep_flows))

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
    tgf = tcpElephantFiller(n_elephants=args.n_elephants,
                            pattern=args.pattern,
                            pattern_args=args.pattern_args,
                            totalTime=args.time,
                            timeStep=args.time_step)

    tgfparser.runArguments(tgf, args)
