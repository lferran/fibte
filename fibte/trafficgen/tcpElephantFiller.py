from udpTrafficGeneratorBase import *
from fibte.trafficgen import nonNICCongestionTest
from fibte.misc.flowEstimation import EstimateDemands

class TGFillerParser(TGParser):
    def __init__(self):
        super(TGFillerParser, self).__init__()

    def loadParser(self):
        # Load base arguments into the parser
        super(TGFillerParser, self).loadParser()

        # Load additional arguments
        self.parser.add_argument('--n_elephants', help='Number of elephant flows to maintain', type=int, default=0)

class tcpElephantFiller(udpTrafficGeneratorBase):
    def __init__(self, n_elephants=0, *args, **kwargs):
        super(tcpElephantFiller, self).__init__(*args, **kwargs)

        # Set target link load
        if n_elephants < 16 and n_elephants != 0 and not nonNICCongestionTest:
            print("ERROR: Can't generate less than 1 mice per host!")
            exit(0)
        else:
            self.n_elephants = n_elephants

    def get_filename(self):
        """Return filename sample pattern"""
        pattern_args_fn = self.get_pattern_args_filename()
        filename = '{0}'.format(self.saved_traffic_dir)
        anames = ['tgf_tcp', '{0}', pattern_args_fn, 'nelep{1}','mavg{2}' 't{3}', 'ts{4}']
        filename += '_'.join([a for a in anames if a != None])

        filename = filename.format(self.pattern,
                                   str(self.n_elephants),
                                   str(self.mice_avg).replace('.', ','),
                                   self.totalTime, self.timeStep)

        filename += '.traffic'
        return filename

    def _get_active_flows(self, all_flows, period, to_host=None, from_host=None):
        return self._get_active_flows_interval(all_flows, start=period, end=period+self.timeStep, to_host=to_host, from_host=from_host)

    def _get_active_flows_interval(self, all_flows, start, end, to_host=None, from_host=None):
        # Flows that start in this period
        all_active_flows = {fid: flow for fid, flow in all_flows.iteritems() if flow.get('startTime') <= start and start + self.getTCPFlowRemainingTime(flow) > end}

        if not to_host and not from_host:
            return all_active_flows

        elif to_host and not from_host:
            return {f_id: flow for f_id, flow in all_active_flows.iteritems() if flow['dst'] == to_host}

        elif from_host and not to_host:
            return {f_id: flow for f_id, flow in all_active_flows.iteritems() if flow['src'] == from_host}

        else:
            raise ValueError

    def _get_stopping_flows(self, all_flows, period):
        return {f_id: flow for f_id, flow in all_flows.iteritems()
                if flow.get('estimated_endtime') >= period
                and flow.get('estimated_endtime') < period + self.timeStep}

    def get_terminating_flows(self, all_flows, period):
        tfs = self._get_stopping_flows(all_flows, period)
        return tfs

    def flowToKey(self, flow):
        return {k: v for (k, v) in flow.iteritems() if k in ['src', 'dst', 'proto', 'startTime']}

    def getRandomRateReduction(self):
        """"""
        return random.uniform(1/3.0, 1/1.2)

    def updateEstimatedEndTimes(self, all_elep_flows, time_period):
        # Make an initial copy
        all_elep_fws_copy = copy.deepcopy(all_elep_flows)

        # Get active flows only at time time_period
        active_flows = self._get_active_flows(all_elep_flows, period=time_period)

        # Brand new estimateDemands object
        flowRates = EstimateDemands()

        # Add active flows to estimateDemands
        for fid, af in active_flows.iteritems():
            fkey = self.flowToKey(af)
            flowRates.addFlow(fkey)

        # Estimate all demands
        flowRates.estimateDemandsAll()

        # Update estimated endtimes accordingly
        for fid, flow in active_flows.iteritems():
            # Get new rate
            fkey = self.flowToKey(flow)
            new_rate = flowRates.getDemand(fkey) * LINK_BANDWIDTH

            # Get old rate
            old_rate = flow['rate']

            # Remaining data to send
            remaining_data = all_elep_fws_copy[fid]['remaining']

            # Substract from data to send what was sent in the last minute
            all_elep_fws_copy[fid]['remaining'] = max(0, remaining_data - (old_rate * self.timeStep))

            # Update new rate
            new_rate = new_rate * self.getRandomRateReduction()
            all_elep_fws_copy[fid]['rate'] = new_rate

            # Update new estimated duration
            estimated_duration = remaining_data/float(new_rate)
            all_elep_fws_copy[fid]['estimated_endtime'] = time_period + estimated_duration

        return all_elep_fws_copy

    def computeNonBlockingCompletionTimes(self, all_elep_flows):
        # Make an initial copy
        all_elep_fws_copy = copy.deepcopy(all_elep_flows)

        cts = {fid: 0 for fid in all_elep_fws_copy.iterkeys()}

        # Initialize
        for fid, fw in all_elep_fws_copy.iteritems():
            fw['rate'] = LINK_BANDWIDTH
            fw['remaining'] = fw['size']
            fw['estimated_endtime'] = fw['startTime'] + fw['remaining']/float(fw['rate'])

        for time_period in range(0, self.totalTime, self.timeStep):
            # Get active flows only at time time_period
            active_flows = self._get_active_flows(all_elep_fws_copy, period=time_period)

            # Brand new estimateDemands object
            flowRates = EstimateDemands()

            # Add active flows to estimateDemands
            for fid, af in active_flows.iteritems():
                fkey = self.flowToKey(af)
                flowRates.addFlow(fkey)

            # Estimate all demands
            flowRates.estimateDemandsAll()

            # Update estimated endtimes accordingly
            for fid, flow in active_flows.iteritems():
                # Get new rate
                fkey = self.flowToKey(flow)
                new_rate = flowRates.getDemand(fkey) * LINK_BANDWIDTH

                # Get old rate
                old_rate = flow['rate']

                # Remaining data to send
                remaining_data = all_elep_fws_copy[fid]['remaining']

                # Substract from data to send what was sent in the last minute
                all_elep_fws_copy[fid]['remaining'] = max(0, remaining_data - (old_rate * self.timeStep))

                # Update new rate
                all_elep_fws_copy[fid]['rate'] = new_rate

                # Update new estimated duration
                estimated_duration = remaining_data/float(new_rate)
                all_elep_fws_copy[fid]['estimated_endtime'] = time_period + estimated_duration

        # Write non-blocking completion times
        for fid, fw in all_elep_fws_copy.iteritems():
            cts[fid] = fw['estimated_endtime'] - fw['startTime']
        return cts

    def plan_elephant_flows(self):
        """
        """
        if nonNICCongestionTest:
            senders = [s for s in self.senders if '0' in s[-1] or '2' in s[-1]]
        else:
            senders = self.senders[:]

        # The resulting traffic is stored here
        flows_per_sender = {s: [] for s in senders}

        # Here we store all flows by id
        all_flows = {}

        # Here we store only the elephants
        all_elep_flows = {}

        # Initial flow id
        next_id = 0

        ## Schedule first elephant flows #############################
        if self.n_elephants > 0:
            # Compute number of elephant flows and fixed sizes
            elep_per_host = int(self.n_elephants / float(len(senders)))

            # Start fws_per_host at each sender
            senders_t = senders[:]
            random.shuffle(senders_t)
            print("Number of senders: {0}".format(len(senders_t)))
            for sender in senders_t:
                for nf in range(elep_per_host):
                    # Get a new tcp elephant flow
                    min_duration = self.get_flow_duration(flow_type='e')
                    max_rate = LINK_BANDWIDTH
                    data_size = min_duration * max_rate
                    max_rate *= self.getRandomRateReduction()
                    # Random start time
                    new_starttime = random.uniform(5, 14.9)
                    # Get a new destination
                    destination = self.get_flow_destination(sender)
                    fid = next_id
                    f = {'id': fid, 'type': 'e',
                         'src': sender, 'dst': destination,
                         'proto': 'tcp', 'startTime': new_starttime,
                         'size': data_size, 'remaining': data_size,
                         'rate': max_rate, 'duration': None, 'estimated_endtime': None
                         }
                    flows_per_sender[sender].append(f)
                    all_elep_flows[fid] = f
                    next_id += 1

            # Allocate elephant flows every time some of them stop
            for i in range(0, self.totalTime, self.timeStep):
                # Estimate new end times and rates
                all_elep_flows = self.updateEstimatedEndTimes(all_elep_flows, time_period=i)

                # Get flows that are terminating
                terminating_flows = self.get_terminating_flows(all_elep_flows, period=i)

                # Start as many flows as the ones that finish
                for tfk, tf in terminating_flows.iteritems():
                    # Get a new elephant flow with similar info
                    original_sender = tf['src']

                    # Get a new tcp elephant flow
                    min_duration = self.get_flow_duration(flow_type='e')
                    max_rate = LINK_BANDWIDTH
                    data_size = min_duration * max_rate
                    max_rate *= self.getRandomRateReduction()

                    # Previous endtime
                    previous_endtime = tf.get('estimated_endtime')

                    # Random start time
                    new_starttime = random.uniform(previous_endtime + 1, previous_endtime + 3)

                    # Get a new destination
                    destination = self.get_flow_destination(original_sender)

                    fid = next_id
                    f = {'id': fid,
                         'type': 'e',
                         'src': original_sender,
                         'dst': destination,
                         'proto': 'tcp',
                         'startTime': new_starttime,
                         'size': data_size,
                         'remaining': data_size,
                         'rate': max_rate,
                         'duration': None,
                         'estimated_endtime': None}

                    flows_per_sender[original_sender].append(f)
                    all_elep_flows[fid] = f
                    next_id += 1

            # Update all flows dict
            all_flows.update(all_elep_flows)

        nonBlockingCT = self.computeNonBlockingCompletionTimes(all_elep_flows)

        # Update the flows_per_sender dict
        new_flows_per_sender = {}
        for sender, flowlist in flows_per_sender.iteritems():
            new_flows_per_sender[sender] = []
            for flow in flowlist:
                if flow['id'] in all_flows.keys():
                    floww = all_flows[flow['id']]
                    floww['sport'] = -1
                    floww['dport'] = -1
                    floww['rate'] = LINK_BANDWIDTH
                    floww['remaining'] = floww['size']
                    floww['non-blocking-ct'] = nonBlockingCT[flow['id']]
                    new_flows_per_sender[sender].append(floww)

        # Re-write correct source and destination ports per each sender
        print("Choosing correct ports for flows...")
        flows_per_sender = self.choose_correct_src_dst_ports(new_flows_per_sender)
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
                            mice_avg=args.mice_avg,
                            totalTime=args.time,
                            timeStep=args.time_step)

    tgfparser.runArguments(tgf, args)
