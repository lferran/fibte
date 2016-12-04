from fibte import LINK_BANDWIDTH

class EstimateDemandError(Exception):
    pass

class EstimationFlow(object):
    def __init__(self, demand, maxSize):
        self.demand = demand
        self.converged = False

        # Indicates that the flow is limited by the receiver.
        # When False, means the flow is limited by the sender
        self.rl = True

        # If maxSize equals to 1: means it is a non-rate-limited flow
        self.maxSize = maxSize
        self.sizeLimited = False

    def __repr__(self):
        return "Demand: {0}, Converged: {1}".format(self.demand, self.converged)

class EstimateDemands(object):
    """Estimates flow demands"""
    def __init__(self):
        # Flow estimation matrix represented by two
        # dictionaries so its faster to compute
        self.senders = {}
        self.receivers = {}

        # Dicitonary with all the flows we keep track of.
        # We need that to get the exact size of a flow.
        self.currentFlows= {}

    @staticmethod
    def flowToKey(flow):
        return tuple(sorted(flow.items()))

    @staticmethod
    def keyToFlow(self, fkey):
        return dict(fkey)

    def clear(self):
        """Set to initial state
        """
        self.senders.clear()
        self.receivers.clear()
        self.currentFlows.clear()

    def __repr__(self):
        """"""
        s = ''
        for src, destinations in self.senders.iteritems():
            s += "\t {0}  ".format(src)
            for dst,flows in destinations.iteritems():
                s+= "{0} :({1})  ".format(dst, [x.demand for x in flows])
            s +="\n"
        return s

    def nonBlockingBandwidth(self):
        """
        """
        bandwidth = 0
        for flow in self.currentFlows.itervalues():
            bandwidth += flow.demand
        return bandwidth

    def getDemand(self, flow):
        """Return estimated demand of flow"""
        if isinstance(flow, dict):
            estimation = self.currentFlows.get(self.flowToKey(flow), None)

        #assuming its already transformed
        elif isinstance(flow, tuple):
            estimation = self.currentFlows.get(flow, None)

        if not estimation:
            raise EstimateDemandError

        else:
            return estimation.demand

    def addFlow(self, flow):
        """Add new flow to the estimation matrix"""
        src = flow["src"]
        dst = flow["dst"]
        maxSize = flow.get('rate', LINK_BANDWIDTH)/LINK_BANDWIDTH

        # Add flow at senders dict
        if not src in self.senders:
            self.senders[src] = {dst: [EstimationFlow(1, maxSize)]}

        else:
            if not dst in self.senders[src]:
                self.senders[src][dst] = [EstimationFlow(1, maxSize)]

            else:
                self.senders[src][dst].append(EstimationFlow(1, maxSize))

        # Add flow in receivers dictionary
        if not dst in self.receivers:
            self.receivers[dst] = {src: self.senders[src][dst]}

        else:
            if not src in self.receivers[dst]:
                self.receivers[dst][src] = self.senders[src][dst]

        # Add flow into self.currentFlows
        self.currentFlows[self.flowToKey(flow)] = self.senders[src][dst][-1]

        return src

    def delFlow(self,flow):
        """Delete flow from estimation matrix"""
        src = flow["src"]
        dst = flow["dst"]

        estimation = self.currentFlows.pop(self.flowToKey(flow), None)

        # Makes sure that it exists when erased
        if src in self.senders:
            if dst in self.senders[src]:
                self.senders[src][dst].remove(estimation)

                # If empty list: remove it
                if not self.senders[src][dst]:
                    del(self.senders[src][dst])
                if not self.receivers[dst][src]:
                    del(self.receivers[dst][src])

                # Check if there is anything: delete if necessary
                if not self.senders[src]:
                    del(self.senders[src])
                if not self.receivers[dst]:
                    del(self.receivers[dst])

        return src

    def delFlow2(self,flow):
        """Deletes flow from estimation matrix"""
        src = flow["src"]
        dst = flow["dst"]

        # Makes sure that it exists when erased
        if src in self.senders:
            if dst in self.senders[src]:
                self.senders[src][dst].pop()

                # If empty list: remove it
                if not self.senders[src][dst]:
                    del(self.senders[src][dst])
                if not self.receivers[dst][src]:
                    del(self.receivers[dst][src])

                # Check if there is anything
                if not self.senders[src]:
                    del(self.senders[src])
                if not self.receivers[dst]:
                    del(self.receivers[dst])

        self.currentFlows.pop(self.flowToKey(flow), None)
        return src

    def unconvergeRow(self, src):
        """Sets all flows of a row to not converged"""
        if src in self.senders:
            for flows in self.senders[src].itervalues():
                for flow in flows:
                    flow.converged = False

    def unconvergeColumn(self, dst):
        """Sets all flows of a column to not converged"""
        if dst in self.receivers:
            for flows in self.receivers[dst].itervalues():
                for flow in flows:
                    flow.converged = False

    def unconvergeAll(self):
        """Unconverge all rows and columns"""
        for senders in self.senders.itervalues():
            for flows in senders.itervalues():
                for flow in flows:
                    flow.converged = False
                    flow.sizeLimited = False

    def update(self):
        self.unconvergeAll()

    def estimateSenders(self):
        """Iterate the rows of the matrix increasing load when needed"""
        something_changed = False

        # Iterate rows
        for sender, receivers in self.senders.iteritems():
            # Accumulate the total converged load
            converged_load = 0

            # Unconverged number of flows
            unconverged_flows = 0

            # Collect converged load and unconverged flows
            for flows in receivers.itervalues():
                for flow in flows:
                    if flow.converged or flow.sizeLimited:
                        converged_load += flow.demand
                    else:
                        unconverged_flows += 1

            # Still some unconverged flows
            if unconverged_flows != 0:
                # Compute flows equal share
                equal_share = (1.0 - converged_load)/unconverged_flows

                # Check if there is any UDP limited flow
                something_was_size_limited = True
                while something_was_size_limited:
                    something_was_size_limited = False
                    unconverged_flows = 0
                    for flows in receivers.itervalues():
                        for flow in flows:
                            if flow.maxSize < equal_share and not flow.converged:
                                if not flow.sizeLimited:
                                    flow.sizeLimited = True
                                    converged_load += flow.maxSize
                                    something_was_size_limited = True
                                    something_changed = True
                                    flow.demand = flow.maxSize
                            elif not flow.converged:
                                unconverged_flows += 1

                    # Recompute equal share with what is left!
                    if unconverged_flows != 0:
                        equal_share = (1.0 - converged_load)/unconverged_flows

            # Allocate remaining flows
            for flows in receivers.itervalues():
                for flow in flows:
                    if not flow.converged and not flow.sizeLimited:
                        if flow.demand != equal_share:
                            something_changed = True
                            flow.demand = equal_share

        return something_changed

    def estimateReceivers(self):
        """Iterate columns of the matrix by converging flows at the receiving side"""
        something_changed = False
        for receiver, senders in self.receivers.iteritems():
            total_demand = 0
            sender_limited_demand = 0
            number_received_limited_flows = 0
            for flows in senders.itervalues():
                for flow in flows:
                    flow.rl = True
                    total_demand += flow.demand
                    number_received_limited_flows += 1

            # If total demand at the receiver is smaller than one
            # means all these flows converged
            if total_demand <= 1.0:
                continue

            equal_share = 1.0/number_received_limited_flows

            some_rl_set_false = True
            while some_rl_set_false:
                some_rl_set_false = False
                number_received_limited_flows = 0
                for flows in senders.itervalues():
                    for flow in flows:
                        if flow.rl:
                            if flow.demand < equal_share:
                                sender_limited_demand += flow.demand
                                flow.rl = False
                                some_rl_set_false = True
                            else:
                                number_received_limited_flows += 1

                if number_received_limited_flows != 0:
                    equal_share = (1.0 - sender_limited_demand) / number_received_limited_flows

            for flows in senders.itervalues():
                for flow in flows:
                    if flow.rl:
                        if flow.demand != equal_share:
                            something_changed = True
                            flow.demand = equal_share
                        flow.converged = True

        return something_changed

    def estimateDemandsAll(self):
        """Estimate demands of all current flows in the matrix"""
        # Set all rows&cols to unconverged
        self.update()

        # Iterate until nothing changes
        something_changed_senders = self.estimateSenders()
        something_changed_receivers = self.estimateReceivers()
        while something_changed_receivers or something_changed_senders:
            something_changed_senders = self.estimateSenders()
            something_changed_receivers = self.estimateReceivers()

    def estimateDemands(self, flow, action="add"):
        """Add or remove a single flow and re-estimate the demands"""
        if action == "add":
            self.addFlow(flow)
            self.update()

        elif action == "del":
            self.delFlow(flow)
            self.update()

        # Iterate until nothing changes
        something_changed_senders = self.estimateSenders()
        something_changed_receivers = self.estimateReceivers()
        while something_changed_receivers or something_changed_senders:
            something_changed_senders = self.estimateSenders()
            something_changed_receivers = self.estimateReceivers()