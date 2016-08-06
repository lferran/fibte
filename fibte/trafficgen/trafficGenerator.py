from flow import Flow, Base
from fibte.misc.topologyGraph import TopologyGraph
import random
import sched
import time
import requests
import os
import subprocess
import bisect
from requests.exceptions import ConnectionError
try:
    import cPickle as pickle
except:
    import pickle

import json

from fibte.misc.unixSockets import UnixClient

#import inspect

from fibte import CFG, LINK_BANDWIDTH

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
    def __init__(self, pMice = 0.9, pElephant = 0.1, remoteHosts=False, *args, **kwargs):
        super(TrafficGenerator, self).__init__(*args,**kwargs)

        # Set traffic mice/elephant flow percent
        self.pMice = pMice
        self.pElephant = pElephant

        self.topology = TopologyGraph(getIfindexes = False, openFlowInformation = False, db = os.path.join(tmp_files,db_topo))

        self.linkBandwidth = LINK_BANDWIDTH

        # Used to communicate with flowServers at the hosts.
        # {0} is because it will be filled with whichever server we want to talk to!
        self.unixClient = UnixClient(tmp_files+"flowServer_{0}")

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
        # Size ranges in % of link bandwidth
        mice_size_step = 0.001
        mice_size_range = [0.002, 0.01+mice_size_step]
        elephant_size_step = 0.05
        elephant_size_range = [0.1, 0.6+elephant_size_step]

        # Size ranges in bits per second
        mice_size_range_bps = map(lambda x: int(x*self.linkBandwidth), mice_size_range)
        elephant_size_range_bps = map(lambda x: int(x*self.linkBandwidth), elephant_size_range)

        if flow_type == 'e':
            return random.choice(range(elephant_size_range_bps[0], elephant_size_range_bps[1], int(self.linkBandwidth*elephant_size_step)))
        elif flow_type == 'm':
            return random.choice(range(mice_size_range_bps[0], mice_size_range_bps[1], int(self.linkBandwidth * mice_size_step)))
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
                        start_time=flow_tmp['startTime'], duration=flow_tmp['duration'],
                        tos=flow_tmp['tos'], proto=flow_tmp['proto'])

            # Append it to the list
            flowlist.append(flow)

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

    def trafficPlanner(self, senders=["r_0_e0"], receivers=["r_1_e0"], flowRate=0.25, totalTime=500):

        # Parse communication pattern limitations
        #TODO: do it per pod: senders=[pod0, pod1], receivers=[pod3]
        if senders[0] == 'all':
            senders = self.topology.getEdgeRouters()

        if receivers[0] == "all":
            receivers = self.topology.getEdgeRouters()

        # Maintains the used ports: {}: host -> {in, out, usedPorts}
        hosts_capacities = {}
        for host in set(senders+receivers):
            hosts_capacities[host] = {'in': 0, 'out': 0 , 'usedPorts' : set({})}

        # Holds {}: host -> flowlist
        traffic_per_host = {}

        for sender in senders:
            # Generate flowlist
            flowlist = self.plan_flows(sender, receivers, flowRate, totalTime)

            # Save it
            traffic_per_host[sender] = flowlist

        # Return all generated traffic
        return traffic_per_host

    #@profile
    def trafficPlanner2(self,numFlows=50,senders=["r_0_e0"],receivers=["r_1_e0"],percentageMice=0.9, percentageElephants=0.1,totalTime=500):


        #First we will compute the length and size ranges of mice and elephant flows. This two parameters will depend
        #on link bandwidth and total time.

        time_range_elephant = [20,500]
        time_range_mice = [2,5]


        #This function should think which flows should be scheduled during totalTime. It should create a list
        #of flows that more or less at all time the number of current "running" flows is equal or close to numFlows.

        #senders should be the list of "edge" or hosts that can generate traffic. And receivers the edge or list of host
        #that can receive traffic in our network.

        #Mice and elephatn parameters are the percentage of numFlows that are mice or elephant.

        #the function should take into account the links Bandwidths and maximum number of paths between nodes (dont really know how to use that because there are collisions everywhere....)

        #for every host we have we will track the current capacity for receiving and sending traffic so we do not generate more than what they can generate or receive.

        if senders[0] == "all":
            senders = self.topology.getEdgeRouters()

        if receivers[0] == "all":
            receivers = self.topology.getEdgeRouters()

        hosts_capacities={}
        #hosts_capacities_receivers = {}
        senders_tmp = []
        for router in senders:
            for host in self.topology.getHostsBehindRouter(router):
                senders_tmp.append(host)
        senders = senders_tmp

        receivers_tmp = []
        for router in receivers:
            for host in self.topology.getHostsBehindRouter(router):
                receivers_tmp.append(host)
        receivers = receivers_tmp

        for host in set(senders+receivers):
            hosts_capacities[host] = {'in':0,'out':0 , 'usedPorts' : set({})}


        #We make the first allocation of flows using the number of flows we want to keep.
        flows = []

        #we use the number of elephant flows allocated to limit how many do we allocate
        maxElephantFlows = int(percentageElephants*numFlows) + 1

        #we keep track of the amount of bytes/kb/mb/gb generated by each type of flows.
        #with that we will be able to allocate the size of the flows following datacenter traffic matrices.
        elephantTrafficCounter = 0

        current_times = []
        #fill the first value with a random number from 0 to 10 so not all the flows start at the same time
        for i in range(numFlows):
            current_times.append((random.randint(0,3),0))


        numRounds = 5
        #first round of allocation

        temporal_port_list = []
        for flow_n in range(numFlows):
            #we get the type of flow we will allocate
            #we repeat this 5 times and we break if a flow is found.
            for round in xrange(numRounds):
                #chooses if the flow will be elephant or mice. However, we have a maximum amount of elephant flows.
                if elephantTrafficCounter > maxElephantFlows:
                    type = "m"
                else:
                    type = self.weighted_choice(percentageMice,percentageElephants)

                #select a sender and receiver and avoids to choose two in the same subnetwork
                #if elephant
                # #getting sender and receiver
                # sender = random.choice(senders)
                # receiver = random.choice(receivers)
                # while self.topology.inSameSubnetwork(sender,receiver):
                #     receiver = random.choice(receivers)

                if type == 'e':
                    size = self.getSizeElephant()
                    duration = self.getTimeElephant()
                    starting_time = current_times[flow_n][0]

                #if mice
                elif type == 'm':
                    size = self.getSizeMice()
                    duration = self.getTimeMice()
                    starting_time = current_times[flow_n][0]

                # compute some temporal senders and receivers removing senders or receivers that can not fit the path

                senders_tmp = []
                for sender in senders:
                    if hosts_capacities[sender]["out"] + size <= self.linkBandwidth:
                        senders_tmp.append(sender)
                if senders_tmp:
                    sender = random.choice(senders_tmp)
                else:
                    sender = []

                receivers_tmp = []
                for receiver in receivers:
                    if hosts_capacities[receiver]["in"] + size <= self.linkBandwidth:
                        receivers_tmp.append(receiver)
                if receivers_tmp:
                    receiver = random.choice(receivers_tmp)
                else:
                    receiver = []

                #if some of them are empty it means that we have to use the previous technique
                if not(receiver) or not(sender):

                    sender = random.choice(senders)
                    receiver = random.choice(receivers)
                    while self.topology.inSameSubnetwork(sender,receiver):
                        sender = random.choice(senders)
                        receiver = random.choice(receivers)

                else:

                    #we use that to limit
                    c = 0
                    while self.topology.inSameSubnetwork(sender,receiver) and c < numRounds:
                        c +=1
                        receiver = random.choice(receivers_tmp)
                        sender = random.choice(senders_tmp)

                    #we did not find it
                    if c == numRounds:
                        if round == numRounds-1:
                            current_times[flow_n] = (current_times[flow_n][0]+duration,'noAllocated')
                            break
                        else:
                            sender = random.choice(senders)
                            receiver = random.choice(receivers)
                            while self.topology.inSameSubnetwork(sender,receiver):
                                receiver = random.choice(receivers)


                #we increase the current time even if the flow can not be allocated.
                #we check if the flow can be allocated or not.

                #if the flow will finish later than the simulation time is not scheduled.
                #or the sender can not fit this flow in its link
                #or the receiver can not fit this flow in its link

                #changed my mind when a flow is to big we will just adapt the duration or size to the remaining

                #however we still have to check if the link is completly full (lets say 0.95% of its capacity), and
                #the remaining time is bigger than the minimum for this type of flow.

                #flow should not be allocated however we try to fit duration and size
                if current_times[flow_n][0]+duration > totalTime or hosts_capacities[sender]['out'] + size > self.linkBandwidth \
                        or hosts_capacities[receiver]['in'] + size > self.linkBandwidth:


                    #remaining duration
                    #update current_times
                    #not allocated flows are not added to the flows to be scheduled list
                    if round == numRounds-1:
                        current_times[flow_n] = (current_times[flow_n][0]+duration,'noAllocated')
                        break
                        #why not a break here
                        #break

                    #here we adjust the time to fit
                    if current_times[flow_n][0]+duration > totalTime:
                        duration = totalTime - current_times[flow_n][0]

                        if type == 'm':
                            #if the remainging time its smaller than the minimum of a mice we do not allocate
                            if duration < time_range_mice[0]:
                                #current_times[flow_n] = (current_times[flow_n]+duration,'noAllocated')
                                continue
                        if type =='e':
                            #same for elephants
                            if duration < time_range_elephant[0]:
                                #current_times[flow_n] = (current_times[flow_n]+duration,'noAllocated')
                                continue

                    #here we ajust the size I check because we could be here only for the time condition
                    if hosts_capacities[sender]['out'] + size > self.linkBandwidth or hosts_capacities[receiver]['in'] + size > self.linkBandwidth:

                        #lets find which one has the least space for flow allocation and adapt to that..
                        size = min(self.linkBandwidth - hosts_capacities[sender]['out'], self.linkBandwidth-hosts_capacities[receiver]['in'])

                        #it there is less than the 5 % of the link we do not allocate
                        if size < self.linkBandwidth*0.05:

                            #current_times[flow_n] = (current_times[flow_n]+duration,'noAllocated')
                            continue

                        #we reduce that a 10%
                        size = int(size*0.90)


                    #we allocate that modified flow

                    hosts_capacities[sender]['out'] += size
                    hosts_capacities[receiver]['in'] += size
                    flow = self.randomFlow(srcHost=sender, dstHost=receiver, size=size, startTime=starting_time, duration=duration,hosts_capacities=hosts_capacities)

                    hosts_capacities[sender]['usedPorts'].add(flow['sport'])
                    hosts_capacities[receiver]['usedPorts'].add(flow['dport'])
                    temporal_port_list.append((flow["sport"],flow["dport"]))
                    flows.append(flow)

                    #update current_times
                    current_times[flow_n] = (current_times[flow_n][0]+duration,flow)

                    #if its long enough we consider it elephant
                    if flow['duration'] >= time_range_elephant[0]:
                        elephantTrafficCounter +=1
                    #Break the loop because a flow was found
                    break



                #allocate flow
                else:
                    #we add the size of the flow | so we are allocating it.
                    hosts_capacities[sender]['out'] += size
                    hosts_capacities[receiver]['in'] += size
                    flow = self.randomFlow(srcHost=sender, dstHost=receiver, size=size, startTime=starting_time, duration=duration,hosts_capacities=hosts_capacities)

                    hosts_capacities[sender]['usedPorts'].add(flow['sport'])
                    hosts_capacities[receiver]['usedPorts'].add(flow['dport'])
                    temporal_port_list.append((flow["sport"],flow["dport"]))
                    flows.append(flow)


                    #update current_times
                    current_times[flow_n] = (current_times[flow_n][0]+duration,flow)

                    #if its long enough we consider it elephant
                    if flow['duration'] >= time_range_elephant[0]:
                        elephantTrafficCounter +=1
                    #Break the loop because a flow was found
                    break

        #now we start allocating flows until all reach time > totalTime
        #for that we will use the list current_times, sort it and erase the first flow and
        #schedule a new one.




        #sorting the list
        current_times.sort()


        #while there is a flow that can be fitted...
        while current_times[0][0] < totalTime:

            #here we do something similar to what we did in the first round but we only schedule 1 flow per round, and
            #we erase 1 flow per round also.

            #first we take that flow and update host_capacities, and try to allocate a new flow

            flow = current_times[0][1]
            if flow != 'noAllocated':

                srcHost = self.topology.getHostName(flow["src"])
                dstHost = self.topology.getHostName(flow["dst"])
                size = flow["size"]

                #free space for new flows
                hosts_capacities[srcHost]['out'] -= size
                hosts_capacities[dstHost]['in'] -= size

                #if its elephant we make 1 slot free
                #if its long enough we consider it elephant
                if flow['duration'] >= time_range_elephant[0]:
                    elephantTrafficCounter -=1

                #print hosts_capacities

                hosts_capacities[srcHost]['usedPorts'].remove(flow['sport'])
                hosts_capacities[dstHost]['usedPorts'].remove(flow['dport'])


            #we look for a new flow
            for round in xrange(numRounds):

                if elephantTrafficCounter >= maxElephantFlows:
                    type = "m"

                else:
                    type = self.weighted_choice(percentageMice,percentageElephants)

                # #getting sender and receiver
                # sender = random.choice(senders)
                # receiver = random.choice(receivers)
                # while self.topology.inSameSubnetwork(sender,receiver):
                #     receiver = random.choice(receivers)

                #if elephant
                if type == 'e':
                    size = self.getSizeElephant()
                    duration = self.getTimeElephant()

                #if mice
                elif type == 'm':
                    size = self.getSizeMice()
                    #print "mice"
                    duration = self.getTimeMice()


                # compute some temporal senders and receivers removing senders or receivers that can not fit the path

                senders_tmp = []
                for sender in senders:
                    if hosts_capacities[sender]["out"] + size <= self.linkBandwidth:
                        senders_tmp.append(sender)
                if senders_tmp:
                    sender = random.choice(senders_tmp)
                else:
                    sender = []

                receivers_tmp = []
                for receiver in receivers:
                    if hosts_capacities[receiver]["in"] + size <= self.linkBandwidth:
                        receivers_tmp.append(receiver)
                if receivers_tmp:
                    receiver = random.choice(receivers_tmp)
                else:
                    receiver = []

                #if some of them are empty it means that we have to use the previous technique
                if not(receiver) or not(sender):

                    sender = random.choice(senders)
                    receiver = random.choice(receivers)
                    while self.topology.inSameSubnetwork(sender,receiver):
                        receiver = random.choice(receivers)


                else:

                    #we use that to limit
                    c = 0
                    while self.topology.inSameSubnetwork(sender,receiver) and c < numRounds:
                        c +=1
                        receiver = random.choice(receivers_tmp)
                        sender = random.choice(senders_tmp)

                    #we did not find it
                    if c == numRounds:
                        if round == numRounds-1:
                            duration_tmp = current_times[0][0]+duration +3
                            del(current_times[0])
                            bisect.insort(current_times,(duration_tmp,'noAllocated'))
                            break
                        else:

                            sender = random.choice(senders)
                            receiver = random.choice(receivers)
                            while self.topology.inSameSubnetwork(sender,receiver):
                                receiver = random.choice(receivers)

                #we increase the current time even if the flow can not be allocated.
                #we check if the flow can be allocated or not.

                #if the flow will finish later than the simulation time is not scheduled.
                #or the sender can not fit this flow in its link
                #or the receiver can not fit this flow in its link

                starting_time = current_times[0][0] + 3

                if current_times[0][0]+3+duration > totalTime or hosts_capacities[sender]['out'] + size > self.linkBandwidth or hosts_capacities[receiver]['in'] + size > self.linkBandwidth:
                    #flow is not allocated


                    #update current_times
                    if round == numRounds-1:
                        duration_tmp = current_times[0][0]+duration +3
                        del(current_times[0])
                        bisect.insort(current_times,(duration_tmp,'noAllocated'))
                        # current_times[0] = (current_times[0][0]+duration+1,'noAllocated').
                        break

                    #remaining duration
                    if current_times[0][0]+duration > totalTime:
                        duration = totalTime - current_times[0][0]

                        if type == 'm':
                            #if the remainging time its smaller than the minimum of a mice we do not allocate
                            if duration < time_range_mice[0]:
                                # duration_tmp = current_times[0][0]+duration +1
                                # del(current_times[0])
                                # bisect.insort(current_times,(duration_tmp,'noAllocated'))

                                continue
                        if type =='e':
                            #same for elephants
                            if duration < time_range_elephant[0]:
                                # duration_tmp = current_times[0][0]+duration +1
                                # del(current_times[0])
                                # bisect.insort(current_times,(duration_tmp,'noAllocated'))
                                continue

                    if hosts_capacities[sender]['out'] + size > self.linkBandwidth or hosts_capacities[receiver]['in'] + size > self.linkBandwidth:

                        #lets find which one has the least space for flow allocation and adapt to that..
                        size = min(self.linkBandwidth - hosts_capacities[sender]['out'], self.linkBandwidth-hosts_capacities[receiver]['in'])

                        #it there is less than the 5 % of the link we do not allocate
                        if size < self.linkBandwidth*0.05:
                            # duration_tmp = current_times[0][0]+duration +1
                            # del(current_times[0])
                            # bisect.insort(current_times,(duration_tmp,'noAllocated'))
                            continue

                        #we reduce that a 10%
                        size = int(size*0.90)


                    #we allocate that modified flow
                    hosts_capacities[sender]['out'] += size
                    hosts_capacities[receiver]['in'] += size
                    flow = self.randomFlow(srcHost=sender, dstHost=receiver, size=size, startTime=starting_time, duration=duration,hosts_capacities=hosts_capacities)

                    hosts_capacities[sender]['usedPorts'].add(flow['sport'])
                    hosts_capacities[receiver]['usedPorts'].add(flow['dport'])

                    temporal_port_list.append((flow["sport"],flow["dport"]))

                    flows.append(flow)

                    #update current_times
                    duration_tmp = current_times[0][0]+duration +1
                    del(current_times[0])
                    bisect.insort(current_times,(duration_tmp,flow))

                    #if its long enough we consider it elephant
                    if flow['duration'] >= time_range_elephant[0]:
                        elephantTrafficCounter +=1

                    #Break the loop because a flow was found
                    break


                    #update current_times
                    #not allocated flows are not added to the flows to be scheduled list


                #allocate flow
                else:
                    #we add the size of the flow | so we are allocating it.
                    hosts_capacities[sender]['out'] += size
                    hosts_capacities[receiver]['in'] += size
                    flow = self.randomFlow(srcHost=sender,dstHost=receiver,size=size,startTime=starting_time,duration=duration,hosts_capacities=hosts_capacities)

                    hosts_capacities[sender]['usedPorts'].add(flow['sport'])
                    hosts_capacities[receiver]['usedPorts'].add(flow['dport'])
                    temporal_port_list.append((flow["sport"],flow["dport"]))
                    flows.append(flow)

                    #update current_times
                    duration_tmp = current_times[0][0]+duration +3
                    del(current_times[0])
                    bisect.insort(current_times,(duration_tmp,flow))
                    # current_times[0] = (current_times[0][0]+duration+1,flow)

                    #if its long enough we consider it elephant
                    if flow['duration'] >= time_range_elephant[0]:
                        elephantTrafficCounter +=1

                    #Break the loop because a flow was found
                    break

            #current_times.sort()


        return flows

    def schedule(self, traffic_per_host):
        self.ControllerClient.send(json.dumps({"type":"reset"}),"")

        # Send all flowlist to each host
        pass

    #@profile
    def schedule2(self,flows):

        #reset flows list
        # url = "http://127.0.0.1:5001/resetFlows"
        # try:
        #     requests.post(url,data = {})
        # except ConnectionError:
        #     pass
        #reset controller
        self.ControllerClient.send(json.dumps({"type":"reset"}),"")

        #schedule all the flows
        try:
            for flow in flows:
                #self.scheduler.enter(flow["start_time"],1, self.pushFlow, [flow])
                self.scheduler.enter(flow["start_time"],1, self.startFlowSingle, [flow])

            self.scheduler.run()
        except KeyboardInterrupt:
            #kill all the iperf3 servers and clients
            print "killing all"
            if self.iperf:

                subprocess.call("sudo killall -9 iperf3",shell=True)
            else:
                for host in self.topology.networkGraph.getHosts():

                    self.unixClient.send(json.dumps({"type":"terminate"}),host)

                #reset to the controller
                self.ControllerClient.send(json.dumps({"type":"reset"}),"")

                #subprocess.call("kill -9  $(ps aux | grep '/flowGenerator.py' | awk '{print $2}')",shell=True)

            #reset flows list
            url = "http://127.0.0.1:5001/resetFlows"
            try:
                requests.post(url,data = {})
            except ConnectionError:
                pass

    #@profile
    def startFlowSingle(self,flow):
        #sends command to the server
        clientName = self.topology.getHostName(flow["src"])
        self.unixClient.send(json.dumps({"type":"flow","data":flow.toDICT()}),clientName)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    #group = parser.add_mutually_exclusive_group()

    parser.add_argument('-t', '--time',
                           help='Duration of the traffic generator',
                           default=400)

    parser.add_argument('-n', '--nflows',
                           help="Number of flows kept constant",
                           default=20)

    parser.add_argument('-e', '--elephant',
                           help='Percentage of elephant flows',
                           default=0.1)

    parser.add_argument('-m', '--mice',
                           help='Percentage of mice flows',
                           default=0.9)


    parser.add_argument('--senders',
                           help='List of switch edges that can send traffic',
                           default="r_0_e0")


    parser.add_argument('--receivers',
                           help='List of switch edges that can receive traffic',
                           default="r_1_e0")


    parser.add_argument('--remote_hosts',
                        help='enables remote host traffic generation option',
                        action='store_true',
                        default=False)

    parser.add_argument('--iperf',
                        help='enables iperf3 traffic generator',
                        action='store_true',
                        default=False)

    parser.add_argument('--save_traffic',
                           help='saves traffic in a file so it can be repeated',
                           default="")

    parser.add_argument('--load_traffic',
                           help='load traffic from a file so it can be repeated',
                           default="")

    import ipdb; ipdb.set_trace()
    args = parser.parse_args()


    totalTime = int(args.time)
    nflows = int(args.nflows)

    pelephant = float(args.elephant)
    pmice = float(args.mice)

    senders = args.senders.split(",")
    receivers = args.receivers.split(",")


    a = TrafficGenerator(remoteHosts=args.remote_hosts,iperf=args.iperf)
    t = time.time()

    if args.load_traffic:
        traffic = pickle.load(open(args.load_traffic,"r"))
    else:
        traffic = a.trafficPlanner(numFlows=nflows, senders=senders, receivers=receivers, totalTime=totalTime, percentageElephants=pelephant, percentageMice=pmice)

    if args.save_traffic:
        with open(args.save_traffic,"w") as f:
            pickle.dump(traffic,f)

    a.schedule(traffic)
    print "elapsed time ", time.time()-t




