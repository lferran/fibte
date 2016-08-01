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

import inspect

from fibte import CFG, LINK_BANDWIDTH
tmp_files = CFG.get("DEFAULT","tmp_files")
db_topo = CFG.get("DEFAULT","db_topo")
controllerServer = CFG.get("DEFAULT","controller_UDS_name")

MIN_PORT = 1
MAX_PORT = 2**16 -1
RangePorts = xrange(MIN_PORT,MAX_PORT)

# def read_pid(n):
#     """
#     Extract a pid from a file
#     :param n: path to a file
#     :return: pid as a string
#     """
#     try:
#         with open(n, 'r') as f:
#             return str(f.read()).strip(' \n\t')
#     except:
#         return None

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

    def __init__(self,remoteHosts=False,iperf = False,*args,**kwargs):
        super(TrafficGenerator, self).__init__(*args,**kwargs)

        self.remoteHosts = remoteHosts

        self.topology = TopologyGraph(getIfindexes = False, openFlowInformation = False, db = os.path.join(tmp_files,db_topo))
        self.linkBandwitdh = LINK_BANDWIDTH

        # Used to communicate with flowServers at the hosts.
        # {0} is because it will be filled with whichever server we want to talk to!
        self.unixClient = UnixClient(tmp_files+"flowServer_{0}")

        # Used to communicate with LoadBalancer Controller
        self.ControllerClient = UnixClient(os.path.join(tmp_files, controllerServer))

    #@profile
    def randomFlow(self,srcHost,dstHost,size,startTime,duration, hosts_capacities,tos=0,proto="UDP"):

        #we use host_capacities to know which ports are available
        sport = random.choice(RangePorts)
        while sport in hosts_capacities[srcHost]['usedPorts']:
            sport = random.choice(RangePorts)

        dport = random.choice(RangePorts)
        while dport in hosts_capacities[dstHost]['usedPorts']:
            dport = random.choice(RangePorts)

        srcIp = self.topology.getHostIp(srcHost)
        dstIp = self.topology.getHostIp(dstHost)
        return Flow(src=srcIp,dst=dstIp,sport=sport,dport=dport,size=size,start_time=startTime,duration=duration,tos=tos,proto=proto)

    def _getSize(self, size_range):

        size = random.randint(*size_range)
        return size

    #THIS TWO FUNCTIONS ARE THE ONES THAT CHOOSE THE SIZE OF THE ELEPHANT AND MICE
    ###################################################################
    def getSizeMice(self, capacity_range=[0.002,0.01]):

        return self._getSize(self.miceSizeRange(capacity_range))


    def getSizeElephant(self,capacity_range = [1,1]):

        return self._getSize(self.elephantSizeRange(capacity_range))
    ######################################################################

    def _getTime(self,time_range):
        return random.randint(*time_range)

    def getTimeMice(self,time_range = [2,5]):
        return self._getTime(time_range)

    def getTimeElephant(self,time_range = [5000,5000]):
        return self._getTime(time_range)

    def elephantSizeRange(self,capacity_range = [0.2,0.8]):
        #range between 20% and 80% of the link capacity
        return [self.linkBandwitdh*capacity_range[0],self.linkBandwitdh*capacity_range[1]]


    def miceSizeRange(self,capacity_range = [0.002,0.01]):
        #range between 0.2% and 1% link capacity

        return [self.linkBandwitdh*capacity_range[0],self.linkBandwitdh*capacity_range[1]]


    @staticmethod
    def weighted_choice(weight_m,weight_e):

        choices = [("e",weight_e),('m',weight_m)]

        total = sum(w for c, w in choices)
        r = random.uniform(0, total)
        upto = 0
        for c, w in choices:
            if upto + w >= r:
                return c
            upto += w
        assert False, "Shouldn't get here"

    #@profile
    def trafficPlanner(self,numFlows=50,senders=["r_0_e0"],receivers=["r_1_e0"],percentageMice=0.9, percentageElephants=0.1,totalTime=500):


        #First we will compute the length and size ranges of mice and elephant flows. This two parameters will depend
        #on link bandwidth and total time.

        time_range_elephant = [20,500]
        time_range_mice = [2,5]


        #This function should think which flows should be scheduled during totalTime. It should create a list
        #of flows that more or less at all time the number of current "running" flows is equal or close to numFlows.

        #senders should be the list of "edge" or hosts that can generate traffic. And receivers the edge or list of host
        #that can receive traffic in our network.

        #Mice and elephatn parameters are the percentage of numFlows that are mice or elephant.

        #the function should take into account the links bandwitdhs and maximum number of paths between nodes (dont really know how to use that because there are collisions everywhere....)

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
                    if hosts_capacities[sender]["out"] + size <= self.linkBandwitdh:
                        senders_tmp.append(sender)
                if senders_tmp:
                    sender = random.choice(senders_tmp)
                else:
                    sender = []

                receivers_tmp = []
                for receiver in receivers:
                    if hosts_capacities[receiver]["in"] + size <= self.linkBandwitdh:
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
                if current_times[flow_n][0]+duration > totalTime or hosts_capacities[sender]['out'] + size > self.linkBandwitdh \
                        or hosts_capacities[receiver]['in'] + size > self.linkBandwitdh:


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
                    if hosts_capacities[sender]['out'] + size > self.linkBandwitdh or hosts_capacities[receiver]['in'] + size > self.linkBandwitdh:

                        #lets find which one has the least space for flow allocation and adapt to that..
                        size = min(self.linkBandwitdh - hosts_capacities[sender]['out'], self.linkBandwitdh-hosts_capacities[receiver]['in'])

                        #it there is less than the 5 % of the link we do not allocate
                        if size < self.linkBandwitdh*0.05:

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
                    if hosts_capacities[sender]["out"] + size <= self.linkBandwitdh:
                        senders_tmp.append(sender)
                if senders_tmp:
                    sender = random.choice(senders_tmp)
                else:
                    sender = []

                receivers_tmp = []
                for receiver in receivers:
                    if hosts_capacities[receiver]["in"] + size <= self.linkBandwitdh:
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

                if current_times[0][0]+3+duration > totalTime or hosts_capacities[sender]['out'] + size > self.linkBandwitdh or hosts_capacities[receiver]['in'] + size > self.linkBandwitdh:
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

                    if hosts_capacities[sender]['out'] + size > self.linkBandwitdh or hosts_capacities[receiver]['in'] + size > self.linkBandwitdh:

                        #lets find which one has the least space for flow allocation and adapt to that..
                        size = min(self.linkBandwitdh - hosts_capacities[sender]['out'], self.linkBandwitdh-hosts_capacities[receiver]['in'])

                        #it there is less than the 5 % of the link we do not allocate
                        if size < self.linkBandwitdh*0.05:
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

    #@profile
    def schedule(self,flows):

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

    #@profile
    def pushFlow(self,flow):

        #url of the controller listening
        url = "http://127.0.0.1:5001/startingFlow"
        try:
            requests.post(url,data = flow.toJSON())
        except ConnectionError:
            pass

        #if hosts are remote (not simulated in this computer)
        if self.remoteHosts:

            pass

        #if hosts are simulated here. Then, we can run mx commands
        else:
            self.mxStartFlow(flow)


    #@profile
    def mxStartFlow(self,flow):

        if self.iperf:
            #start server
            serverName = self.topology.getHostName(flow["dst"])

            dstPort = flow["dport"]

            #pidName = "/tmp/iperf_{0}_{1}.pid".format(serverName,dstPort)
            cmd = "mx {0} iperf3 -s -p {1} -I {2}  > /dev/null &".format(serverName,dstPort,pidName)
            #cmd = "mx {0} iperf3 -s -p {1} -1  --logfile {2}iperf_server_{0}_{1} &".format(serverName,dstPort,tmp_files)
            subprocess.call(cmd,shell=True)
            #print cmd

        self.scheduler.enter(1,1,self.mxStartFlow_client,[flow])

    #@profile
    def mxStartFlow_client(self,flow):
        #read pid and delete file

        clientName = self.topology.getHostName(flow["src"])

        if self.iperf:

            #start client
            cmd = "mx {0} iperf3 -c {1} --cport {2} --bind 0 -Z -u -b {3} -p {4} -t {5} > /dev/null &".format(clientName,flow["dst"],flow["sport"], flow.setSizeToStr(flow["size"]),flow["dport"],flow["duration"])
            #cmd = "mx {0} iperf3 -c {1} --cport {2} --bind 0 -Z -u -b {3} -p {4} -t {5}  --logfile {6}iperf_client_{0}_{2}_{4} &".format(clientName,flow["dst"],flow["sport"], flow.setSizeToStr(flow["size"]),flow["dport"],flow["duration"],tmp_files)
            subprocess.call(cmd,shell=True)
        else:

            self.unixClient.send(pickle.dumps({"type":"flow","data":{'dst':flow["dst"],'sport':flow['sport'],'size':flow['size'],'dport':flow['dport'],'duration':flow['duration']}}),clientName)

        #schedule stop function
        self.scheduler.enter(flow["duration"],1,self.mxStopFlow,[flow])

    #@profile
    def mxStopFlow(self,flow):

        url = "http://127.0.0.1:5001/stoppingFlow"
        try:
            requests.post(url,data= flow.toJSON())
        except ConnectionError:
            pass
        #kills server

        #pid = read_pid(pidName).strip('\x00')

        #del_file(pidName)

        #cmd = "kill -9 {0}".format(pid)
        #subprocess.call(cmd,shell=True)


    def restStartFlow(self,flow):
        pass


    def restStopFlow(self):
        pass

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
        traffic = a.trafficPlanner(numFlows=nflows,senders=senders,receivers=receivers,totalTime=totalTime,percentageElephants=pelephant,percentageMice=pmice)

    #print traffic

    if args.save_traffic:
        with open(args.save_traffic,"w") as f:
            pickle.dump(traffic,f)

    a.schedule(traffic)
    print "elapsed time ", time.time()-t




