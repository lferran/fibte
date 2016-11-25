#!/usr/bin/env python

import multiprocessing
from multiprocessing import Process
import json
import signal
import sched
import time
import sys
import random
import logging
import Queue
import subprocess
from threading import Thread, Lock, Event

import flowGenerator
from fibte.monitoring.traceroute import traceroute, traceroute_fast
from fibte.misc.unixSockets import UnixServerTCP, UnixClientTCP, UnixServer, UnixClient
from fibte.trafficgen import isElephant, isMice
from fibte.misc.ipalias import get_secondary_ip
from fibte.logger import log
from fibte.trafficgen.flow import Base
from fibte.misc.topology_graph import NamesToIps
from fibte import tmp_files, db_topo
from fibte import LINK_BANDWIDTH

def time_func(function):
    def wrapper(*args,**kwargs):
        t = time.time()
        res = function(*args,**kwargs)
        log.debug("{0} took {1}s to execute".format(function.func_name, time.time()-t))
        return res
    return wrapper

import select

def my_sleep(seconds):
    select.select([], [], [], seconds)

class FlowServer(object):
    def __init__(self, name, own_ip, ip_alias=False, sampling_rate=1, notify_period=10):
        # Setup flowServer name and ip
        self.name = name
        self.primary_ip = own_ip

        # Weather secondary ip's are present at the host or not
        self.ip_alias = ip_alias

        # Generate secondary ip if needed
        if self.ip_alias:
            self.secondary_ip = get_secondary_ip(self.primary_ip)
        else:
            self.secondary_ip = None

        # Get parent pid
        self.parentPid = os.getpid()

        # Setup logs
        self.setup_logging()

        # Configure sigterm handler
        signal.signal(signal.SIGTERM, self.signal_term_handler)

        # Setup black hole
        self.setupUDPBlackHole()

        # Reset ICMP ratelimit
        self.setICMPRateLimit()

        # Accumulate processes for TCP servers
        self.popens = []
        self.processes = []

        # Own's socket address
        self.address = "/tmp/flowServer_{0}".format(name)

        # Instantiate TCP Unix Socket Server - commands from TG are received here
        self.q_server = Queue.Queue(0)
        self.server_from_tg = UnixServerTCP(self.address, self.q_server)

        # Unix Client to communicate to controller
        self.client_to_controller = UnixClientTCP("/tmp/controllerServer")

        # Create the scheduler instance and run it in a separate process
        self.scheduler = sched.scheduler(time.time, my_sleep)
        self.scheduler_thread = Thread(target=self.scheduler.run)
        self.scheduler_thread.setDaemon(True)

        # Traceroute sockets
        self.traceroute_server_name = "/tmp/tracerouteServer_{0}".format(name)
        self.traceroute_server = UnixServer("/tmp/tracerouteServer_{0}".format(name))
        self.own_pod_hosts = []

        # Client that sends to the server -- we must have one like this in the controller
        self.traceroute_client = UnixClient(self.traceroute_server_name)

        # Start process that listens from server
        process = multiprocessing.Process(target=self.tracerouteServer)
        # process.daemon = True
        process.start()

        # Utilities
        self.base = Base()
        self.namesToIps = NamesToIps(os.path.join(tmp_files, db_topo))

        # Mice generation variables
        # Open mice connections data
        self.open_mice_connections = []
        # Keep sending mices
        self.keep_mices = False
        # We store the current mice thread here
        self.mices_thread = None

    def setup_logging(self):
        """"""
        mlog = multiprocessing.get_logger()
        self.logfile = '/tmp/flowServer_{0}.log'.format(self.name)
        handler = logging.FileHandler(filename=self.logfile)
        fmt = logging.Formatter('[%(levelname)20s] %(asctime)s %(funcName)s: %(message)s ')
        handler.setFormatter(fmt)
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        mlog.addHandler(handler)
        mlog.setLevel(logging.DEBUG)

    def setupUDPBlackHole(self):
        """Starts blackhole at the mininet host"""
        # Check first if rule already exists
        s = subprocess.Popen(["iptables-save"], stdout=subprocess.PIPE)
        out, err = s.communicate()
        out = out.split('\n')
        #import ipdb; ipdb.set_trace()
        already_there = any([True for line in out if 'udp' in line and 'DROP' in line])
        if not already_there:
            # It means it doesn't exist, so we have to set it!
            log.info("Setting up UDP black hole!")
            #cmd = "iptables -A INPUT -s {0} -p udp -j DROP".format(self.ip)
            # Drop all UDP traffic
            cmd = "iptables -A INPUT -p udp -j DROP"
            subprocess.call(cmd, shell=True)
        else:
            log.info("UDP black hole already up!")

    def setICMPRateLimit(self):
        # Modify icmp ratelimit too
        subprocess.call('sysctl -w net.ipv4.icmp_ratelimit=0', shell=True)

    def signal_term_handler(self, signal, frame):
        # Only parent will do this
        if os.getpid() == self.parentPid:
            log.info("{0}: Parent exiting".format(self.name))
            self.traceroute_server.close()
            self.server_from_tg.close()
            for process in self.processes:
                log.info("{0} : Terminating pid: {1}".format(self.name, process))
                process.terminate()
            sys.exit(0)
        else:
            log.info("{0} : Children exiting!".format(self.name))
            sys.exit(0)

    def tracerouteThread(self, client, **flow):
        """Starts a traceroute procedure"""
        now = time.time()

        # Result is stored here
        route_info = {'flow': flow, 'route': []}

        # If flowServer hasn't received initial data
        if not self.own_pod_hosts:
            # Compute slow version of traceroute
            hops = 6

            # Use traceroute (original)
            traceroute_fun = traceroute

        else:
            if flow['dst'] in self.own_pod_hosts:
                hops = 2
            else:
                hops = 3

            # use traceroute (fast version)
            traceroute_fun = traceroute_fast

        # Run function
        route = traceroute_fun(hops=hops, **flow)

        try_again = 10
        while not route and try_again > 0:
            # Try it again
            route = traceroute_fun(hops=hops, **flow)
            try_again -= 1

        # How many trials needed?
        n_trials = 10 - try_again

        # Add found route
        route_info['route'] = route

        # Send the result back
        client.send(json.dumps(route_info), "")

    def tracerouteServer(self):
        """
        This function is ran in a separate thread:
        it receives orders from the controller to know
        the route taken by certain flows
        """
        # Results are sent to the controller
        client = UnixClient("/tmp/controllerServer_traceroute")

        try:
            while True:
                # Wait for traceroute commands
                command = json.loads(self.traceroute_server.receive())

                if not self.own_pod_hosts and command['type'] == 'own_pod_hosts':
                    # Update own pod hosts list
                    own_pods_hosts = command['data']
                    self.own_pod_hosts = own_pods_hosts
                    log.info("{0}: list of own pod hosts received: {1}".format(self.name, own_pods_hosts))
                    continue

                elif command['type'] == 'flow':
                    # Extract flow
                    flow = command['data']

                    # Start traceroute in a dedicated thread!
                    thread = multiprocessing.Process(target=self.tracerouteThread, args=(client,), kwargs=(flow))
                    thread.daemon = True
                    # thread.setDaemon(True)
                    thread.start()
                else:
                    continue
        except KeyboardInterrupt:
            log.info("{0} : KeyboardInterrupt catched! Exiting".format(self.name))
            sys.exit(0)

    def startElephantFlow(self, flow, logDelay=False):
        """
        Start elephant flow
        """
        if isElephant(flow):
            # Start flow notifying controller
            process = Process(target=flowGenerator.sendElephantFlow, args=(logDelay,), kwargs=flow)
            process.daemon = True
            process.start()

            # Append it to processes
            self.processes.append(process)

            # Log a bit
            size = self.base.setSizeToStr(flow.get('size'))
            rate = self.base.setSizeToStr(flow.get('rate'))
            duration = self._getFlowDuration(flow)
            dst = self.namesToIps['ipToName'][flow['dst']]
            proto = flow['proto']
            log.debug("{0} : Starting {4} elephant -> {1} rate: {2} duration: {3}".format(self.name, dst, rate, duration, proto))
        else:
            log.error("{0} : not an elephant flow! {1}".format(self.name, flow))

    def localStartReceiveTCP(self, dport):
        """"""
        # Start netcat process that listens on port
        process = subprocess.Popen(["nc", "-k", "-l", "-p", str(dport)], stdout=open(os.devnull, 'w'), close_fds=True)
        self.popens.append(process)

    def _getFlowEndTime(self, flow):
        """"""
        return flow.get('startTime') + self._getFlowDuration(flow)

    def _getFlowDuration(self, flow):
        if flow['proto'].lower() == 'udp':
            return flow.get('duration')
        else:
            return flow.get('size')/float(flow.get('rate'))

    def getTrafficDuration(self, flowlist):
        """
        Iterates the flowlist and returns the ending time of the last ending flow
        """
        if flowlist:
            return max([self._getFlowEndTime(flow) for flow in flowlist])
        else:
            log.error("NO FLOWLIST FOR THIS HOST! ")
            return 3600

    def terminateTraffic(self):
        # No need to terminate startFlow processes, since they are killed when
        # the scheduler process is terminated!
        # Cancel all upcoming scheduler events
        action = [self.scheduler.cancel(e) for e in self.scheduler.queue]

        # Terminate all mice processes/threads
        self.keep_mices = False
        for conndata in self.open_mice_connections:
            thread = conndata.get('thread')
            queue = conndata.get('queue')
            sending = conndata.get('sending')
            # Stop flow if it is sending
            if sending.isSet(): sending.clear()
            # Terminate sendeing thread
            queue.put('terminate')
            # Join the thread
            if thread: thread.join()

        self.open_mice_connections = []
        if self.mices_thread:
            try:
                self.mices_thread.join()
            except Exception as e:
                log.exception(e)

        # Terminate all tcpSenders
        for process in self.processes:
            if process.is_alive():
                try:
                    time.sleep(0.001)
                    process.terminate()
                    process.join()
                except OSError:
                    pass

        for popen in self.popens:
            popen.kill()
            popen.wait()

        # Restart thread
        self._restartSchedulerThread()

        # Restart lists
        self.processes = []
        self.popens = []

    def _startTrafficGeneratorListenerThread(self):
        """Start thread that reads from the server TCP socket"""
        tcp_server_thread = Thread(target = self.server_from_tg.run)
        tcp_server_thread.setDaemon(True)
        tcp_server_thread.start()

    def scheduleElephants(self, flowlist, logDelay=False):
        """Schedules a list of elephant flows to start"""
        udp = 0
        tcp = 0
        # Iterate flowlist
        for flow in flowlist:
            # Get start time with lower bound
            starttime = max(0, flow.get('start_time'))
            # Keep track of udp/tcp count
            if flow['proto'].lower() == 'udp':
                udp += 1
                logDelay = False

            else:
                tcp += 1

            # Schedule startFlow
            self.scheduler.enter(starttime, 1, self.startElephantFlow, [flow, logDelay])
        # Log a bit
        log.debug("{0} : Scheduled {1} elephant flows: [tcp: {2} | udp: {3}]".format(self.name, len(flowlist), tcp, udp))
        # Start schedule.run()
        self.startSchedulerThread()

    def scheduleTCPServers(self, receivelist, buffer_time=2):
        """Schedules the start of the TCP servers
        a few seconds before the sender starts
        """
        # Iterate flowlist
        for (flowtime, dport) in receivelist:
            self.scheduler.enter(flowtime - buffer_time, 1, self.localStartReceiveTCP, [dport])
        log.debug("{0} : scheduled {1} TCP servers".format(self.name, len(receivelist)))
        # Start schedule.run()
        self.startSchedulerThread()

    def startTCPServers(self, receivelist):
        """
        Given a list of ports to which mice flows will
        be received, start the TCP servers processes to listen
        to these ports
        """
        # Iterate flowlist
        for item in receivelist:
            if isinstance(item, tuple) and len(item) == 2:
                (flowtime, dport) = item
            else:
                dport = item
            self.localStartReceiveTCP(dport)
        log.debug("{0} : started {1} TCP servers".format(self.name, len(receivelist)))

    @staticmethod
    def getMiceSize():
        min_len_mice = 0.2
        max_len_mice = 6

        # Get mice size flow
        duration_mean = 3.0
        mice_duration = random.expovariate(1 / duration_mean)
        while mice_duration > max_len_mice or mice_duration < min_len_mice:
            mice_duration = random.expovariate(1 / duration_mean)

        mice_size = mice_duration * LINK_BANDWIDTH
        return mice_size

    def establishOutgoingMiceTCPConns(self, toSend, logDelay=False):
        """"""
        total_send_conns = len(toSend)
        erroneous_conns = 0

        for flowdata in toSend:
            sport = flowdata['sport']
            dport = flowdata['dport']

            # Rewrite destination if needed
            if self.ip_alias:
                dst = get_secondary_ip(flowdata['dst'])
                src = self.secondary_ip
            else:
                dst = flowdata['dst']
                src = self.primary_ip

            flow = {'src': src, 'sport': sport, 'dst': dst, 'dport': dport}

            # Setup connection
            socket = flowGenerator.setupTCPConnection(**flow)

            if socket: # Connection was setup correctly
                # Create queue
                queue = Queue.Queue(0)
                # Create Event
                sending = Event()

                if logDelay:
                    # Get completiontime file pattern
                    completionTimeFile = "{0}_{1}_{2}_{3}".format(self.secondary_ip, sport, dst, dport) + "_{0}"
                else:
                    completionTimeFile = None

                # Start thread that handles the open connection
                thread = Thread(target=flowGenerator.sendMiceThroughOpenSocket,
                                args=(socket, queue, sending, completionTimeFile))
                thread.start()

                # Save connection data
                connection_data = {'queue': queue, 'socket': socket, 'sending': sending,
                                   'filename': completionTimeFile, 'thread': thread}
                self.open_mice_connections.append(connection_data)

            else: # There was an error setting up the connection
                log.error("{2} : Connection at {0}:{1} could not be established".format(dst, dport, self.name))
                erroneous_conns += 1
                continue

        log.debug("{0} : {1}/{2} connections established".format(self.name, total_send_conns - erroneous_conns,
                                                                 total_send_conns))

    def startMiceSenderThread(self, toSend, average, logDelay=False):
        """This function generates the required TCP outgoing connections and then loops forever
        generating mice traffic with a certain flow arrival rate"""

        # Sleep a bit initially, to leave time to he TCP receivers to setup
        time.sleep(3)

        # Setup outoing connections first
        self.establishOutgoingMiceTCPConns(toSend, logDelay)

        if not self.open_mice_connections:
            log.error("{0} : There are no established mice connections. Returning...".format(self.name))
            return

        self.keep_mices = True
        while self.keep_mices:
            # Start flows following at Poisson arrival times
            next_flow = random.expovariate(1/float(average))

            # Sleep until then
            time.sleep(next_flow)

            # Get one of the sockets at random
            unactive_connections = [conn for conn in self.open_mice_connections if not conn['sending'].isSet()]
            if unactive_connections and self.keep_mices:
                # Pick one at random
                conn = random.choice(unactive_connections)
                # Put new mice size into the queue
                queue = conn['queue']
                queue.put(self.getMiceSize())

            elif self.keep_mices:
                log.warning("{0} : All connections are active already!".format(self.name))
                continue

        log.info("{0} : Finishing mice threads".format(self.name))

    def startSchedulerThread(self):
        """Starts the action of schedulerThread making sure that the queued
        events will be executed!"""
        if not self.scheduler.queue:
            log.warning("No events in the queue: doing nothing...")
            return

        if self.scheduler_thread.is_alive():
            #log.info("Scheduler thread is already alive!: doing nothing...")
            return

        try:
            self.scheduler_thread.start()
        except:
            #log.warning("Scheduler thread finished: restarting it!")
            self._restartSchedulerThread()
            self.scheduler_thread.start()
            return
        else:
            #log.info("Scheduler thread successfully started!")
            pass

    def _restartSchedulerThread(self):
        """Joins the previous scheduler thread and creates a new instance
        so that self.scheduler.run can be called again!
        """
        try:
            #log.info("Joining previous thread...")
            st = time.time()
            self.scheduler_thread.join()
            log.info("It took {0}s".format(time.time() - st))
        except RuntimeError:
            #log.warning("Thread wasn't previously started -> we can't join it")
            pass
        finally:
            self.scheduler_thread = Thread(target=self.scheduler.run, name="Scheduler Thread")
            self.scheduler_thread.setDaemon(True)

    def run(self):
        # Choose whether to log completion times for mices and elephants
        LOG_ELEPHANTS_COMPLETION_TIME = False
        LOG_MICE_COMPLETION_TIME = False

        log.info("{0} : flowServer started!".format(self.name))

        # Start thread that listends for commands and puts them in the queue
        self._startTrafficGeneratorListenerThread()

        while True:
            try:
                # Read from TG queue
                event = json.loads(self.q_server.get())
                self.q_server.task_done()

                if event["type"] == "flowlist":
                    flowlist = event["data"]
                    if flowlist:
                        # Schedule flows relative to current time
                        log.debug("{0} : Flow_list arrived".format(self.name))
                        self.scheduleElephants(flowlist=flowlist, logDelay=LOG_ELEPHANTS_COMPLETION_TIME)

                elif event['type'] == "receivelist":
                    receivelist = event['data']
                    if receivelist:
                        # Schedule times at which we need to start TCP servers
                        log.debug("{0} : Receive_list arrived".format(self.name))
                        self.scheduleTCPServers(receivelist, buffer_time=2)

                elif event['type'] == 'startReceiver':
                    port = event['data']
                    self.localStartReceiveTCP(dport=port)

                elif event["type"] == "terminate":
                    # Stop traffic during ongoing traffic
                    log.debug("{0} : Terminate event received!".format(self.name))
                    self.terminateTraffic()

                elif event["type"] == "mice_bijections":
                    # Parse data from event
                    average = event['data']['average']
                    toReceive = event['data']['bijections']['toReceive']
                    toSend = event['data']['bijections']['toSend']

                    # Start receiver threads
                    self.startTCPServers(toReceive)

                    # Start sender threads
                    self.mices_thread = Thread(target=self.startMiceSenderThread, args=(toSend, average))
                    self.mices_thread.start()

                else:
                    log.debug("{0} : Unknown event received {1}".format(self.name, event))
                    continue

            except KeyboardInterrupt:
                log.info("{0} : KeyboardInterrupt catched! Exiting".format(self.name))
                self.terminateTraffic()
                break

        log.info("{0} : Shutting down".format(self.name))
        sys.exit(0)

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', help='flowServer name', type=str, required=True)
    parser.add_argument('--own_ip', help='flowServer primary ip', type=str, required=True)
    parser.add_argument('--ip_alias', help='Are aliases active for mice flows?', action='store_true', default=False)

    args = parser.parse_args()

    # Store the pid of the process so we can stop it when we stop the network
    with open("/tmp/flowServer_{0}.pid".format(args.name), "w") as f:
        f.write(str(os.getpid()))

    # Start flowServer
    FlowServer(args.name, args.own_ip, ip_alias=args.ip_alias).run()