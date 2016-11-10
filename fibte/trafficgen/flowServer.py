#!/usr/bin/env python

import multiprocessing
from multiprocessing import Process
import json
import signal
import sched
import time
import sys
import traceback
import logging
import Queue
import numpy as np
import subprocess
from threading import Thread, Lock

import flowGenerator
from fibte.monitoring.traceroute import traceroute, traceroute_fast
from fibte.misc.unixSockets import UnixServerTCP, UnixClientTCP, UnixServer, UnixClient
from fibte.trafficgen import isElephant, isMice
from fibte.misc.ipalias import get_secondary_ip
from fibte.logger import log
from fibte.trafficgen.flow import Base
from fibte.misc.topology_graph import NamesToIps
from fibte import tmp_files, db_topo


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

class Joiner(Thread):
    def __init__(self, q):
        super(Joiner,self).__init__()
        self.__q = q
    def run(self):
        while True:
            child = self.__q.get()
            if child == None:
                return
            child.join()

class FlowServer(object):
    def __init__(self, name, ip_alias=False, sampling_rate=1, notify_period=10):
        # Setup flowServer name
        self.name = name

        # Get parent pid
        self.parentPid = os.getpid()

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

        # Flow generation start time
        self.starttime = 0

        # Create the scheduler instance and run it in a separate process
        self.scheduler = sched.scheduler(time.time, my_sleep)
        self.scheduler_thread = Thread(target=self.scheduler.run)
        self.scheduler_thread.setDaemon(True)

        # Weather secondary ip's are present at the host or not
        self.ip_alias = ip_alias

        # To store mice estimation
        self.estimation_lock = Lock()
        self.samples_lock = Lock()
        self.mice_estimation = {}
        self.mice_estimation_samples = {}

        # In samples/s
        self.sampling_rate = sampling_rate
        self.notify_period = notify_period

        # UnixUDPServer for mice start/stop notifications
        self.own_mice_server_name = "/tmp/miceServer_{0}"
        self.own_mice_server = UnixServer(self.own_mice_server_name.format(self.name))

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

        # Start count of wrong notifications
        self.controllerNotFoundCount = 0

        # start joiner
        self.queue = Queue.Queue(maxsize=0)
        joiner = Joiner(self.queue)
        joiner.setDaemon(True)
        joiner.start()

        # Configure sigterm handler
        signal.signal(signal.SIGTERM, self.signal_term_handler)

        self.setup_logging()

    def miceCounterServer(self):
        """Thread that keeps track of the mice level observed by the host"""
        while True:
            # Wait until mice start/stop events arrive
            data = json.loads(self.own_mice_server.receive())

            if data['type'] == 'mice_stop':
                mice_flow = data['flow']
                # Decrease estimation
                self.decreaseMiceLoad(mice_flow)

            elif data['type'] == 'reset':
                with self.estimation_lock:
                    self.mice_estimation.clear()
            else:
                continue

    def setup_logging(self):
        """"""
        mlog = multiprocessing.get_logger()
        handler = logging.FileHandler(filename='/tmp/flowServer_{0}.log'.format(self.name))
        fmt = logging.Formatter('[%(levelname)20s] %(asctime)s %(funcName)s: %(message)s ')
        handler.setFormatter(fmt)
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        mlog.addHandler(handler)
        mlog.setLevel(logging.DEBUG)

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

        # Log a bit
        #print "Server {1}: time doing {2}-traceroute: {0} (trials: {3})".format(time.time() - now, self.name, traceroute_type, n_trials)

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

    def setStartTime(self, starttime):
        if time.time() < starttime:
            self.starttime = starttime
        else:
            raise ValueError("starttime is older than the current time!")

    def startFlow(self, flow):
        """
        Start flow
        :param flow:
        """
        if isElephant(flow):
            # Start flow notifying controller
            process = Process(target=flowGenerator.sendElephantFlow, kwargs=flow)
            process.daemon = True
            process.start()

            # Append it to processes
            self.processes.append(process)

            # Log a bit
            size = self.base.setSizeToStr(flow.get('size'))
            rate = self.base.setSizeToStr(flow.get('rate')) if flow.get('rate', None) else size
            duration = self._getFlowDuration(flow)
            dst = self.namesToIps['ipToName'][flow['dst']]
            proto = flow['proto']
            log.debug("{0} : {4} ELEPHANT flow is STARTING: to {1} rate of {2} during {3}".format(self.name, dst, rate, duration, proto))

        # if is Elephant
        else:
            # Create client to send mice notification
            mice_client = UnixClient(self.own_mice_server_name)

            # Start flow without notifying controller
            process = Process(target=flowGenerator.sendMiceFlow, args=(mice_client, self.name), kwargs=flow)
            process.daemon = True
            process.start()

            # Append it to processes
            self.processes.append(process)

            # Add size of flow to
            self.increaseMiceLoad(flow)

    def remoteStartReceiveTCP(self, dst, dport):
        """"""
        if 'h_' not in dst:
            # Convert dst ip to dname
            dname = self.namesToIps['ipToName'][dst]

        # Only for debugging
        else:
            dname = dst

        # Start netcat process that listens on port
        process = subprocess.Popen(["mx", dname, "nc", "-l", "-p", str(dport)], stdout=open(os.devnull, "w"))
        self.popens.append(process)

    def threadedStartReceiveTCP(self, dport):
        """"""
        # Start netcat process that listens on port
        th = Thread(target=self._localStartReceiveTCP, args=[dport])
        th.start()

    def _localStartReceiveTCP(self, dport):
        log.debug("{0} : Started TCP server at port {1}".format(self.name, dport))

        # Start netcat process that listens on port
        process = subprocess.Popen(["nc", "-l", "-p", str(dport)], stdout=open(os.devnull, "w"), stderr=subprocess.PIPE)
        out, err = process.communicate()
        if err:
            log.error(err)

    def localStartReceiveTCP(self, dport):
        """"""
        # Start netcat process that listens on port
        process = subprocess.Popen(["nc", "-l", "-p", str(dport)], stdout=open(os.devnull, "w"))
        self.popens.append(process)
        log.info("{0} : Started TCP server at port {1}".format(self.name, dport))

    def increaseMiceLoad(self, flow):
        """"""
        if flow['proto'] == 'UDP':
            to_increase = flow['size']

        else:
            to_increase = flow['rate']

        # Take flow destination
        dst = flow['dst']

        with self.estimation_lock:
            if dst in self.mice_estimation.keys():
                self.mice_estimation[dst] += to_increase
            else:
                self.mice_estimation[dst] = to_increase

        #log.info("(+) mice estimation")

    def decreaseMiceLoad(self, flow):
        """"""
        if flow['proto'] == 'UDP':
            to_decrease = flow['size']

        else:
            to_decrease = flow['rate']

        # Take flow destination
        dst = flow['dst']
        with self.estimation_lock:
            if dst in self.mice_estimation.keys():
                self.mice_estimation[dst] -= to_decrease
            else:
                log.error("Decreasing flow that wasn't increased previously")

        #log.info("(-) mice estimation")

    def takeMiceLoadSample(self):
        """Start process that takes sample"""
        try:
            with self.samples_lock:
                with self.estimation_lock:
                    for (dst, load) in self.mice_estimation.iteritems():
                        if dst in self.mice_estimation_samples.keys():
                            self.mice_estimation_samples[dst].append(load)
                        else:
                            self.mice_estimation_samples[dst] = [load]
            #log.info("[+] sample taken!")
        except Exception as e:
            log.error("Error while taking mice load sample")
            log.exception(e)

    def _getFlowEndTime(self, flow):
        """"""
        return flow.get('startTime') + self._getFlowDuration(flow)

    def _getFlowDuration(self, flow):
        if flow['proto'] == 'UDP':
            return flow.get('duration')
        else:
            return flow.get('size')/flow.get('rate')

    def getTrafficDuration(self, flowlist):
        """
        Iterates the flowlist and returns the ending time of the last ending flow
        """
        if flowlist:
            return max([self._getFlowEndTime(flow) for flow in flowlist])
        else:
            log.error("NO FLOWLIST FOR THIS HOST! ")
            return 3600

    def scheduleNotifyMiceLoads(self):
        """
        Schedules the notifications of the mice loads to the controller
        """
        if self.controllerNotFoundCount > 40:
            log.warning("{0} : Not scheduling mice notifications anymore".format(self.name))
            pass
        else:
            looptime = 100

            # Log a bit
            log.info("{0} : Scheduling mice notificationss: every {1}s".format(self.name, self.notify_period))

            # Schedule samplings at correct sampling intervals
            for st in range(0, looptime, self.notify_period):
                self.scheduler.enter(st, 1, self.notifyMiceLoads, [])

            # Schedule himself too!
            self.scheduler.enter(looptime-1, 1, self.scheduleNotifyMiceLoads, [])

    def scheduleMiceSamplings(self):
        """
        Schedules the samplings for the mice estimation
        """
        if self.controllerNotFoundCount > 40:
            log.warning("{0} : Not scheduling mice samplings anymore".format(self.name))
            pass

        else:
            looptime = 100

            sampling_period = 1 / self.sampling_rate
            log.info("{0} : Scheduling mice samplings: every {1}s".format(self.name, 1 / self.sampling_rate))

            # Schedule samplings at correct sampling intervals
            for sp in range(0, looptime, sampling_period):
                self.scheduler.enter(sp, 1, self.takeMiceLoadSample, [])

            # Schedule himself too!
            self.scheduler.enter(looptime-1, 1, self.scheduleMiceSamplings, [])

    def notifyMiceLoads(self):
        """Send samples from last period to controller"""
        try:
            with self.samples_lock:
                # Copy dict
                samples_to_send = self.mice_estimation_samples.copy()

                # Empty previous samples
                self.mice_estimation_samples.clear()

            # Send them to the controller
            self.client_to_controller.send(json.dumps({"type": "miceEstimation", 'data': {'src': self.name, 'samples': samples_to_send}}), "")
        except Exception as e:
            self.controllerNotFoundCount += 1
            log.error("{0} : Controller not found: {1}".format(self.name, e))

    def terminateTraffic(self):
        # No need to terminate startFlow processes, since they are killed when
        # the scheduler process is terminated!

        # Cancel all upcoming scheduler events
        action = [self.scheduler.cancel(e) for e in self.scheduler.queue]

        # Restart thread
        self._restartSchedulerThread()

        # Terminate netcat processes
        #log.info("{0} : Killing all ongoing processes".format(self.name))
        for popen in self.popens:
            popen.kill()

        for process in self.processes:
            if process.is_alive():
                try:
                    time.sleep(0.001)
                    process.terminate()
                    process.join()
                except OSError:
                    pass

        # Restart lists
        self.processes = []
        self.popens = []

        # Empty the mice estimation count
        client = UnixClient(self.own_mice_server_name)
        client.send(json.dumps({'type':'reset'}), self.name)

        self.mice_estimation_samples = {}

        # Schedule mice loads again
        if self.ip_alias:
            self.scheduleNotifyMiceLoads()
            self.scheduleMiceSamplings()

        self.controllerNotFoundCount = 0

    def waitForTrafficToFinish(self):
        """
        Scheuler waits for flows to finish and then terminates the run of
        the traffic
        :return:
        """
        log.info("Waiting a bit for last flows to finish...")
        time.sleep(2)
        log.info("Re-setting flowServer!")
        self.terminateTraffic()

    def _startTrafficGeneratorListenerThread(self):
        # Start thread that reads from the server TCP socket
        tcp_server_thread = Thread(target = self.server_from_tg.run)
        tcp_server_thread.setDaemon(True)
        tcp_server_thread.start()

    def _startMiceCounterThread(self):
        # Start thread that counts the mice sizes
        mice_thread = Thread(target=self.miceCounterServer)
        #mice_thread.setDaemon(True)
        mice_thread.start()

    def scheduleFlowList(self, flowlist):
        """Schedules a list of flows to start"""

        # Keep track of elephant|mice count
        flow_count = {'elephant': 0, 'mice': 0}

        # Iterate flowlist
        for flow in flowlist:

            # Rewrite destination address when needed
            if isMice(flow) and self.ip_alias == True:
                flow['dst'] = get_secondary_ip(flow['dst'])

            # Get start time with lower bound
            starttime = max(0, flow.get('start_time'))

            # Schedule startFlow
            self.scheduler.enter(starttime, 1, self.startFlow, [flow])

            # Add counts
            if isElephant(flow):
                flow_count['elephant'] += 1
            else:
                flow_count['mice'] += 1

        # Log a bit
        log.debug("{0} : All flows were scheduled!".format(self.name))
        log.debug("{0} : {1} Mice | {2} Elephant".format(self.name, flow_count['mice'], flow_count['elephant']))

        # Start schedule.run()
        self.startSchedulerThread()

    def scheduleReceiveList(self, receivelist):
        """Schedules a list of flow receivers to start"""
        # Time in advance for which we start the receiver before the sender starts
        BUFFER_TIME = 3

        # Iterate flowlist
        for (flowtime, dport) in receivelist:
            receivertime = max(0, flowtime - BUFFER_TIME)

            # Schedule the start of the TCP server too
            self.scheduler.enter(receivertime, 1, self.localStartReceiveTCP, [dport])
            #self.scheduler.enter(receivertime, 1, self.threadedStartReceiveTCP, [dport])

        log.debug("{0} : All TCP flow servers were scheduled!".format(self.name))
        log.debug("{0} : Will receive a total of {1} TCP flows".format(self.name, len(receivelist)))

        # Start schedule.run()
        self.startSchedulerThread()

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
        log.info("{0} : flowServer started!".format(self.name))

        # Start some threads
        self._startTrafficGeneratorListenerThread()
        self._startMiceCounterThread()

        if self.ip_alias:
            # Schedule notify mice loads
            self.scheduleNotifyMiceLoads()
            self.scheduleMiceSamplings()

        # Loop forever
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
                        self.scheduleFlowList(flowlist=flowlist)

                elif event['type'] == "receivelist":
                    receivelist = event['data']
                    if receivelist:
                        # Schedule times at which we need to start TCP servers
                        log.debug("{0} : Receive_list arrived".format(self.name))
                        self.scheduleReceiveList(receivelist)

                elif event["type"] == "terminate":
                    # Stop traffic during ongoing traffic
                    log.debug("{0} : Terminate event received!".format(self.name))
                    self.terminateTraffic()

                else:
                    log.debug("{0} : Unknown event received {1}".format(self.name, event))
                    continue

            except KeyboardInterrupt:
                log.info("{0} : KeyboardInterrupt catched! Exiting".format(self.name))
                self.terminateTraffic()
                sys.exit(0)

if __name__ == "__main__":
    import os

    # Name of the flowServer is passed when called
    if len(sys.argv) == 3 and sys.argv[2] == '--ip_alias':
        name = sys.argv[1]
        ip_alias = True

    elif len(sys.argv) == 2:
        name = sys.argv[1]
        ip_alias = False

    else:
        raise Exception("Wrong number of arguments")

    # Store the pid of the process so we can stop it when we stop the network
    with open("/tmp/flowServer_{0}.pid".format(name),"w") as f:
        f.write(str(os.getpid()))

    FlowServer(name, ip_alias=ip_alias).run()
