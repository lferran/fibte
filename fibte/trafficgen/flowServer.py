#!/usr/bin/env python

import flowGenerator
from fibte.misc.unixSockets import UnixServerTCP
import multiprocessing
from multiprocessing import Process
import json
import signal
import sched
import time
import sys
import traceback
import logging
from fibte.logger import log
from threading import Thread
import Queue
import numpy as np
from threading import Lock

from fibte.trafficgen import isElephant
from fibte.misc.ipalias import get_secondary_ip
from fibte.misc.unixSockets import UnixClientTCP

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

    def __init__(self, name, ip_alias=False, sampling_rate=1, notify_period=10):
        # Setup flowServer name
        self.name = name

        # Setup logging stuff
        self.setup_logging()

        # Own's socket address
        self.address = "/tmp/flowServer_{0}".format(name)

        # Instantiate TCP Unix Socket Server - commands from TG are received here
        self.q_server = Queue.Queue(0)
        self.server_from_tg = UnixServerTCP(self.address, self.q_server)

        # Configure sigterm handler
        signal.signal(signal.SIGTERM, self.signal_term_handler)

        # Get own pid
        self.parentPid = os.getpid()
        log.debug("First time p:{0},{1}".format(os.getppid(), os.getpid()))

        # Flow generation start time
        self.starttime = 0
        self.received_starttime = False

        # List of flows to start with their respective times []-> (t, flow)
        self.flowlist = []
        self.received_flowlist = False

        # Create the scheduler instance and run it in a separate process
        self.scheduler = sched.scheduler(time.time, my_sleep)
        self.scheduler_process = Process(target=self.scheduler.run)
        self.scheduler_process.daemon = False

        # Weather secondary ip's are present at the host or not
        self.ip_alias = ip_alias
        log.debug("Is ip alias for elephants is active? {0}".format(str(self.ip_alias)))

        # To store mice estimation
        self.estimation_lock = Lock()
        self.samples_lock = Lock()
        self.mice_estimation = {}
        self.mice_estimation_samples = {}

        # In samples/s
        self.sampling_rate = sampling_rate
        self.notify_period = notify_period

        # Unix Client to communicate to controller
        self.client_to_controller = UnixClientTCP("/tmp/controllerServer")

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
            #self.queue.put(None)
            self.server_from_tg.close()
            sys.exit(0)
        else:
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
            process = Process(target=flowGenerator.sendFlowNotifyController, kwargs=flow)
            process.daemon = True
            process.start()

        # if is Elephant
        else:
            # Start flow without notifying controller
            process = Process(target=flowGenerator.sendFlowNotNotify, kwargs=flow)
            process.daemon = True
            process.start()

            # Add size of flow to
            self.increaseMiceLoad(flow)

    def stopFlow(self, flow):
        """
        Stop flow
        """
        if isElephant(flow):
            process = Process(target=flowGenerator.stopFlowNotifyController, kwargs=flow)
            process.daemon = True
            process.start()

        # if is mice
        else:
            # Remove size of flow
            self.decreaseMiceLoad(flow)

    def increaseMiceLoad(self, flow):
        # Take flow size
        size = flow['size']

        # Take flow destination
        dst = flow['dst']

        with self.estimation_lock:
            if dst in self.mice_estimation.keys():
                self.mice_estimation[dst] += size
            else:
                self.mice_estimation[dst] = size

    def decreaseMiceLoad(self, flow):
        # Take flow size
        size = flow['size']

        # Take flow destination
        dst = flow['dst']

        with self.estimation_lock:
            if dst in self.mice_estimation.keys():
                self.mice_estimation[dst] -= size
            else:
                log.error("Decreasing flow that wasn't increased previously")

    def _takeMiceLoadSample(self, mice_estimation_samples, estimation_lock, samples_lock):
        """Traverse all other host loads and save samples"""
        with estimation_lock:
            for (dst, load) in self.mice_estimation.iteritems():
                if dst in self.mice_estimation_samples.keys():
                    mice_estimation_samples[dst].append(load)
                else:
                    mice_estimation_samples[dst] = [load]

    @time_func
    def takeMiceLoadSample(self):
        """Start process that takes sample"""
        with self.samples_lock:
            with self.estimation_lock:
                for (dst, load) in self.mice_estimation.iteritems():
                    if dst in self.mice_estimation_samples.keys():
                        self.mice_estimation_samples[dst].append(load)
                    else:
                        self.mice_estimation_samples[dst] = [load]
        #process = Process(target=self._takeMiceLoadSample, args=(mice_estimation_samples, self.estimation_lock, self.samples_lock))
        #process.daemon = True
        #process.start()

    def scheduleSamplings(self):
        """
        Schedules the samplings for the mice estimation
        """
        sampling_period = 1/self.sampling_rate
        log.info("Scheduling mice samplings: every {0}s".format(1/self.sampling_rate))

        # Compute total traffic duration
        endtime = self.starttime + self.getTrafficDuration(self.flowlist)

        # Schedule samplings at correct sampling intervals
        for st in np.arange(self.starttime + sampling_period, endtime, sampling_period):
            self.scheduler.enterabs(float(st), 1, self.takeMiceLoadSample, [])

        log.info("Samplings scheduled!")

    def getTrafficDuration(self, flowlist):
        """
        Iterates the flowlist and returns the ending time of the last ending flow
        """
        end_times = [flow['start_time'] + flow['duration'] for flow in flowlist]
        max_end_time = max(end_times)
        return max_end_time

    def scheduleNotifyMiceLoads(self):
        """
        Schedules the notifications of the mice loads to the controller
        """
        # Compute total traffic duration
        endtime = self.starttime + self.getTrafficDuration(self.flowlist)

        # Schedule samplings at correct sampling intervals
        for st in np.arange(self.starttime + self.notify_period, endtime, self.notify_period):
            self.scheduler.enterabs(float(st), 1, self.notifyMiceLoads, [])

    @time_func
    def notifyMiceLoads(self):
        """Send samples from last period to controller"""
        with self.samples_lock:
            # Copy dict
            samples_to_send = self.mice_estimation_samples.copy()

            # Empty previous samples
            self.mice_estimation_samples.clear()

        # Send them to the controller
        self.client_to_controller.send(json.dumps({"type": "miceEstimation", 'data': {'src': self.name, 'samples': samples_to_send}}), "")

    def terminateTraffic(self):
        # No need to terminate startFlow processes, since they are killed when
        # the scheduler process is terminated!

        # Cancel all upcoming scheduler events
        action = [self.scheduler.cancel(e) for e in self.scheduler.queue]

        # Terminate old scheduler process if alive
        if self.scheduler_process.is_alive(): self.scheduler_process.terminate()

        # Create a new instance of the process
        self.scheduler_process = Process(target=self.scheduler.run)

        # Reset everything
        self.received_flowlist = False
        self.received_starttime = False
        self.starttime = 0
        self.flowlist = []
        self.mice_estimation = {}
        self.mice_estimation_samples = {}

    def waitForTrafficToFinish(self):
        """
        Scheduler waits for flows to finish and then terminates the run of
        the traffic
        :return:
        """
        log.info("Waiting a bit for last flows to finish...")
        time.sleep(2)
        log.info("Re-setting flowServer!")
        self.terminateTraffic()

    def run(self):
        # Start thread that reads from the server TCP socket
        tcp_server_thread = Thread(target = self.server_from_tg.run)
        tcp_server_thread.setDaemon(True)
        tcp_server_thread.start()

        while True:
            # No simulation ongoing -- waiting for events
            if not self.scheduler_process.is_alive():

                # Wait for initial data to arrive from traffic generator
                while not(self.received_flowlist) or not(self.received_starttime):
                    log.debug('Waiting for flowlist and starttime events...')

                    # Receive event from Socket server and convert it to a dict (--blocking)
                    event = json.loads(self.q_server.get())
                    self.q_server.task_done()

                    if event["type"] == "starttime":
                        self.setStartTime(event["data"])
                        self.received_starttime = True
                        log.debug("Event starttime arrived")

                    elif event["type"] == "flowlist":
                        self.flowlist = event["data"]
                        self.received_flowlist = True
                        log.debug("Event flowlist arrived")

                log.debug("Flowlist and starttime events received")
                log.debug("Scheduling flows... ")
                log.debug("DELTA time observed: {0}".format(self.starttime - time.time()))

                # Initialize counters
                flow_count = {'elephant': 0, 'mice': 0}
                if self.received_flowlist and self.received_starttime:
                    # Schedule mice sampling and notifications
                    self.scheduleSamplings()
                    self.scheduleNotifyMiceLoads()

                    # Iterate flowlist
                    for flow in self.flowlist:
                        delta = self.starttime - time.time()
                        if delta < 0:
                            log.error("We neet do wait a bit more in the TrafficGenerator!! Delta is negative!")

                        # Rewrite destination address when needed
                        if not isElephant(flow) and self.ip_alias == True:
                            flow['dst'] = get_secondary_ip(flow['dst'])

                        # Schedule the flow start
                        self.scheduler.enterabs(self.starttime + flow["start_time"], 1, self.startFlow, [flow])

                        # Compute flow's ending time
                        ending_time = self.starttime + flow["start_time"] + flow["duration"]

                        if isElephant(flow):
                            flow_count['elephant'] += 1
                            log.debug("ELEPHANT flow to {0} with {1} (bps) will start in {2} and last for {3}".format(flow['dst'], flow['size'], flow['start_time'], flow["duration"]))

                        else:
                            flow_count['mice'] += 1

                        # Schedule stopFlow function
                        self.scheduler.enterabs(ending_time, 1, self.stopFlow, [flow])

                    log.debug("All flows were scheduled! Let's run the scheduler (in a different thread)")
                    log.debug("A total of {0} flows will be started at host. {1} MICE | {2} ELEPHANT".format(sum(flow_count.values()), flow_count['mice'], flow_count['elephant']))

                    # Run scheduler in another thread
                    self.scheduler_process.start()

            # Simulation ongoing -- only terminate event allowed
            else:
                try:
                    # While traffic stil ongoing
                    while self.scheduler_process.is_alive():
                        #log.debug("Scheduler process is still alive -- Traffic ongoing")
                        #log.info("Processes: {0}".format(len(self.processes)))

                        # Check if new event in the queue
                        try:
                            data = self.q_server.get(timeout=3)#self.notify_period)#block=False)
                        except Queue.Empty:
                            # Send current mice loads to the controller
                            #self.notifyMiceLoads()
                            pass
                        else:
                            #log.debug("Timeout didn't occur: loading json object...")
                            self.q_server.task_done()
                            event = json.loads(data)

                            if event["type"] == "terminate":
                                # Stop traffic during ongoing traffic
                                log.debug("Terminate event received from trafficGenerator - terminating...")
                                self.terminateTraffic()

                    # Stop traffic immediately
                    self.waitForTrafficToFinish()
                except Exception as e:
                    log.error("EXCEPTION OCCURRED: {0}".format(traceback.print_exc()))

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
