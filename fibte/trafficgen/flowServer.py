#!/usr/bin/env python

import flowGenerator
from fibte.misc.unixSockets import UnixServerTCP
from multiprocessing import Process
import json
import signal
import sched
import time
import sys
import logging
from fibte.logger import log
from threading import Thread
import Queue


import select
def my_sleep(seconds):
    select.select([], [], [], seconds)

class Joiner(Thread):
    def __init__(self, q, scheduler_pid=-1):
        super(Joiner,self).__init__()
        # Queue of pids of processes
        self.__q = q

    def run(self):
        while True:
            child = self.__q.get()
            if child == None:
                return
            # Wait for child to finish
            child.join()


class FlowServer(object):

    def __init__(self, name):

        mlog = multiprocessing.get_logger()

        handler = logging.FileHandler(filename='/tmp/flowServer_{0}.log'.format(name))
        fmt = logging.Formatter('[%(levelname)20s] %(asctime)s %(funcName)s: %(message)s ')

        handler.setFormatter(fmt)

        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        mlog.addHandler(handler)
        mlog.setLevel(logging.NOTSET)

        self.address = "/tmp/flowServer_{0}".format(name)

        # Instantiate TCP Unix Socket Server - commands from TG are received here
        self.q_server = Queue.Queue(0)
        self.server = UnixServerTCP(self.address, self.q_server)

        self.processes = []

        signal.signal(signal.SIGTERM, self.signal_term_handler)
        # signal.signal(signal.SIGCHLD,signal.SIG_IGN)

        self.parentPid = os.getpid()
        log.debug("First time p:{0},{1}".format(os.getppid(), os.getpid()))

        # Start process Joiner
        self.queue = Queue.Queue(maxsize=0)
        self.joiner = Joiner(self.queue)
        self.joiner.setDaemon(True)
        self.joiner.start()

        # Flow generation start time
        self.startime = 0

        # List of flows to start with their respective times []-> (t, flow)
        self.flowlist = []

        # Indicators
        self.received_starttime = False
        self.received_flowlist = False

        # Schedules own flows
        self.scheduler = sched.scheduler(time.time, my_sleep)

        # Start the Scheduler Thread
        self.scheduler_process = Thread(target=self.scheduler.run)
        self.scheduler_process.daemon = True


    def signal_term_handler(self,signal,frame):
        # Only parent will do this
        if os.getpid() == self.parentPid:
            #self.queue.put(None)
            self.server.close()
            sys.exit(0)
        else:
            sys.exit(0)

    #TODO this should be a thread. When reading self.processes we should use a lock.
    def terminateALL(self):
        #log.debug(str(len(self.processes)))
        # for process in self.processes

        for process in self.processes:
            if process.is_alive():
                try:
                    process.terminate()
                    process.join()
                except OSError:
                    pass

        self.processes = []

    def setStartTime(self, starttime):
        if time.time() < starttime:
            self.starttime = starttime
        else:
            raise ValueError("starttime is older than the current time!")

    def startFlow(self, flow):
        """
        Start flow calling the flow generation function
        :param flow:
        """
        process = Process(target = flowGenerator.sendFlowNotifyController, kwargs = (flow))
        process.daemon = True
        process.start()
        self.processes.append(process)
        self.queue.put(process)
        log.debug("New flow started: {0}".format(str(flow)))

    def terminateTraffic(self):
        # Terminate all flow start processes
        self.terminateALL()

        # Cancel all upcoming scheduler events
        action = [self.scheduler.cancel(e) for e in self.scheduler.queue]

        # Terminate old scheduler process
        self.scheduler_process.terminate()

        # Create a new instance of the process
        self.scheduler_process = Process(target=self.scheduler.run)

        # Reset everything
        self.received_flowlist = False
        self.received_starttime = False
        self.starttime = 0
        self.flowlist = []

        # Flush the receiving queue
        with self.q_server.mutex:
            self.q_server.queue.clear()

    def run(self):
        # Start thread that reads from the server TCP socket
        tcp_server_thread = Thread(target = self.server.run)
        tcp_server_thread.setDaemon(True)
        tcp_server_thread.start()

        while True:
            # No simulation ongoing -- waiting for events
            if not self.scheduler_process.is_alive():

                # Wait for initial data to arrive from traffic generator
                while not(self.received_flowlist) or not(self.received_starttime):

                    # Receive event from Socket server and convert it to a dict (--blocking)
                    event = json.loads(self.q_server.get())
                    self.q_server.task_done()

                    # Log a bit
                    log.debug("server: {0}, event: {1}".format(self.address, str(event['type'])))

                    if event["type"] == "starttime":
                        self.setStartTime(event["data"])
                        self.received_starttime = True
                        log.debug("Event starttime arrived")

                    elif event["type"] == "flowlist":
                        self.flowlist = event["data"]
                        self.received_flowlist = True
                        log.debug("Event flowlist arrived")

                log.debug("flowlist and starttime events received")
                if self.received_flowlist and self.received_starttime:
                    for flow in self.flowlist:
                        delta = self.starttime - time.time()
                        if delta < 0:
                            log.error("We neet do wait a bit more in the TrafficGenerator!! Delta is negative!")

                        # Schedule the flow start
                        self.scheduler.enterabs(self.starttime + flow["start_time"], 1, self.startFlow, [flow])

                    # Run scheduler in another thread
                    self.scheduler_process.start()


            # Simulation ongoing -- only terminate event allowed
            else:
                # While traffic stil ongoing
                while self.scheduler_process.is_alive():

                    # Check if new event in the queue
                    try:
                        data = self.q_server.get(timeout=3)
                    except Queue.Empty:
                        self.q_server.task_done()
                    else:
                        event = json.loads(data)
                        self.q_server.task_done()

                        if event["type"] == "terminate":
                            # Stop traffic during ongoing traffic
                            self.terminateTraffic()

                # Stop traffic
                self.terminateTraffic()

if __name__ == "__main__":
    import os
    # Name of the flowServer is passed when called
    name = sys.argv[1]

    # Store the pid of the process so we can stop it when we stop the network
    with open("/tmp/flowServer_{0}.pid".format(name),"w") as f:
        f.write(str(os.getpid()))

    FlowServer(name).run()
