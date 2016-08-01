#!/usr/bin/env python

import flowGenerator
from fibte.misc.unixSockets import UnixServer
import multiprocessing
import json
import signal
import sched
import time
import sys
import logging
# from lb.logger import log
from threading import Thread
import Queue


import select
def my_sleep(seconds):
    select.select([], [], [], seconds)

class Joiner(Thread):
    def __init__(self, q):
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

        # log.addHandler(handler)
        mlog.addHandler(handler)
        mlog.setLevel(logging.NOTSET)

        # log.setLevel(logging.NOTSET)

        self.address = "/tmp/flowServer_{0}".format(name)

        # Instantiate UDP Unix Socket Server
        self.server = UnixServer(self.address)
        self.processes = []

        signal.signal(signal.SIGTERM, self.signal_term_handler)
        # signal.signal(signal.SIGCHLD,signal.SIG_IGN)

        self.parentPid = os.getpid()
        # log.debug("First time p:{0},{1}".format(os.getppid(), os.getpid()))

        # Start process Joiner
        self.queue = Queue.Queue(maxsize=0)
        joiner = Joiner(self.queue)
        joiner.setDaemon(True)
        joiner.start()

        # Flow generation start time
        self.startime = 0

        # List of flows to start with their respective times []-> (t, flow)
        self.flowlist = []

        # Indicators
        self.received_starttime = False
        self.received_flowlist = False

        # Schedules own flows
        self.scheduler = sched.scheduler(time.time, my_sleep)

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
        process = multiprocessing.Process(target = flowGenerator.sendFlowNotifyController, kwargs = (flow))
        process.daemon = True
        process.start()
        self.processes.append(process)
        self.queue.put(process)

    def run(self):
        while not(self.received_flowlist) and not(self.received_starttime):
            # Receive event from Socket server and convert it to a dict (--blocking)
            event = json.loads(self.server.receive())
            # log.debug("server: {0}, event: {1}".format(self.address,str(event)))
            if event["type"] == "terminate":
                self.terminateALL()
                break

            elif event["type"] == 'startime':
                self.setStartTime(event["data"])
                self.received_starttime = True

            elif event["type"] == "flowlist":
                self.flowlist = event["data"]
                self.received_flowlist = True

        if self.received_flowlist and self.received_starttime:
            try:
                for (flowtime, flow) in self.flowlist:
                    delta = self.starttime - time.time()
                    self.scheduler.enter(delta+flow["start_time"], 1, self.startFlow, [flow])

                self.scheduler.run()

            except KeyboardInterrupt:
                self.terminateALL()
                sys.exit(0)

if __name__ == "__main__":
    import os
    # Name of the flowServer is passed when called
    name = sys.argv[1]

    # Store the pid of the process so we can stop it when we stop the network
    with open("/tmp/flowServer_{0}.pid".format(name),"w") as f:
        f.write(str(os.getpid()))

    FlowServer(name).run()
