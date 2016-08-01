#!/usr/bin/python

from fibbingnode.algorithms.southbound_interface import SouthboundManager
from fibbingnode import CFG
import fibte.res.config as cfg
from fibte.misc.unixSockets import UnixServer
import threading
import os
import argparse
import json

from fibte import tmp_files, db_topo, LINK_BANDWIDTH

# Threading event to signal that the initial topo graph
# has been received from the Fibbing controller
HAS_INITIAL_GRAPH = threading.Event()

UDS_server_name = CFG.get("DEFAULT","controller_UDS_name")

class MyGraphProvider(SouthboundManager):
    """This class overrwides the received_initial_graph abstract method of
    the SouthboundManager class. It is used to receive the initial
    graph from the Fibbing controller.
    The HAS_INITIAL_GRAPH is set when the method is called.
    """
    def __init__(self):
        super(MyGraphProvider, self).__init__()

    def received_initial_graph(self):
        super(MyGraphProvider, self).received_initial_graph()
        HAS_INITIAL_GRAPH.set()

class LBController(object):
    def __init__(self, doBalance = True):

        self.doBalance = doBalance

        # Unix domain server to make things faster and possibility to communicate with hosts
        self.server = UnixServer(os.path.join(tmp_files, UDS_server_name))

        # Connects to the southbound controller. Must be called before
        # creating the instance of SouthboundManager
        CFG.read(cfg.C1_cfg)

        # Start the Southbound manager in a different thread
        self.sbmanager = MyGraphProvider()
        t = threading.Thread(target=self.sbmanager.run, name="Southbound Manager")
        t.start()

        # Blocks until initial graph received from SouthBound Manager
        HAS_INITIAL_GRAPH.wait()

        # Receive network graph
        self.networw_graph = self.sbmanager.igp_graph

    def reset(self):
        pass

    def handleFlow(self):
        pass

    def run(self):
        # Receive events and handle them
        while True:
            try:
                if not(self.doBalance):
                    while True:
                        event = self.server.receive()
                        print json.loads(event)
                else:
                    event = json.loads(self.server.receive())
                    if event["type"] == "reset":
                        self.reset()
                        continue
                    else:
                        self.handleFlow(event)
                        
            except KeyboardInterrupt:
                break

if __name__ == '__main__':
    #from lb.logger import log
    #import logging

    parser = argparse.ArgumentParser()
    #group = parser.add_mutually_exclusive_group()

    parser.add_argument('--doBalance',
                        help='If set to False, ignores all events and just prints them',
                        action='store_true',
                        default = True)
    args = parser.parse_args()

    #log.setLevel(logging.DEBUG)
    #log.info("Starting Controller")

    lb = LBController(doBalance = args.doBalance)
    lb.run()

    #import ipdb; ipdb.set_trace()