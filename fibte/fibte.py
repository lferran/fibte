from fibbingnode.algorithms.southbound_interface import SouthboundManager
from fibte import CFG

import threading

# Threading event to signal that the initial topo graph
# has been received from the Fibbing controller
HAS_INITIAL_GRAPH = threading.Event()

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
    def __init__(self):
        # Connects to the southbound controller. Must be called before
        # creating the instance of SouthboundManager
        CFG.read(dconf.C1_Cfg)

        # Start the Southbound manager in a different thread
        self.sbmanager = MyGraphProvider()
        t = threading.Thread(target=self.sbmanager.run, name="Southbound Manager")
        t.start()

        # Blocks until initial graph received from SouthBound Manager
        HAS_INITIAL_GRAPH.wait()

        # Receive network graph
        self.networw_graph = self.sbmanager.igp_graph


    def run(self):
        pass

if __name__ == '__main__':
    lb = LBController()
    lb.run()