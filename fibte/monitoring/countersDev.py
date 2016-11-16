import time
import json

from fibte import CFG, LINK_BANDWIDTH

link_capacity = LINK_BANDWIDTH/8
net_dev_path = CFG.get("DEFAULT", "net_dev_path")
interface_index_path = CFG.get("DEFAULT","interface_index_path")

"""
This module defines two objects: IfDescr and CountersDev, that allow us
to read the router interface counters and router interface names from
the dev system file directly, without having to use the SNMP protocol.

credit to: Edgar Costa - edgarcosta@hotmail.es
"""


class IfDescr(object):
    def __init__(self):
        # Holds the mappings between interface names and interface indexes
        self.ifindexMapping = {}

        # Makes the mapping
        self.loadMapping()

    def getInterfaceNames(self):
        """
        Reads the /proc/net/dev file to fetch the list of interfaces

        :return: list of interface names
        """
        with  open(net_dev_path,"r") as f:
            ifnames = [x for x in [x.split()[0][:-1] for x in f.readlines()][2:]
            if (x != "lo" and ("mon" not in x))]
        return ifnames

    def getIfIndex(self,ifname):
        """
        Given an interface name, reads its interface index
        from /sys/class/net/ifname/ifindex and returns it.

        :param ifname: interface name
        :return: interface index
        """
        with open(interface_index_path.format(ifname),"r") as f:

            ifindex = int(f.read())

        return ifindex

    def loadMapping(self):
        """
        Iterates the interface names and collects their indexes. Results
        are stored in self.ifIndexMapping dict: {}: iface_name -> iface_index
        """
        for interface in self.getInterfaceNames():
            self.ifindexMapping[interface] = self.getIfIndex(interface)

    def getIfMapping(self):
        """
        Return the mappings dictionary
        """
        return self.ifindexMapping

    def saveIfMapping(self,path):
        """
        Saves the mappings in a file specified by path
        """
        with open(path,"w") as f:
            json.dump(self.ifindexMapping, f)


class CountersDev(object):
    def __init__(self,interfaces = [], isEdge = True, isAggr = True):

        # Interfaces for which we want to keep track of the counters
        self.interfaces = interfaces

        # Keeps track of time
        self.countersTimeStamp = time.time()

        # Marks if interface is connecting the host to the edge routers
        self.isEdge = isEdge
        self.isAggr = isAggr

        if self.isEdge or self.isAggr:
            self.counters = {"in": {x: 0 for x in interfaces}, "out": {x: 0 for x in interfaces}}
            self.totalBytes = {"out": 0, "in": 0}
        else:
            self.counters = {"out": {x: 0 for x in interfaces}}
            self.totalBytes = {"out": 0}

        # File that is updated by the linux kernel. We get the RX and TX counters from it.
        self.countersFile = open(net_dev_path,"r")

    def readFile(self):
        self.countersFile.seek(0)
        return [x.split() for x in self.countersFile.readlines()][2:]

    def getCounters(self):

        # Read the counter file first (from the OS)
        values = self.readFile()

        # Update timestamps
        now = time.time()
        self.timeDiff = now - self.countersTimeStamp
        self.countersTimeStamp = now

        # Parse counters
        if self.isEdge or self.isAggr:
            counters = {"in": {}, "out": {}}
            for interface in values:
                if interface[0][:-1] in self.interfaces:
                    counters["in"][interface[0][:-1]] = float(interface[1])
                    counters["out"][interface[0][:-1]] = float(interface[9])

            self.countersDiff = {"in": {x: counters["in"][x] - self.counters["in"][x] for x in counters["in"] if x in self.counters["in"]},
                                 "out": {x: counters["out"][x] - self.counters["out"][x] for x in counters["out"] if x in self.counters["out"]}}
            self.totalBytes["in"] = sum(self.countersDiff["in"].values())
            self.totalBytes["out"] = sum(self.countersDiff["out"].values())

        else:
            counters = {"out": {}}
            for interface in values:
                if interface[0][:-1] in self.interfaces:
                    counters["out"][interface[0][:-1]] = float(interface[9])

            self.countersDiff = {"out": {x: counters["out"][x] - self.counters["out"][x] for x in counters["out"] if x in self.counters["out"]}}
            self.totalBytes["out"] = sum(self.countersDiff["out"].values())

        # Update counters
        self.counters = counters

    def link_capacity(self):
        if self.isEdge or self.isAggr:
            return {"in": {x: self.countersDiff["in"][x]/(link_capacity*self.timeDiff) for x in self.countersDiff["in"]},
                    "out": {x:self.countersDiff["out"][x]/(link_capacity*self.timeDiff) for x in self.countersDiff["out"]}}
        else:
            return {"out": {x:self.countersDiff["out"][x]/(link_capacity*self.timeDiff) for x in self.countersDiff["out"]}}

    def link_traffic(self):
        # This is harder to implement have to think how.
        pass

    def fromLastLecture(self):
        return time.time() - self.countersTimeStamp
