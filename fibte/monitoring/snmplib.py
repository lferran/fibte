from pysnmp.hlapi import *
# import numpy as np
import time
import subprocess
from pysnmp.entity.rfc3413.oneliner import cmdgen
import math

from fibte import LINK_BANDWIDTH

max_capacity = LINK_BANDWIDTH / 8

ifDescr = "1.3.6.1.2.1.2.2.1.2"
outs = "1.3.6.1.2.1.2.2.1.16"


class SnmpIfDescr(object):
    def __init__(self, routerIp="127.0.0.1", port=161):

        self.routerIp = routerIp
        self.port = port

        self.ifindexMapping = {}

        # get the ifindexes from the router
        # self.getIfIndex()
        self.getIfIndexWalk()

    def getIfIndexWalk(self):

        # alternative to getIfindex using snmpwalk intead of python library that makes
        # everything quite slow
        values = subprocess.check_output("snmpwalk -v 2c -c public {0} {1}".format(self.routerIp, ifDescr), shell=True)
        values = values.strip().split("\n")
        for value in values:
            index, name = value.split("=")
            index = index.strip().split('.')[-1]
            name = name.strip().split(':')[-1].strip()
            if name == "lo" or ("mon" in name):
                continue
            self.ifindexMapping[name] = index

    def getIfIndex(self):
        cmdGen = cmdgen.CommandGenerator()
        errorIndication, errorStatus, errorIndex, varBindTable = cmdGen.nextCmd(
            cmdgen.CommunityData('public', mpModel=1),
            cmdgen.UdpTransportTarget((self.routerIp, 161)),
            ifDescr,
            lookupNames=True, lookupValues=True
            # get all oids
            # lexicographicMode=True, maxRows=10,
            # ignoreNonIncreasingOid=True
        )

        if errorIndication:
            print(errorIndication)
        else:
            if errorStatus:
                print('%s at %s' % (
                    errorStatus.prettyPrint(),
                    errorIndex and varBindTable[-1][int(errorIndex) - 1] or '?'
                )
                      )
            else:
                for varBindTableRow in varBindTable:
                    for name, val in varBindTableRow:
                        # do not consider loopback interface
                        if str(val) == "lo" or ("mon" in str(val)):
                            continue
                        self.ifindexMapping[str(val)] = str(name).split('.')[-1]

    def getIfMapping(self):
        return self.ifindexMapping


class SnmpCounters(object):
    def __init__(self, interfaces=["2", "3", "4", "5"], routerIp="1.0.9.2", port=161):

        self.routerIp = routerIp
        self.port = port
        self.interfaces = interfaces
        # special attributre used to filter interfaces when bulking counters
        # self.interfacesInt = [int(x)-1 for x in self.interfaces]
        self.countersTimeStamp = time.time()

        self.counters = {x: 0 for x in interfaces}
        self.totalBytes = 0

    def getCounters32(self):
        """
        Interfaces is a list of the interfaces counters we want to get
        """

        ifHCOutOctets = "1.3.6.1.2.1.2.2.1.16.%s"
        myoids = [ObjectType(ObjectIdentity(ifHCOutOctets % x)) for x in self.interfaces]
        errorIndication, errorStatus, errorIndex, varBinds = next(
            getCmd(SnmpEngine(),
                   CommunityData('public', mpModel=1),
                   UdpTransportTarget((self.routerIp, self.port)),
                   ContextData(),
                   *myoids)
        )

        if errorIndication:
            print(errorIndication)
        elif errorStatus:
            print('%s at %s' % (
                errorStatus.prettyPrint(),
                errorIndex and varBinds[int(errorIndex) - 1][0] or '?'
            )
                  )
        else:
            countersTimeStamp = time.time()
            self.a = varBinds
            counters = {str(oid.getOid().asTuple()[-1]): float(counter) for oid, counter in varBinds if
                        str(oid.getOid().asTuple()[-1]) in self.interfaces}

            self.timeDiff = time.time() - self.countersTimeStamp
            self.countersDiff = {k: counters[k] - self.counters[k] for k in counters if k in self.counters}

            self.totalBytes = sum(self.countersDiff.values())

            self.countersTimeStamp = countersTimeStamp
            self.counters = counters

    def getCounters64(self):
        """
        Interfaces is a list of the interfaces counters we want to get
        """

        ifHCOutOctets = "1.3.6.1.2.1.31.1.1.1.10.%s"
        myoids = [ObjectType(ObjectIdentity(ifHCOutOctets % x)) for x in self.interfaces]

        errorIndication, errorStatus, errorIndex, varBinds = next(
            getCmd(SnmpEngine(),
                   CommunityData('public', mpModel=1),
                   UdpTransportTarget((self.routerIp, self.port)),
                   ContextData(),
                   *myoids)
        )

        if errorIndication:
            print(errorIndication)
        elif errorStatus:
            print('%s at %s' % (
                errorStatus.prettyPrint(),
                errorIndex and varBinds[int(errorIndex) - 1][0] or '?'
            )
                  )
        else:
            countersTimeStamp = time.time()
            counters = {str(oid.getOid().asTuple()[-1]): float(counter) for oid, counter in varBinds if
                        str(oid.getOid().asTuple()[-1]) in self.interfaces}

            self.timeDiff = time.time() - self.countersTimeStamp
            self.countersDiff = {k: counters[k] - self.counters[k] for k in counters if k in self.counters}

            self.totalBytes = sum(self.countersDiff.values())

            self.countersTimeStamp = countersTimeStamp
            self.counters = counters

    # @profile
    def getCounters64Walk(self, out=True):
        if out:
            oid = "1.3.6.1.2.1.31.1.1.1.10"
        else:
            oid = "1.3.6.1.2.1.31.1.1.1.6"
        values = subprocess.check_output("snmpwalk -v 2c -c public {0} {1}".format(self.routerIp, oid), shell=True)

        counters = {x.split(" = ")[0].strip(".")[-1]: float(x.split("Counter64:")[-1].strip()) for x in
                    values.strip().split("\n")}
        # filter only the interfaces we want
        counters = {x: y for x, y in counters.items() if x in self.interfaces}

        countersTimeStamp = time.time()

        self.timeDiff = time.time() - self.countersTimeStamp

        self.countersDiff = {k: counters[k] - self.counters[k] for k in counters if k in self.counters}

        self.totalBytes = sum(self.countersDiff.values())

        self.countersTimeStamp = countersTimeStamp
        self.counters = counters

    def link_capacity(self):

        # if we are sampling with time differences smaller than 1 second, that means that we are
        # still have to divide by 1 since counters can not be updated faster than every second
        if self.timeDiff < 1:
            self.timeDiff = 1.0

        self.timeDiff = math.floor(self.timeDiff)

        return {x: self.countersDiff[x] / (max_capacity * self.timeDiff) for x in self.countersDiff}

    def link_traffic(self):

        if self.totalBytes == 0:
            link_traffic = {x: 0 for x in self.interfaces}
        else:
            link_traffic = {x: self.countersDiff[x] / self.totalBytes for x in self.countersDiff}

        return link_traffic

    def fromLastLecture(self):
        # time from last lecture
        return time.time() - self.countersTimeStamp

