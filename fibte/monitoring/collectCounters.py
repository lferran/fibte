#!/usr/bin/python

import subprocess
import time
import json
import os
from fibte.misc.topologyGraph import TopologyGraph

from countersDev import IfDescr, CountersDev

from fibte import CFG

tmp_files = CFG.get("DEFAULT","tmp_files")
db_topo = CFG.get("DEFAULT","db_topo")

"""
This module defines a process that collects the counter information of a single host/router
in an isolated process. It either gathers snmp data or directly reads from the dev file.

credit: Edgar Costa - edgarcosta@hotmail.es
"""

def collectCounters(name='h_0_0', interval=1.5):
    try:
        #topology = TopologyGraph(getIfindexes=False,db=os.path.join(tmp_files,db_topo))

        #THIS WAS DONE USING THE TOPOLOGY TO KNOW IF THE ROUTER IS EDGE, HOWEVER AT TIME OF CALLING THIS FUNCTION
        #THE TOPOLOGY IS NOT READY SO WE CAN NOT RELAY ON THIS OBJECT, UNLESS WE CALL THIS SCRIPT LATER ON THE MAIN
        #SCRIPT. SINCE NOW IS NOT THAT IMPORTANT I WILL CHANGE THAT AND JUST CHECK IF THE ROUTER NAME CONTAINS AN E
        #SINCE THIS IS ONLY USED FOR SNMP COLLECTION MAYBE IS NOT EVEN NECESSARY FOR THIS THESIS..

        #isEdge = topology.networkGraph.graph.node[name].has_key("edge")

        # Checks if its a edge router
        isEdge = "e" in name

        # We use /proc/net/dev
        counters = CountersDev(interfaces=IfDescr().getIfMapping(), isEdge=isEdge)

        # Pid of each collectCounters process is written in a different file - to be able to stop them gracefully
        with open("{1}load_{0}.pid".format(name, tmp_files),"w") as f:
            f.write(str(os.getpid()))

        # Callibrate counters with the first lecture
        counters.getCounters()

        start_time = time.time()
        i = 1

        d = {"in": {}, "out": {}}
        while True:
            try:
                # I do this so it does not get too big and computing the w_time is not expensive
                if i == 1000:
                    start_time = time.time()
                    i = 1
                #time.sleep(interval - counters.fromLastLecture())
                w_time = (start_time + i*interval - time.time())
                if w_time < 0:
                    w_time = 0
                time.sleep(w_time)

                # Get the counter read-outs
                counters.getCounters()
                # Write it in the load tmp file
                with open("{1}load_{0}_tmp".format(name,tmp_files), "w") as f:
                    json.dump(counters.link_capacity(),f)

                # mv it.
                os.rename("{1}load_{0}_tmp".format(name,tmp_files) ,"{1}load_{0}".format(name,tmp_files))
                i +=1

            except KeyboardInterrupt:
                # time_file.close()
                break

            except Exception, e:
                # time_file.write(str(e))
                # time_file.write("w_time:"+str(w_time))
                # time_file.flush()
                # time_file.close()
                # load_file.close()
                break

    except Exception:
        import traceback
        with open("/tmp/"+name + "debug","w") as f:
            f.write(traceback.format_exc())


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group()

    parser.add_argument('-n', '--name', help='Host name', default='h_0_0')

    parser.add_argument('-t', '--time', help='Polling interval', type=float, default=1.5)

    args = parser.parse_args()

    collectCounters(name=args.name, interval=args.time)