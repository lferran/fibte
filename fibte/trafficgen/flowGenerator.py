import time
import socket
import argparse
import sys

#thats what I get checking wireshark and the ovs switches
#in theory it should be 54
minSizeUDP = 42
#maxUDPSize = 1500
maxUDPSize = 65000
import math

from fibte import ELEPHANT_SIZE_RANGE

from fibte.misc.unixSockets import UnixClient

import json

import logging
from fibte.logger import log

def setSizeToInt(size):
    """" Converts the sizes string notation to the corresponding integer
    (in bytes).  Input size can be given with the following
    magnitudes: B, K, M and G.
    """
    if isinstance(size, int):
        return size
    try:
        conversions = {'B': 1, 'K': 1e3, 'M': 1e6, 'G': 1e9}
        digits_list = range(48,58) + [ord(".")]
        magnitude = chr(sum([ord(x) if (ord(x) not in digits_list) else 0 for x in size]))
        digit = float(size[0:(size.index(magnitude))])
        magnitude = conversions[magnitude]
        return int(magnitude*digit)
    except:
        return 0

def sendMax(dst,dport):

    while True:
        s.sendto("A"*maxUDPSize,(dst,dport))

def sendRate_old(s,dst,dport,bytesPerSec):
    now=time.time()
    while bytesPerSec > minSizeUDP:
        bytesPerSec -= (s.sendto("A"*min(maxUDPSize,bytesPerSec-minSizeUDP),(dst,dport)) + minSizeUDP)
    #print  time.time()-now
    time.sleep(max(0,1-(time.time()-now)))

def sendRate(s,dst,dport,bytesPerSec,length=None):
    if length:
        maxUDPSize = length
    times = math.ceil(float(bytesPerSec) / (maxUDPSize+minSizeUDP))
    time_step= 1/times
    start = time.time()
    i = 0
    while bytesPerSec > minSizeUDP:
        bytesPerSec -= (s.sendto("A"*min(maxUDPSize,bytesPerSec-minSizeUDP),(dst,dport)) + minSizeUDP)
        i +=1
        next_send_time = start + i * time_step
        time.sleep(max(0,next_send_time - time.time()))
    print time.time()-start
    time.sleep(max(0,1-(time.time()-start)))

def sendRate_batch(s,dst,dport,bytesPerSec,length=None,packets_round=1):
    if length:
        maxUDPSize = length
    times = math.ceil((float(bytesPerSec) / (maxUDPSize+minSizeUDP))/packets_round)
    time_step= 1/times
    start = time.time()
    i = 0
    while bytesPerSec > minSizeUDP:
        for _ in range(packets_round):
            bytesPerSec -= (s.sendto("A"*min(maxUDPSize,bytesPerSec-minSizeUDP),(dst,dport)) + minSizeUDP)
        i +=1
        next_send_time = start + (i * time_step)
        time.sleep(max(0,next_send_time - time.time()))
    time.sleep(max(0,1-(time.time()-start)))

def die(signal,frame):
    sys.exit(0)

def sendFlow(dst="10.0.32.2",sport=5000,size='10M',dport=5001,duration=10,**kwargs):
    #register signal handler
    #signal.signal(signal.SIGTERM,die)
    #rates are in bits per second
    rate = setSizeToInt(size)/8

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('',sport))

    totalTime = int(duration)

    startTime = time.time()
    while (time.time() - startTime < totalTime):
        sendRate_batch(s,dst,dport,rate,length=10000, packets_round=5)
        #sendRate_old(s, dst, dport, rate)

def sendRound(socket,dst,rate,dport,offset):

    while rate > 0 and dport < 65535:
        for destination in dst[offset:]:
            socket.sendto("",(destination,dport))
            rate -=1
            if rate == 0:
                dport +=1
                break

        #so we onlyoffset the first round
        if offset != 0:
            offset = 0
        dport +=1

    return dport, dst.index(destination)+1

def keepSending(initialDestinations,rate,totalTime):

    dport = 6005
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    #if rate is samller than the amount of hosts we set it to that value

    startTime = time.time()
    offset = 0
    while (time.time() - startTime < totalTime):
        now = time.time()
        dport,offset= sendRound(s,initialDestinations,rate,dport,offset)
        time.sleep(max(0,1-(time.time()-now)))

        #restart the source port
        if dport == 65535:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            dport = 6005

def isElephant(flow):
    return flow['size'] >= ELEPHANT_SIZE_RANGE[0]
            
def sendFlowNotifyController(**flow):
    # Store time so we sleep 1 seconds - time needed for the following commands
    now  = time.time()

    # Start controller client
    client = UnixClient("/tmp/controllerServer")

    # Tell controller that flow will start
    if isElephant(flow):
        try:
            log.debug("New ELEPHANT is STARTING: to {0} {1}(bps) during {2}".format(flow['dst'], flow['size'], flow['duration']))
            # Notify controller that an elephant flow started
            client.send(json.dumps({"type": "startingFlow", "flow": flow}), "")
        except socket.error, v:
            log.error("[Connectoin refused] Controller cound not be informed.")
    # Close the socket
    client.sock.close()

    # Start sending flow
    sendFlow(**flow)

    
def stopFlowNotifyController(**flow):
    # Open socket with controller
    client = UnixClient("/tmp/controllerServer")

    log.debug("New ELEPHANT is STOPPING: to {0} {1}(bps) during {2}".format(flow['dst'], flow['size'], flow['duration']))
    
    # Notify controller that elephant flow finished
    client.send(json.dumps({"type": "stoppingFlow", "flow": flow}), "")

    # Close socket
    client.close()    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #group = parser.add_mutually_exclusive_group()

    parser.add_argument('-s', '--server',
                       help='',
                       default='10.0.32.3')

    parser.add_argument('-c', '--cport',
                       help='',
                       default='5001')

    parser.add_argument('-b', '--bytes',
                       help='',
                       default='1M')

    parser.add_argument('-p', '--dport',
                       help='',
                       default='5000')

    parser.add_argument('-t', '--time',
                       help='',
                       default='500')


    parser.add_argument('-d', '--debug',
                       help='Does not start sending data if set to something',
                       default="")

    args = parser.parse_args()

    dst = args.server
    rate = setSizeToInt(args.bytes)/8
    dport = int(args.dport)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('',int(args.cport)))

    totalTime = int(args.time)

    startTime = time.time()
    if not(args.debug):
        while (time.time() - startTime < totalTime):
            sendRate(s,dst,dport,rate)
