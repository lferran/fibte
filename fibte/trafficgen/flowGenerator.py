import time
import socket
import argparse
import signal
import sys

#thats what I get checking wireshark and the ovs switches
#in theory it should be 54
minSizeUDP = 42
#maxUDPSize = 1500
maxUDPSize = 65507

from fibte.misc.unixSockets import UnixClient
import json

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
        s.sendto("A"*maxUDPSize, (dst,dport))

def sendRate(s, dst, dport, bytesPerSec):
    now = time.time()
    while bytesPerSec > minSizeUDP:
        bytesPerSec -= (s.sendto("A"*min(maxUDPSize, bytesPerSec - minSizeUDP), (dst, dport)) + minSizeUDP)
    print  time.time() - now
    time.sleep(max(0,1 - (time.time() - now)))

def die(signal, frame):
    sys.exit(0)

def sendFlow(dst="10.0.32.2", sport=5000, size='10M', dport=5001, duration = 10,**kwargs):
    #register signal handler
    #signal.signal(signal.SIGTERM,die)

    # Rates are in bits per second
    rate = setSizeToInt(size)/8

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('',sport))

    totalTime = int(duration)

    startTime = time.time()
    while (time.time() - startTime < totalTime):
        sendRate(s, dst, dport, rate)


def sendRound(socket,dst,rate,dport,offset):
    while rate > 0 and dport < 65535:
        for destination in dst[offset:]:
            socket.sendto("",(destination,dport))
            rate -=1
            if rate == 0:
                dport +=1
                break

        # So we only offset the first round
        if offset != 0:
            offset = 0
        dport +=1
    return dport, dst.index(destination) + 1


def keepSending(initialDestinations, rate, totalTime):
    dport = 6005
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # If rate is samller than the amount of hosts we set it to that value
    startTime = time.time()
    offset = 0
    while (time.time() - startTime < totalTime):
        now = time.time()
        dport,offset= sendRound(s, initialDestinations, rate, dport, offset)
        time.sleep(max(0,1-(time.time()-now)))

        #restart the source port
        if dport == 65535:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            dport = 6005


def sendFlowNotifyController(**flow):
    client = UnixClient("/tmp/controllerServer")
    # Tell controller that flow will start - only if it is an elephant flow

    if flow["duration"] >= 20:
        client.send(json.dumps({"type": "startingFlow", "flow": flow}),"")
    #time.sleep(1)
    # Start flow
    sendFlow(**flow)

    if flow["duration"] >= 20:
        client.send(json.dumps({"type" : "stoppingFlow", "flow" : flow}),"")
    client.sock.close()

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
                       help='flow duration in seconds',
                       default='500')


    parser.add_argument('-d', '--debug',
                       help='Does not start sending data if set to something',
                       default="")

    args = parser.parse_args()

    dst = args.server
    rate = setSizeToInt(args.bytes)/8
    dport = int(args.dport)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('', int(args.cport)))

    totalTime = int(args.time)

    startTime = time.time()
    if not(args.debug):
        while (time.time() - startTime < totalTime):
            sendRate(s, dst, dport, rate)


