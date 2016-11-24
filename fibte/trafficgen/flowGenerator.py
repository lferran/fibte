import time
import socket
import argparse
import sys
import struct
import math
import json
import logging
import subprocess
import os

from fibte.misc.unixSockets import UnixClientTCP
from fibte.logger import log
from fibte import LINK_BANDWIDTH

#thats what I get checking wireshark and the ovs switches
#in theory it should be 54
minSizeUDP = 42
minSizeTCP = 66
maxUDPSize = 10000

# Time that we sleep between notification
# is sent and the flow starts
SLEEP_BEFORE_FLOW_S = 0

delay_folder = os.path.join(os.path.dirname(__file__), '../monitoring/results/delay/')

def setSizeToInt(size):
    """" Converts the sizes string notation to the corresponding integer
    (in bytes).  Input size can be given with the following
    magnitudes: B, K, M and G.
    """
    if isinstance(size, int):
        return size
    if isinstance(size, float):
        return int(size)
    try:
        conversions = {'B': 1, 'K': 1e3, 'M': 1e6, 'G': 1e9}
        digits_list = range(48,58) + [ord(".")]
        magnitude = chr(sum([ord(x) if (ord(x) not in digits_list) else 0 for x in size]))
        digit = float(size[0:(size.index(magnitude))])
        magnitude = conversions[magnitude]
        return int(magnitude*digit)
    except:
        return 0

# TCP FUNCTIONS
def send_msg(sock, msg):
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = ''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def sendFlowTCP(dst="10.0.32.3", sport=5000, dport=5001, size=None, rate=None, duration=None,**kwargs):
    """
    :param size: Size of the data we want to send
    :param rate: Rate at which we want to send it. If not specified, we will try to send as much as possible
    :param duration: Expected in seconds
    """
    # Setup TCP connection
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1500)
    s.bind(('', sport))
    try:
        reconnections = 4
        time_to_wait = 0.3
        while reconnections:
            try:
                s.connect((dst, dport))
                break
            except Exception as e:
                reconnections -=1
                print ("Trying to reconnect to {1}:{2}... trials left {0}".format(reconnections, dst, dport))
                print ("Error trace: {0}".format(e))
                time.sleep(time_to_wait)
                time_to_wait *= 2

        # Could not connect to the server
        if reconnections == 0:
            print "We couldn't connect to the server {1}:{0}! Returning...".format(dport, dst)
            s.close()
            return False

        # If rate and duration are specified
        if not size and rate and duration:
            # Convert rate to integer (bytes/s)
            rate = setSizeToInt(rate)/8

            # Minimum time (assuming the rate we limit is possible)
            totalTime = int(duration)

            # Compute real rate and size due to TCP/IP overhead
            headers_overhead = minSizeTCP * (rate / 4096)
            rate = rate - (headers_overhead)

            # Start sending at specified rate durint that amount of time
            startTime = time.time()
            # Round counter
            i = 0
            time_step_s = 1
            # While totalTime hasn't passed yet
            while (time.time() - startTime <= totalTime):
                # Send at specified rate
                send_msg(s, "A" * rate)
                i += 1
                # Sleep until the next second
                next_send_time = startTime + i * time_step_s
                time.sleep(max(0, next_send_time - time.time()))

        # If size is given
        elif size:
            # If rate not specified
            if not rate:
                # We try at highest rate possible
                rate = LINK_BANDWIDTH
            else:
                # Limit it to LINK_BANDWIDTH
                rate = min(setSizeToInt(rate), LINK_BANDWIDTH)

            # Express it in bytes and bytes/s
            totalSize = setSizeToInt(size)/8
            rate = rate/8

            # Compute real rate and size due to TCP/IP overhead
            headers_overhead = minSizeTCP * (rate / 4096)
            headers_overhead_total = minSizeTCP * (totalSize / 4096)
            rate = rate - (headers_overhead)
            totalSize = totalSize - (headers_overhead_total)

            # We send the size in bytes with rate as maximum rate
            startTime = time.time()
            i = 0
            time_step_s = 1
            # While still some data to send
            while (totalSize > minSizeTCP):
                #print "Sending a bit more..."
                # Compute data to send next second
                rate = min(rate, totalSize - minSizeUDP)
                # Send it
                send_msg(s, "A" * rate)
                # Substract it from the remaining data
                totalSize -= rate
                # Increment round count
                i += 1
                # Compute how much to sleep until the next second
                next_send_time = startTime + i * time_step_s
                # Sleep
                time.sleep(max(0, next_send_time - time.time()))
            #print "Finished sending!"

        else:
            print "Wrong arguments!"
            s.close()
            return False

    except socket.error as e:
        print "socket.error: {0}".format(e)
        s.close()
        return False

    finally:
        s.close()
        #print "Finishing flow gracefully"
        return True

def setupTCPConnection(dst="10.0.32.3", sport=5000, dport=5001, **kwargs):
    # Setup TCP connection
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1500)
    s.bind(('', sport))

    reconnections = 1
    time_to_wait = 0.1
    while reconnections != 0:
        try:
            s.connect((dst, dport))
            break
        except Exception as e:
            reconnections -= 1
            print ("Trying to reconnect to {1}:{2}... trials left {0}".format(reconnections, dst, dport))
            print ("Error trace: {0}".format(e))
            time.sleep(time_to_wait)
            time_to_wait *= 2

    # Could not connect to the server
    if reconnections == 0:
        print "We couldn't connect to the server {1}:{0}! Returning...".format(dport, dst)
        s.close()
        return None

    else:
        # Return open socket with the established connection
        return s

def sendMiceThroughOpenSocket(s, queue, sending, completionTimeFile=None):
    """
    s: open TCP socket
    queue: queue from where it gets the events: new mice sizes to start and terminate
    sending: threading event that is set() when the function is sending, clear() otherwise
    completionTimeFile: file name pattern where the completion times will be written
    """
    try:
        terminate = False
        round = 0
        fname_pattern = completionTimeFile

        while not terminate:
            # Blocking get from the queue
            event = queue.get()
            queue.task_done()

            if event == 'terminate':
                print("Terminate received")
                terminate = True
                continue

            else:
                size = event
                rate = None
                #print("ROUND: {0}".format(round))

                # Set state to sending
                sending.set()

                # If rate not specified
                if not rate:
                    # We try at highest rate possible
                    rate = setSizeToInt(LINK_BANDWIDTH)

                else:
                    # Limit it to LINK_BANDWIDTH
                    rate = min(setSizeToInt(rate), LINK_BANDWIDTH)

                # Express it in bytes and bytes/s
                totalSize = setSizeToInt(size)/8
                rate = rate/8

                # Compute real rate and size due to TCP/IP overhead
                headers_overhead = minSizeTCP * (rate / 4096)
                headers_overhead_total = minSizeTCP * (totalSize / 4096)
                rate = rate - (headers_overhead)
                totalSize = totalSize - (headers_overhead_total)

                estimated_duration = totalSize/float(rate)

                # We send the size in bytes with rate as maximum rate
                startTime = time.time()
                i = 0
                time_step_s = 1

                # Write down final time here
                if completionTimeFile:
                    filename = fname_pattern.format(round)
                    flow = {'proto': 'tcp', 'size': totalSize, 'rate': rate}
                    filename = writeStartingTime(flow, filename=filename)

                start = time.time()
                # While still some data to send
                while (totalSize > minSizeTCP):
                    #print "Sending a bit more..."
                    # Compute data to send next second
                    rate = min(rate, totalSize - minSizeUDP)
                    # Send it
                    send_msg(s, "A" * rate)
                    # Substract it from the remaining data
                    totalSize -= rate
                    # Increment round count
                    i += 1
                    # Compute how much to sleep until the next second
                    next_send_time = startTime + i * time_step_s
                    # Sleep
                    time.sleep(max(0, next_send_time - time.time()))

                # Write down final time here
                if completionTimeFile:
                    writeEndingTime(filename=filename)

                # Send status to finished
                sending.clear()

                # Increment round
                round = round + 1

    except socket.error as e:
        print "socket.error: {0}".format(e)
        s.close()
        return False

    except Exception as e:
        print "Other exception: {0}".format(e)

    finally:
        s.close()
        return True

def recvFlowTCP(dport=5001):
    subprocess.Popen(["nc", "-l", "-p", str(dport)], stdout=open(os.devnull, "w"))
    return

def sendRate(s,dst,dport,bytesPerSec,length=None):
    if not length:
        maxSize = maxUDPSize
    else:
        maxSize = length
    times = math.ceil(float(bytesPerSec) / (maxSize+minSizeUDP))
    time_step= 1/times
    start = time.time()
    i = 0
    while bytesPerSec > minSizeUDP:
        bytesPerSec -= (s.sendto("A"*min(maxSize, bytesPerSec - minSizeUDP), (dst,dport)) + minSizeUDP)
        i +=1
        next_send_time = start + i * time_step
        time.sleep(max(0,next_send_time - time.time()))
    print time.time()-start
    time.sleep(max(0,1-(time.time()-start)))

def sendRate_batch(s, dst, dport, bytesPerSec, length=None, packets_round=1):
    if not length:
        maxSize = maxUDPSize
    else:
        maxSize = length

    times = math.ceil((float(bytesPerSec)/(maxSize+minSizeUDP))/packets_round)
    time_step= 1/times
    start = time.time()
    i = 0
    while bytesPerSec > minSizeUDP:
        for _ in range(packets_round):
            bytesPerSec -= (s.sendto("A"*min(maxSize, bytesPerSec - minSizeUDP), (dst, dport)) + minSizeUDP)
        i +=1
        next_send_time = start + (i * time_step)
        time.sleep(max(0,next_send_time - time.time()))
    time.sleep(max(0,1-(time.time()-start)))

def die(signal, frame):
    sys.exit(0)

def sendFlowUDP(dst="10.0.32.2", sport=5000, size='10M', dport=5001, duration=10, **kwargs):
    #register signal handler
    #signal.signal(signal.SIGTERM,die)
    #rates are in bits per second
    rate = setSizeToInt(size)/8

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('',sport))

    totalTime = int(duration)
    try:
        startTime = time.time()
        while (time.time() - startTime < totalTime):
            sendRate_batch(s, dst, dport, rate, length=5000, packets_round=5)
    except:
        s.close()
        return False
    else:
        s.close()
        return True

def sendRound(socket, dst, rate, dport, offset):
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

def _sendFlow(notify=False, **flow):
    """
    Starts sending flow as specified and notifies the controller when needed
    """

    # Take the time
    now  = time.time()
    successful = False

    # Notify controller first (if needed)
    if notify:
        # Start controller client
        client = UnixClientTCP("/tmp/controllerServer")
        try:
            # Notify controller that an elephant flow started
            client.send(json.dumps({"type": "startingFlow", "flow": flow}), "")
        except Exception as e:
            log.error("Controller cound not be informed about startingFlow event")

    # Sleep a bit before starting the flow
    time.sleep(max(0, SLEEP_BEFORE_FLOW_S - (time.time() - now)))

    # If flow is UDP
    if flow['proto'] == 'UDP':
        # Start UDP flow
        successful = sendFlowUDP(**flow)

    else:# flow['proto'] == 'TCP':
        # Start TCP flow
        successful = sendFlowTCP(**flow)

    # Send a stop notification to the controller if needed
    if notify:
        try:
            client.send(json.dumps({"type": "stoppingFlow", "flow": flow}), "")
        except Exception as e:
            log.error("Controller cound not be informed about stoppingFlow event")
        finally:
            # Close socket
            client.close()

    if not successful:
        log.error("Flow didn't finish successfully!")

    return successful

def writeStartingTime(flow, filename=None):
    if not filename:
        file_name = str(delay_folder) + "{0}_{1}_{2}_{3}".format(flow["src"],
                                                                 flow["sport"],
                                                                 flow["dst"],
                                                                 flow["dport"])
    else:
        file_name = str(delay_folder) + filename

    if flow['proto'].lower() == 'udp':
        duration = flow.get('duration')
    else:
        duration = flow.get('size')/float(flow.get('rate'))

    # Save flow starting time
    with open(file_name, "w") as f:
        f.write("expected {0}".format(duration) + "\n")
        f.write(str(int(round(time.time() * 1000))) + "\n")
    return file_name

def writeEndingTime(filename):
    # Write finishing time
    with open(filename, "a") as f:
        f.write(str(int(round(time.time() * 1000))) + "\n")

def sendMiceFlow(logtime=False, **flow):
    file_name = None
    if flow['proto'].lower() == 'tcp' and logtime:
        # Save flow starting time
        file_name = writeStartingTime(flow)

    # Call internal sendFlow
    successful = _sendFlow(notify=False, **flow)

    if file_name and successful and logtime:
        writeEndingTime(file_name)

    # Exit the function gracefully
    sys.exit(0)

def sendElephantFlow(logtime=False, **flow):
    filename = None

    # Write starting time
    if flow['proto'].lower() == 'tcp' and logtime:
        filename = writeStartingTime(flow)

    # Call internal function
    successful = _sendFlow(notify=True, **flow)

    if filename and successful and logtime:
        writeEndingTime(filename)

    # Exit the function gracefully
    sys.exit(0)

# Not used
def stopFlowNotifyController(**flow):
    # Open socket with controller
    client = UnixClientTCP("/tmp/controllerServer")
    try:
        # Notify controller that elephant flow finished
        client.send(json.dumps({"type": "stoppingFlow", "flow": flow}), "")
    except Exception as e:
        log.error("Controller cound not be informed about stoppingFlow event")

    # Close socket
    client.close()

    sys.exit(0)

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
