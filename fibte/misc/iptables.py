#!/usr/bin/python
import subprocess
from sys import argv

# OSPF: marks ospf packets with priority 0x14 (20)
subprocess.call("iptables -w -t mangle -A POSTROUTING -o %s -p 89 -j MARK --set-mark 20" % argv[1], shell=True)

# ICMP: marks icmp packets with priority 0xa (10)
subprocess.call("iptables -w -t mangle -A POSTROUTING -o %s -p 1 -j MARK --set-mark 10" % argv[1], shell=True)

# Traceroute: marks pakets with TTL < 10 that are not OSPF with priority 0xa (10)
subprocess.call("iptables -w -t mangle -A POSTROUTING -o %s -m ttl --ttl-lt 10 ! -p 89 -j MARK --set-mark 10" % argv[1], shell=True)

# TCP flags: Gives priority 20 to TCP SYN and ACK packets
subprocess.call("iptables -w -t mangle -A POSTROUTING -o %s -m ttl --ttl-gt 10 -p tcp --tcp-flags ACK ACK -m length --length :64 -j MARK --set-mark 20" % argv[1], shell=True)
subprocess.call("iptables -w -t mangle -A POSTROUTING -o %s -m ttl --ttl-gt 10 -p tcp --tcp-flags SYN SYN -m length --length :64 -j MARK --set-mark 20" % argv[1], shell=True)