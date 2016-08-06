#!/usr/bin/python
from fibte.monitoring.countersDev import IfDescr
import sys

"""
Script to be runned within the namespace. It gets ifnames and ifindexes from /proc/net/dev and
stores them in a file so that scritps running in the root namespace can access to interface information.

credit: Edgar Costa - edgarcosta@hotmail.es
"""

if __name__ == "__main__":
    path = sys.argv[1]
    IfDescr().saveIfMapping(path)