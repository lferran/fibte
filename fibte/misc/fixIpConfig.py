import os
import pickle
import subprocess

from fibte import FIX_IPS_CONFIG

class FixIpsConfig(object):
    def __init__(self):
        self.loadIps()

    def loadIps(self):
        """Loads the stored hash seeds from the file"""
        if os.path.isfile(FIX_IPS_CONFIG):
            with open(FIX_IPS_CONFIG,"r") as f:
                self.fix_ips = pickle.load(f)
        else:
            self.fix_ips = {}

    def saveFixIps(self):
        """Stores the hash seeds in the file"""
        with open(FIX_IPS_CONFIG+"_tmp", 'w') as f:
            pickle.dump(self.fix_ips, f)
        subprocess.Popen(["mv", "{0}_tmp".format(FIX_IPS_CONFIG), FIX_IPS_CONFIG])

    def getIp(self, name):
        """Gets the current seed of the node defined by name"""
        return self.fix_ips.get(name, None)

    def setIp(self, name, ip):
        """Checks whether the host is in the seeds database. If it isn't, we add it.
        It does not rewrite the seed, but updates the database.
        """
        self.fix_ips[name] = ip
        self.saveFixIps()