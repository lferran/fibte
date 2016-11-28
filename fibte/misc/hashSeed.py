import os
import pickle
import subprocess

from fibte import HASH_SEED_CONFIG

class HashSeedConfig(object):
    def __init__(self):
        self.loadHashSeeds()

    def loadHashSeeds(self):
        """Loads the stored hash seeds from the file"""
        if os.path.isfile(HASH_SEED_CONFIG):
            with open(HASH_SEED_CONFIG,"r") as f:
                self.hash_seeds = pickle.load(f)
        else:
            self.hash_seeds = {}

    def saveHashSeeds(self):
        """Stores the hash seeds in the file"""
        with open(HASH_SEED_CONFIG+"_tmp", 'w') as f:
            pickle.dump(self.hash_seeds, f)
        subprocess.Popen(["mv", "{0}_tmp".format(HASH_SEED_CONFIG), HASH_SEED_CONFIG])

    def getSeed(self, name):
        """Gets the current seed of the node defined by name"""
        seed = self.hash_seeds.get(name, None)
        if seed:
            return seed
        else:
            handler = subprocess.Popen(["mx", name, "sysctl","net.ipv4.route.mphash_perturb"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = handler.communicate()
            if not err and out:
                try:
                    seed = out.split("=")[1].strip()
                    return seed
                except IndexError:
                    print("ERROR: Could not get the previously stored hash seed\n")
            else:
                print("ERROR: Could not get the previously stored hash seed\n")
            return None

    def setSeed(self, name):
        """Checks whether the host is in the seeds database. If it isn't, we add it.
        It does not rewrite the seed, but updates the database.
        """
        # Get the current seed
        seed = self.getSeed(name)

        # If it is already in the dictionary, we set the current one
        if self.hash_seeds.has_key(name):
            handler = subprocess.Popen(["mx", name, "sysctl","-w","net.ipv4.route.mphash_perturb=%s"%seed], stderr=subprocess.PIPE)
            out, err = handler.communicate()
            if err:
                print("ERROR: The seed could not be set for {0}".format(name))

        # Else we save it in the dictionary
        elif seed:
            self.hash_seeds[name] = seed
            self.saveHashSeeds()