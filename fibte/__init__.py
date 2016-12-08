import ConfigParser
import os
import pkgutil
import time

RES = os.path.join(os.path.dirname(__file__),'res')

CFG = ConfigParser.ConfigParser()

with open(os.path.join(RES,'config.cfg'),'r') as f:
    CFG.readfp(f)

tmp_files = CFG.get("DEFAULT","tmp_files")
db_topo = CFG.get("DEFAULT","db_topo")
UDS_server_name = CFG.get("DEFAULT","controller_UDS_name")
UDS_server_traceroute = CFG.get("DEFAULT", 'controller_UDS_traceroute')
C1_cfg = CFG.get("DEFAULT", "C1_cfg")
LINK_BANDWIDTH = float(CFG.get("DEFAULT","link_bandwidth"))

MIN_MICE_SIZE = float(CFG.get("DEFAULT", "min_mice_size"))*LINK_BANDWIDTH
MAX_MICE_SIZE = float(CFG.get("DEFAULT", "max_mice_size"))*LINK_BANDWIDTH
MIN_ELEPHANT_SIZE = float(CFG.get("DEFAULT", "min_elephant_size"))*LINK_BANDWIDTH
MAX_ELEPHANT_SIZE = float(CFG.get("DEFAULT", "max_elephant_size"))*LINK_BANDWIDTH

MICE_SIZE_RANGE = [MIN_MICE_SIZE, MAX_MICE_SIZE]
ELEPHANT_SIZE_RANGE = [MIN_ELEPHANT_SIZE, MAX_ELEPHANT_SIZE]

MICE_SIZE_STEP = float(CFG.get("DEFAULT", "mice_size_step"))*LINK_BANDWIDTH
ELEPHANT_SIZE_STEP = float(CFG.get("DEFAULT", "elephant_size_step"))*LINK_BANDWIDTH

flowServer_path = pkgutil.get_loader("fibte.trafficgen.flowServer").filename
iptables_path = pkgutil.get_loader("fibte.misc.iptables").filename
ifDescrNamespace_path= pkgutil.get_loader("fibte.monitoring.ifDescrNamespace").filename
counterCollector_path = pkgutil.get_loader("fibte.monitoring.collectCounters").filename
getLoads_path = pkgutil.get_loader("fibte.monitoring.getLoads").filename

HASH_SEED_CONFIG = os.path.dirname(__file__)+"/hash_seed.config"
FIX_IPS_CONFIG = os.path.dirname(__file__)+"/fix_ips.config"
COLORS_CONFIG = os.path.dirname(__file__)+"/colors.config"

import logging
from fibte.logger import log

# Define decorator
def time_func(function):
    def wrapper(*args,**kwargs):
        t = time.time()
        res = function(*args,**kwargs)
        log.debug("{0} took {1}s to execute".format(function.func_name, time.time()-t))
        return res
    return wrapper