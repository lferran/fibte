import ConfigParser
import os
import pkgutil
RES = os.path.join(os.path.dirname(__file__),'res')

CFG = ConfigParser.ConfigParser()

with open(os.path.join(RES,'config.cfg'),'r') as f:
    CFG.readfp(f)

tmp_files = CFG.get("DEFAULT","tmp_files")
db_topo = CFG.get("DEFAULT","db_topo")

LINK_BANDWIDTH = float(CFG.get("DEFAULT","link_bandwidth"))

MIN_MICE_SIZE = float(CFG.get("DEFAULT", "min_mice_size"))*LINK_BANDWIDTH
MAX_MICE_SIZE = float(CFG.get("DEFAULT", "max_mice_size"))*LINK_BANDWIDTH
MIN_ELEPHANT_SIZE = float(CFG.get("DEFAULT", "min_elephant_size"))*LINK_BANDWIDTH
MAX_ELEPHANT_SIZE = float(CFG.get("DEFAULT", "max_elephant_size"))*LINK_BANDWIDTH

MICE_SIZE_RANGE = [MIN_MICE_SIZE, MAX_MICE_SIZE]
ELEPHANT_SIZE_RANGE = [MIN_ELEPHANT_SIZE, MAX_ELEPHANT_SIZE]

MICE_SIZE_STEP = float(CFG.get("DEFAULT", "mice_size_step"))*LINK_BANDWIDTH
ELEPHANT_SIZE_STEP = float(CFG.get("DEFAULT", "elephant_size_step"))*LINK_BANDWIDTH

import inspect
import fibte.trafficgen.flowServer
flowServer_path = inspect.getsourcefile(fibte.trafficgen.flowServer)
iptables_path = pkgutil.get_loader("fibte.misc.iptables").filename