import inspect
import ConfigParser
import os

RES= os.path.join(os.path.dirname(__file__),'res')

CFG = ConfigParser.ConfigParser()

with open(os.path.join(RES,'config.cfg'),'r') as f:
    CFG.readfp(f)

tmp_files = CFG.get("DEFAULT","tmp_files")
db_topo = CFG.get("DEFAULT","db_topo")
LINK_BANDWIDTH = float(CFG.get("DEFAULT","link_bandwidth"))

import fibte.trafficgen.flowServer
flowServer_path= inspect.getsourcefile(fibte.trafficgen.flowServer)