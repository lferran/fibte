import ConfigParser
import os

# Get the path of the res folder
RES = os.path.join(os.path.dirname(__file__), 'res')

# Instantiate the configuration parser
CFG = ConfigParser.ConfigParser()

# Load res/config.cfg and parse it with the parser
with open(os.path.join(RES,'config.cfg'),'r') as f:
    CFG.readfp(f)

import inspect
import fibte.trafficgen.flowServer

# Get the path of the flowServer script that receives
# instructions from the Traffic Generator to start flows
flowServer_path= inspect.getsourcefile(fibte.trafficgen.flowServer)

# Get some other configs
tmp_files = CFG.get("DEFAULT","tmp_files")
db_topo = CFG.get("DEFAULT","db_topo")