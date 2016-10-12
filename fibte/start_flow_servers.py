from fibte import flowServer_path, tmp_files,db_topo
from fibte.misc.topology_graph import TopologyGraph
import os
import argparse
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hosts',
                        help='Choose host where to start the flowServer in',
                        default = None)
    parser.add_argument('--ip_alias',
                        help='Are aliases active for mice flows?',
                        action='store_true',
                        default = False)

    args = parser.parse_args()
    if args.hosts:
        hosts = args.hosts.split(',')
    else:
        hosts = []

    print('*** Start Flow Servers and Learning Servers')
    topology = TopologyGraph(getIfindexes=False, db=os.path.join(tmp_files,db_topo))
    for h in topology.getHosts().keys():
        if hosts:
            if h in hosts:
                if args.ip_alias:
                    command = "mx {0} {1} {0} {2} &"
                    os.system(command.format(h, flowServer_path, args.ip_alias))
                else:
                    command = "mx {0} {1} {0} &"
                    os.system(command.format(h, flowServer_path))
        else:
            if args.ip_alias:
                command = "mx {0} {1} {0} {2} &"
                subprocess.call(command.format(h, flowServer_path, args.ip_alias), shell=True)
            else:
                command = "mx {0} {1} {0} &"
                subprocess.call(command.format(h, flowServer_path), shell=True)