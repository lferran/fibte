from fibte import flowServer_path, tmp_files,db_topo
from fibte.misc.topology_graph import TopologyGraph
import os
import argparse
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        help='Choose host where to start the flowServer in',
                        default = None)

    parser.add_argument('--ip_alias',
                        help='Are aliases active for mice flows?',
                        action='store_true',
                        default = False)

    args = parser.parse_args()

    # Load topodb
    topology = TopologyGraph(getIfindexes=False, db=os.path.join(tmp_files, db_topo))
    if args.host:
        print('*** Starting a single flowServer')
        hostname = args.host
        host_ip = topology.getHostIp(hostname)
        if args.ip_alias:
            command = "mx {0} {1} --name {0} --ip_alias --own_ip {2}"
        else:
            command = "mx {0} {1} --name {0} --own_ip {2}"
        os.system(command.format(hostname, flowServer_path, host_ip))

    else:
        print('*** Starting Flow Servers')
        for h in topology.getHosts().keys():
            host_ip = topology.getHostIp(h)
            if args.ip_alias: command = "mx {0} {1} --name {0} --ip_alias --own_ip {2} &"
            else: command = "mx {0} {1} --name {0} --own_ip {2} &"
            subprocess.call(command.format(h, flowServer_path, host_ip), shell=True)