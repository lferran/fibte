import subprocess
import sys

if __name__ == '__main__':
    subprocess.call("python stop_flow_servers.py", shell=True)
    if len(sys.argv) > 1:
        # Pass the parameters to the start_flow_servers script
        args = ' '.join(sys.argv[1:])
        subprocess.call("python start_flow_servers.py {0}".format(args), shell=True)
    else:
        subprocess.call("python start_flow_servers.py", shell=True)