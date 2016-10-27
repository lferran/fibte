import subprocess, signal, os

def areFlowServersRunning():
    """Check if flowServers are running"""
    p = subprocess.Popen("ps aux | grep 'flowServer'", stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if not err:
        out = out.split('\n')
        if len(out) > 3:
            return True
    return False

def killFlowServers():
    p = subprocess.Popen("ps aux", stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    for line in out.splitlines():
        if 'flowServer' in line:
            pid = int(line.split(' ')[5])
            os.kill(pid, signal.SIGTERM)


print ("*** Stopping Flow Servers")
if areFlowServersRunning():
    try:
        killFlowServers()
    except:
        subprocess.call("kill -9 $(ps aux | grep 'flowServer' | awk '{print $2}')", shell=True)
    finally:
        if areFlowServersRunning():
            subprocess.call("kill -9 $(ps aux | grep 'flowServer' | awk '{print $2}')", shell=True)

        if areFlowServersRunning():
            print("ERROR: FlowServers couldn't be stopped!")