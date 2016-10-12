import subprocess

subprocess.call("kill -9 $(ps aux | grep 'flowServer' | awk '{print $2}')", shell=True)