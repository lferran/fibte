import mininet.node as _node
import logging
import subprocess
import fibte.res.config as cfg

from fibbingnode.misc.mininetlib import get_logger

log = get_logger()


class MyCustomHost(_node.Host):
    def __init__(self, *args, **kwargs):
        super(MyCustomHost, self).__init__(*args, **kwargs)
