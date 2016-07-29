import mininet.node as _node
import logging

from fibte.logger import log

class MyCustomHost(_node.Host):
    def __init__(self, *args, **kwargs):
        super(MyCustomHost, self).__init__(*args, **kwargs)

        if 'fibte' in kwargs.keys() and kwargs.get('fibte') == True:
            log.setLevel(logging.DEBUG)
            log.info()