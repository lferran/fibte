
# Assuming the networks are /24, this would be the address reserved
# for the elephant ip addresses
ELEPHANT_IP_ALIAS_ADDRESS = '222'

def setup_alias(host):
    """Setup alias at host h"""

    # Get default interface
    hintf = host.defaultIntf()
    # Get assigned ip
    hip = host.IP()

    # Remove host side
    alias_ip = convert_to_elephant_ip(hip)

    command = "ifconfig {0}:0 {1} netmask 255.255.255.0".format(hintf, alias_ip)

    print ("{0} ({1})\t miceIP: {2}\t elephIP: {3}".format(host.name, str(hintf), host.IP() ,alias_ip))

    # Run command
    host.cmd(command)

def convert_to_elephant_ip(dst):
    new_ip = dst.split('.')[:-1] + [ELEPHANT_IP_ALIAS_ADDRESS]
    return '.'.join(new_ip)