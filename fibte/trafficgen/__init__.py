
# Assuming the networks are /24, this would be the
# address reserved for all hosts's secondary ips
IP_ALIAS_ADDRESS = '222'

def setup_alias(host):
    """Setup alias at host h"""

    # Get default interface
    hintf = host.defaultIntf()
    # Get assigned ip
    hip = host.IP()

    # Remove host side
    alias_ip = get_secondary_ip(hip)

    command = "ifconfig {0}:0 {1} netmask 255.255.255.0".format(hintf, alias_ip)

    print ("{0} ({1})\t miceIP: {2}\t elephIP: {3}".format(host.name, str(hintf), host.IP() ,alias_ip))

    # Run command
    host.cmd(command)

def get_secondary_ip(dst_ip):
    """
    :param dst_ip: string representing an ipv4 address
    :return:
    """
    new_ip = dst_ip.split('.')[:-1] + [IP_ALIAS_ADDRESS]
    return '.'.join(new_ip)

def get_secondary_ip_prefix(dst_prefix):
    """

    :param dst_prefix: string representing an ipv4 network prefix
    :return:
    """
    new_dst = dst_prefix.split('/')[0].split('.')[:-1] + [IP_ALIAS_ADDRESS]
    new_dst = '.'.join(new_dst) + '/32'
    return new_dst