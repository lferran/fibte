# Assuming the networks are /24, this would be the
# address reserved for all hosts's secondary ips
IP_ALIAS_ADDRESS = '222'
IP_ALIAS_NETWORK_PREFIXLEN = '/32'
DEFAULT_NETWORK_ADDR = '0'
DEFAULT_NETWORK_PREFIXLEN = '/24'

SECONDARY_ADDRESS_FOR = 'elephant'

def setup_alias(host):
    """Setup alias at host h"""

    # Get default interface
    hintf = host.defaultIntf()
    # Get assigned ip
    hip = host.IP()

    # Remove host side
    alias_ip = get_secondary_ip(hip)

    command = "ifconfig {0}:0 {1} netmask 255.255.255.0".format(hintf, alias_ip)

    print ("{0} ({1})\t primary ip: {2}\t secondary ip: {3}".format(host.name, str(hintf), host.IP(), alias_ip))

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
    new_dst = '.'.join(new_dst) + IP_ALIAS_NETWORK_PREFIXLEN
    return new_dst

def get_primary_ip_prefix(secondary_prefix):
    new_dst = secondary_prefix.split('/')[0].split('.')[:-1] + [DEFAULT_NETWORK_ADDR]
    new_dst = '.'.join(new_dst) + DEFAULT_NETWORK_PREFIXLEN
    return new_dst

def is_secondary_ip(dst_ip):
    """"""
    return dst_ip.split('.')[-1] == IP_ALIAS_ADDRESS

def is_secondary_ip_prefix(dst_prefix):
    """"""
    return dst_prefix.split('/')[0].split('.')[-1] == IP_ALIAS_ADDRESS

def get_ip_type(ip):
    """Returns either primary or secondary"""
    if is_secondary_ip(ip):
        return "secondary"
    else:
        return "primary"

def get_ip_prefix_type(prefix):
    """Returns either primary or secondary"""
    if is_secondary_ip_prefix(prefix):
        return "secondary"
    else:
        return "primary"