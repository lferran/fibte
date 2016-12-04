from fibte import ELEPHANT_SIZE_RANGE

def isElephant(flow):
    """
    Function that cheks if flow is elephant

    returns: boolean
    """
    if flow['proto'].lower() == 'udp':
        return flow['size'] >= ELEPHANT_SIZE_RANGE[0]

    elif flow['proto'].lower() == 'tcp':
        # Check duration first
        duration = flow.get('size')/float(flow.get('rate'))
        if duration >= 20:
            return True
        else:
            # Check the data to send
            to_send_bits = flow.get('size')
            to_send_kbits = to_send_bits/1000.0
            to_send_kB = to_send_kbits/8
            if to_send_kB > 200:
                return True
            else:
                return False

def isMice(flow):
    """
    Checkis if flow is considered a mice flow

    :return: boolean
    """
    return not isElephant(flow)



nonNICCongestionTest = False