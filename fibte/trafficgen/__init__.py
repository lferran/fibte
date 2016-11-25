from fibte import ELEPHANT_SIZE_RANGE

def isElephant(flow):
    """
    Function that cheks if flow is elephant

    returns: boolean
    """
    if flow['proto'].lower() == 'udp':
        return flow['size'] >= ELEPHANT_SIZE_RANGE[0]

    elif flow['proto'].lower() == 'tcp':
        duration = flow.get('size')/float(flow.get('rate'))
        return duration >= 20

def isMice(flow):
    """
    Checkis if flow is considered a mice flow

    :return: boolean
    """
    return not isElephant(flow)