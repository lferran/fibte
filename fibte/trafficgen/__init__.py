from fibte import ELEPHANT_SIZE_RANGE

def isElephant(flow):
    """
    Function that cheks if flow is elephant

    returns: boolean
    """
    if flow['proto'] == 'UDP':
        return flow['size'] >= ELEPHANT_SIZE_RANGE[0]

    elif flow['proto'] == 'TCP':
        if flow.get('duration'):
            return flow['duration'] >= 20
        else:
            duration = flow.get('size')/flow.get('rate')
            return duration >= 20

def isMice(flow):
    """
    Checkis if flow is considered a mice flow

    :return: boolean
    """
    return not isElephant(flow)