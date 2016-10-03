from fibte import ELEPHANT_SIZE_RANGE

def isElephant(flow):
    """
    Function that cheks if flow is elephant

    returns: boolean
    """
    return flow['size'] >= ELEPHANT_SIZE_RANGE[0]