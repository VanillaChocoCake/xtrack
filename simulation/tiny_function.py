import numpy as np


def cal_weight(num:int, steepness=None, gain=None):
    """

    Parameters
    ----------
    num: number of points
    steepness: bigger, steeper
    gain: dB

    Returns
    -------
    weight: to make the Schottky spectrum steeper

    """
    if gain is None:
        gain = 100
    if steepness is None:
        steepness = 3
    weight_range = {'min': 0, 'max': np.power(10, gain / 20)}
    x = np.arange(num) * np.pi / num
    weight = np.power(np.sin(x), steepness)
    weight = weight - np.min(weight)
    weight = weight * (weight_range['max'] - weight_range['min']) / np.max(weight)
    weight = weight + weight_range['min']
    return weight
