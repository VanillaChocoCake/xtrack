import numpy as np


def cal_weight(num:int, steepness=2, weight_range=None):
    if weight_range is None:
        weight_range = {'min': 0, 'max': 1e3}
    x = np.arange(num) * np.pi / num
    weight = np.power(np.sin(x), steepness)
    weight = weight - np.min(weight)
    weight = weight * (weight_range['max'] - weight_range['min']) / np.max(weight)
    weight = weight + weight_range['min']
    return weight
