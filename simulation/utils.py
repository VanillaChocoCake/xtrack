"""
Toolkit for Spectral Analysis, Filtering and Algorithm Implementation
Author: Sun Peihan
Date: 2025-02-28
"""

import numpy as np
from scipy.fft import fft, fftshift
import pickle
import h5py
import scipy.constants as const

def cal_tune_shift_rate(chromaticity, Ek, circumference, rf_voltage):
    E0 = const.value('proton mass energy equivalent in MeV')
    gamma = Ek/E0 + 1
    beta = np.sqrt(1 - 1/gamma**2)
    E0_J = E0 * 1e6 * const.eV
    return chromaticity*const.e*rf_voltage*const.c/(beta*gamma*E0_J*circumference)

def read_pkl(filename: str) -> dict:
    """Load serialized Python objects from pickle file.

    Args:
        filename: Path to .pkl file

    Returns:
        Deserialized dictionary object

    Raises:
        FileNotFoundError: If specified file doesn't exist
        pickle.UnpicklingError: For corrupted pickle files
    """
    with open(filename, "rb") as f:
        dic = pickle.load(f)
    return dic

def load_matlab_v73(filename: str,
                    variable_name: str) -> np.ndarray:
    """Load MATLAB v7.3+ data files using HDF5 interface.

    Args:
        filename: Path to .mat file
        variable_name: Target dataset name

    Returns:
        Numpy array containing requested data

    Raises:
        KeyError: If specified variable not found
        OSError: For invalid file paths
    """
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        print("Available datasets:", keys)

        if variable_name in f:
            return np.array(f[variable_name])
        raise KeyError(f"Variable {variable_name} not found")

def cal_psd(data: np.ndarray,
            f_sampling: float = None) -> tuple:
    """
    Computes the Power Spectral Density (PSD) of the input signal using the Fast Fourier Transform (FFT).

    Parameters:
    data (np.ndarray): Input signal data, which can be one-dimensional or multi-dimensional.
    f_sampling (float, optional): Sampling frequency of the signal. If provided, the function returns frequency values.

    Returns:
    tuple: A tuple containing:
        - full_freqs (np.ndarray or None): Frequency axis values if `f_sampling` is provided; otherwise, None.
        - psd (np.ndarray): Computed Power Spectral Density (PSD) of the signal.
    """

    # Compute the FFT along the last axis
    psd = fft(data, axis=-1)

    # Shift zero-frequency component to the center of the spectrum
    psd = fftshift(psd, axes=-1)

    if f_sampling:
        # Compute frequency axis values when sampling frequency is provided
        full_freqs = fftshift(np.fft.fftfreq(data.shape[-1], 1 / f_sampling))

        # Compute the PSD using the squared magnitude of the FFT, normalized by the signal length and sampling frequency
        psd = np.abs(psd) ** 2 / (data.shape[-1] * f_sampling)
    else:
        # Compute the PSD without frequency scaling if sampling frequency is not provided
        psd = np.abs(psd) ** 2 / data.shape[-1]
        full_freqs = None

    # If the input data has multiple dimensions, sum the PSD across all dimensions except the last
    if data.ndim > 1:
        psd = np.sum(psd, axis=0)

    # Estimate the noise floor using the 5th percentile to enhance robustness
    noise_floor = np.percentile(psd, 5)

    # Subtract the estimated noise floor from the PSD
    psd -= noise_floor

    # Ensure the PSD does not contain negative values by clipping it to a minimum of zero
    psd = np.clip(psd, 0, None)

    return full_freqs, psd

def save_structured_txt(data_list, name_list, filename):
    """
    将多个数据列保存为结构化TXT文件

    参数：
    data_list  : 包含多个一维数组的列表，每个数组代表一列数据
    name_list  : 包含每列名称的列表，长度需与data_list一致
    filename   : 要保存的文件名（包含路径）
    """
    # 校验输入参数
    assert len(data_list) == len(name_list), "数据列数与名称数不匹配"
    assert all(len(arr) == len(data_list[0]) for arr in data_list), "各数据列长度不一致"

    # 按列堆叠数据
    data_to_save = np.column_stack(tuple(data_list))

    # 生成文件头（带#号，名称空格分隔）
    header = "# " + " ".join(name_list)

    # 保存文件
    np.savetxt(
        filename,
        data_to_save,
        fmt='%.6e',  # 科学计数法，保留6位小数
        delimiter='    ',  # 4空格分隔符
        header=header,
        comments='',  # 禁用自动添加的注释符
        encoding='utf-8'
    )



