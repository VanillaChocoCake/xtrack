import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from scipy.fft import fft, fftshift
from scipy import signal
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

def plot(data):
    plt.figure()
    for d in data:
        plt.plot(d)
    plt.show()

def reshape_with_padding(arr, batch_size):
    # 获取原始数组长度
    n = arr.size
    # 计算不足 batch_size 的部分
    remainder = n % batch_size
    if remainder != 0:
        # 需要补齐的0的个数
        pad_length = batch_size - remainder
        # 使用 np.concatenate 在末尾补0
        arr = np.concatenate([arr, np.zeros(pad_length, dtype=arr.dtype)])
    # 重塑为 (-1, batch_size) 形状，其中 -1 表示自动计算行数
    return arr.reshape(-1, batch_size)


def apply_bandpass_filter(x_data, fs=84e6, lowcut=37e6, highcut=40e6, order=5):
    """
    应用带通滤波器，保留37-40MHz频段信号。

    参数：
        x_data (array): 输入信号数据
        fs (float): 采样率 (Hz)
        lowcut (float): 通带下限频率 (Hz)
        highcut (float): 通带上限频率 (Hz)
        order (int): 滤波器阶数

    返回：
        filtered_data (array): 滤波后的信号
    """
    # 设计Butterworth带通滤波器
    b, a = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs)

    # 应用零相位滤波
    filtered_data = signal.filtfilt(b, a, x_data)

    return filtered_data


def generate_noisy_signal(x_data, snr_db):
    """
    生成满足指定信噪比的高斯噪声，并计算缩放系数a
    :param x_data: 原始信号（一维数组）
    :param snr_db: 目标信噪比（单位：dB）
    :return: (a, noisy_signal) - 缩放系数和加噪后的信号
    """
    # 生成高斯白噪声
    n = np.random.normal(loc=0, scale=2.6, size=len(x_data))

    # 计算信号功率和噪声功率
    p_signal = np.mean(x_data ** 2)  # 原始信号功率
    p_noise = 2.6 ** 2  # 噪声理论功率

    # 将信噪比从dB转换为线性值
    snr_linear = 10 ** (snr_db / 10)

    # 计算缩放系数a
    a = np.sqrt((snr_linear * p_noise) / p_signal)

    # 生成加噪后的信号
    noisy_signal = a * x_data + n

    return noisy_signal

def normalize_to_01(x_data):
    """
    将输入的数据 x_data 正则化到 [0, 1] 范围内。

    参数:
    x_data (list or numpy array): 输入数据，可以是列表或 numpy 数组。

    返回:
    numpy array: 归一化后的数据，范围在 [0, 1] 之间。
    """

    # 转换为 numpy 数组，以便支持广播和数学运算
    x_data = np.array(x_data)

    # 计算最大值和最小值
    x_min = np.min(x_data)
    x_max = np.max(x_data)

    # 防止除以零的错误
    if x_max == x_min:
        return np.zeros_like(x_data)

    # 归一化到 [0, 1] 范围
    normalized_data = (x_data - x_min) / (x_max - x_min)

    return normalized_data


def sgolay_filter(data, window_length, polyorder=4, mode='nearest'):
    """
    Savitzky-Golay滤波器实现

    参数:
    - data: 输入数据（1D数组）
    - window_length: 滤波窗口长度（必须为正奇数）
    - polyorder: 多项式阶数（必须小于window_length）
    - mode: 边界处理模式（'mirror', 'nearest', 'constant', 'interp'等）

    返回:
    - filtered_data: 滤波后的数据
    """
    # 参数校验
    if window_length % 2 == 0:
        raise ValueError("window_length必须是奇数")
    if polyorder >= window_length:
        raise ValueError("polyorder必须小于window_length")

    return savgol_filter(data, window_length, polyorder, mode=mode)


import numpy as np


def noise_reduction_gate(x_data, f_rev, f_sampling):
    """
    处理横向肖特基信号数据，保留每个回旋周期前50%的数据

    参数：
    x_data : numpy.ndarray - 输入信号数组
    f_rev : float - 回旋频率 (Hz)
    f_sampling : float - 采样频率 (必须是f_rev的偶数倍)

    返回：
    processed : numpy.ndarray - 处理后的信号数组，未保留部分置零
    """
    # 计算每个周期的采样点数
    samples_per_cycle = int(f_sampling / f_rev)

    # 验证采样率是否为回旋频率的偶数倍
    if samples_per_cycle % 2 != 0:
        raise ValueError("f_sampling必须是f_rev的偶数倍")

    # 初始化输出数组
    processed = np.zeros_like(x_data)

    # 计算每个周期需要保留的点数
    keep_points = samples_per_cycle // 2

    # 分块处理数据
    total_samples = len(x_data)
    num_full_cycles = total_samples // samples_per_cycle

    # 处理完整周期
    for i in range(num_full_cycles):
        start = i * samples_per_cycle
        end = start + samples_per_cycle
        processed[start:start + keep_points] = x_data[start:start + keep_points]

    # 处理剩余样本
    remaining_samples = total_samples % samples_per_cycle
    if remaining_samples > 0:
        start = num_full_cycles * samples_per_cycle
        keep_remaining = min(keep_points, remaining_samples)
        processed[start:start + keep_remaining] = x_data[start:start + keep_remaining]

    return processed


import numpy as np


def fold_spectrum(spectrum, f_rev, f_sampling):
    """
    将宽频PSD频谱折叠到0~f_rev基带

    参数：
    spectrum : ndarray
        经过fftshift的功率谱密度数组，频率范围(-f_sampling/2, f_sampling/2)
    f_rev : float
        回旋频率（基带频率）
    f_sampling : float
        采样率，必须是f_rev的偶数倍

    返回：
    base_freqs : ndarray
        基带频率分箱（0 ~ f_rev）
    folded_psd : ndarray
        折叠后的功率谱密度
    """
    # 验证输入条件
    assert f_sampling % (2 * f_rev) == 0, "f_sampling必须是f_rev的偶数倍"

    n = len(spectrum)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1 / f_sampling))  # 生成频率轴

    # 计算基本参数
    bands_per_side = int(2*f_sampling / (2 * f_rev))  # 单边半频带数
    bin_per_band = n // (2 * bands_per_side)  # 每个频带的分箱数
    base_mask = (freqs >= 0) & (freqs < f_rev/2)  # 基带分箱掩码
    base_bins = np.where(base_mask)[0]  # 基带分箱索引

    # 初始化输出数组
    folded_psd = []

    # 遍历所有频带（正负）
    for band_idx in range(bands_per_side + 1):
        if band_idx != bands_per_side - 1:
            continue  # 跳过基带本身
        # if band_idx == 0:
        #     continue  # 跳过基带本身
        # 计算当前频带的频率范围
        f_start = band_idx * f_rev/2
        f_end = (band_idx + 1) * f_rev/2

        # 获取当前频带的分箱索引
        band_mask = (freqs >= f_start) & (freqs < f_end)
        band_bins = np.where(band_mask)[0]

        # 跳过空频带（边界情况）
        if len(band_bins) == 0:
            continue

        # 提取频带数据
        band_data = spectrum[band_bins]

        # 判断奇偶性并处理
        if abs(band_idx) % 2 == 1:  # 奇数频带需要反转
            band_data = band_data[::-1]  # 镜像翻转

        # # 计算目标索引（基带内的对应位置）
        # target_slice = slice(0, len(band_data))
        #
        # # 叠加到基带
        # folded_psd[target_slice] += band_data
        folded_psd.append(band_data)

    # 生成基带频率轴
    max_len = max(len(band_data) for band_data in folded_psd)
    padded_arrays = [np.pad(band_data, (0, max_len - len(band_data)), mode='constant') for band_data in folded_psd]
    folded_psd = np.sum(padded_arrays, axis=0)
    base_freqs = np.linspace(0, f_rev / 2, len(folded_psd), endpoint=False)
    return base_freqs, folded_psd


def gaussian_filter(x_data, window_size):
    """
    对一维信号进行高斯滤波
    :param x_data: 输入信号（list或np.array）
    :param window_size: 滤波窗口大小（必须为奇数）
    :return: 滤波后信号（np.array）
    """
    # 输入校验
    if window_size < 3:
        raise ValueError("窗口大小必须≥3")
    if window_size % 2 == 0:
        window_size += 1
        print(f"警告：窗口大小自动调整为奇数 {window_size}")

    # 转换为numpy数组
    x = np.asarray(x_data, dtype=np.float64)

    # 计算高斯核参数
    truncate = 3.0  # 覆盖99.7%能量
    radius = (window_size - 1) // 2
    sigma = radius / truncate

    # 执行滤波（边界处理模式可调整）
    return gaussian_filter1d(x, sigma=sigma, truncate=truncate, mode='nearest')


import numpy as np


def find_local_maxima(x_data, include_edges=True, plateau_detection=False):
    """
    生成一维数据的局部极大值布尔掩码
    :param x_data: 输入一维数据（支持list/np.array）
    :param include_edges: 是否包含边界点（默认为True）
    :param plateau_detection: 是否检测平台极大值（默认为False）
    :return: 布尔掩码数组（True表示对应位置是极大值）
    """
    x = np.asarray(x_data)
    n = len(x)
    mask = np.zeros(n, dtype=bool)

    # 处理短数据情况
    if n < 3:
        if n == 1:
            mask[0] = include_edges
        elif n == 2:
            if include_edges:
                mask[np.argmax(x)] = True
        return mask.tolist()

    # 核心极大值检测逻辑
    if plateau_detection:
        left = x[:-2]
        center = x[1:-1]
        right = x[2:]

        strict_max = (center > left) & (center > right)
        plateau_start = (center >= left) & (center > right)
        plateau_end = (center > left) & (center >= right)
        maxima_mid = strict_max | plateau_start | plateau_end
    else:
        maxima_mid = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])

    mask[1:-1] = maxima_mid

    # 处理边界点
    if include_edges:
        if x[0] > x[1]:
            mask[0] = True
        if x[-1] > x[-2]:
            mask[-1] = True

    return mask.tolist()

from collections import deque

class q_queue:
    def __init__(self, max_len):
        self.max_len = max_len
        self.q_queue = deque(maxlen=max_len)
        self.confidence_queue = deque(maxlen=max_len)
        self.weight_decay = [i/(max_len + 2) for i in range(1, max_len+1)]
        self.first_append = True
    def append(self, q, confidence):
        if self.first_append:
            self.q_queue.extend(np.zeros(self.max_len))
            self.confidence_queue.extend(np.zeros(self.max_len))
            self.first_append = False
        self.q_queue.append(q)
        self.confidence_queue.append(confidence)
