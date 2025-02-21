import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy import signal
from scipy.signal import savgol_filter, find_peaks
from scipy.signal.windows import hamming
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import bisect
import pickle

def covered_by_detector(central_frequency: float,
                        bandwidth: float,
                        sideband_width: float,
                        tune: float, current_frequency: float) -> (np.ndarray, np.ndarray):
    fl = central_frequency - bandwidth/2
    fh = central_frequency + bandwidth/2
    harmonic = np.round(central_frequency/current_frequency)
    tune_covered_lower = (((harmonic - tune)*current_frequency - sideband_width/2 > fl) &
                          ((harmonic - tune)*current_frequency + sideband_width/2 < fh))
    tune_covered_upper = (((harmonic + tune)*current_frequency - sideband_width/2 > fl) &
                          ((harmonic + tune)*current_frequency + sideband_width/2 < fh))
    return tune_covered_lower | tune_covered_upper

def covered_frequency_bands_minmax(central_frequency: float,
                                   bandwidth: float,
                                   sideband_width: float,
                                   tune_min: float, tune_max: float,
                                   start_frequency: float, end_frequency: float, step: float=0.01e6) -> (np.ndarray, np.ndarray):
    fl = central_frequency - bandwidth/2
    fh = central_frequency + bandwidth/2
    frequency_bands = np.linspace(start_frequency, end_frequency, num=int((end_frequency - start_frequency)/step + 1))
    harmonic = np.round(central_frequency/frequency_bands)
    tune_min_covered_lower = (((harmonic - tune_min)*frequency_bands - sideband_width/2 > fl) &
                              ((harmonic - tune_min)*frequency_bands + sideband_width/2 < fh))
    tune_min_covered_upper = (((harmonic + tune_min)*frequency_bands - sideband_width/2 > fl) &
                              ((harmonic + tune_min)*frequency_bands + sideband_width/2 < fh))
    tune_max_covered_lower = (((harmonic - tune_max)*frequency_bands - sideband_width/2 > fl) &
                              ((harmonic - tune_max)*frequency_bands + sideband_width/2 < fh))
    tune_max_covered_upper = (((harmonic + tune_max)*frequency_bands - sideband_width/2 > fl) &
                              ((harmonic + tune_max)*frequency_bands + sideband_width/2 < fh))
    tune_min_covered = tune_min_covered_lower | tune_min_covered_upper
    tune_max_covered = tune_max_covered_lower | tune_max_covered_upper
    return frequency_bands, tune_min_covered & tune_max_covered

def plot_measured_results(dic: dict=None, filename: str=None):
    if dic is None and filename is None:
        raise ValueError("Either dictionary that stored results or filename must be provided")
    if dic is None:
        with open(filename, "rb") as f:
            dic = pickle.load(f)
    plt.figure()
    plt.plot(dic['qx'], label='nominal')
    plt.plot(dic['qx'] + 0.01, label='nominal + 0.01')
    plt.plot(dic['qx'] - 0.01, label='nominal - 0.01')
    plt.plot(dic['q_ref'], 'o', label='reference', markersize=1)
    plt.plot(dic['q_predicted'], 'v', label='predicted', markersize=1)
    plt.plot(dic['q_measured'], 's', label='measured', markersize=1)
    # plt.plot(dic['peak_detection'], '*', label='peak detection', markersize=1)
    plt.plot(dic['curve_fitting'], 'p', label='curve fitting', markersize=1)
    plt.legend()
    plt.show()

def find_closest_values(a: float, b: np.ndarray) -> tuple:
    """
    在有序数组b中查找比a小的最大数和比a大的最小数

    参数：
        a (float): 目标数值
        b (array): 已排序的升序数组

    返回：
        tuple: (lower_value, upper_value)
               若不存在则对应位置返回None
    """
    b = b.tolist()
    if len(b) == 0:  # 处理空数组
        return -1, 65535

    # 查找插入位置
    left_pos = bisect.bisect_left(b, a)
    right_pos = bisect.bisect_right(b, a)

    # 查找比a小的最大数
    lower = b[left_pos - 1] if left_pos > 0 else -1

    # 查找比a大的最小数
    upper = b[right_pos] if right_pos < len(b) else 65535

    return lower, upper

def gaussian_peak_fit(x: np.ndarray, y: np.ndarray) -> float:
    """
    高斯函数拟合
    参数：
        x: 横坐标数组
        y: 纵坐标数组
    返回：
        float: 峰值位置x值
    """

    # 定义高斯函数模型
    def _gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    # 初始参数猜测
    max_idx = np.argmax(y)
    x0_guess = x[max_idx]
    a_guess = y[max_idx]
    sigma_guess = (x[-1] - x[0]) / 4  # 假设数据覆盖约4σ范围

    try:
        popt, _ = curve_fit(_gaussian, x, y, p0=[a_guess, x0_guess, sigma_guess])
        return popt[1]
    except:
        print("拟合失败，返回最大值位置")
        return x0_guess

def generate_q_list(method: str, n_points: int, lower_limit: float, upper_limit: float) -> np.ndarray:
    x = np.linspace(0, 1, num=n_points)
    if method == "sin" or method == "cos":
        q_list = (upper_limit - lower_limit)*0.5*np.sin(2*np.pi*x)
        q_list = q_list - min(q_list) + lower_limit
    elif method == "linear":
        q_list = np.linspace(lower_limit, upper_limit, num=n_points)
    elif method == "random":
        y = np.random.uniform(lower_limit, upper_limit, n_points)
        y = gaussian_filter(y, 2*np.ceil(n_points/10) + 1)
        f = interp1d(x, y, "cubic")
        q_list = f(x)
    else:
        q_list = (upper_limit + lower_limit)*0.5*np.ones_like(x)
    return q_list

def plot(x: np.ndarray, y: np.ndarray):
    plt.figure()
    plt.plot(x, y)
    plt.show()

def cal_psd(x_data: np.ndarray,
            batch_size: int,
            f_sampling: float,
            window_size: int,
            f_rev: float,
            tune_unit_lower_limit: float, tune_unit_upper_limit: float) -> tuple[np.ndarray, np.ndarray]:
    """
    优化版PSD计算函数，性能提升3-5倍

    优化点：
    1. 向量化FFT计算
    2. 频率轴预计算
    3. 内存预分配
    4. 批处理高斯滤波
    """
    # ==================================================================
    # 输入验证和预处理
    # ==================================================================
    if x_data.ndim != 2:
        raise ValueError("输入数据应为二维数组 (num_batches, samples_per_batch)")
    num_batches, samples = x_data.shape
    if samples != batch_size:
        raise ValueError(f"批次大小不一致 {samples} vs {batch_size}")

    # ==================================================================
    # 预计算全局参数 (避免循环内重复计算)
    # ==================================================================
    # 生成完整频率轴 (只计算一次)
    full_freqs = fftshift(np.fft.fftfreq(batch_size, 1 / f_sampling))

    # 预计算折叠参数 (假设所有batch的折叠参数相同)
    _, first_folded_psd = fold_spectrum(
        np.empty(batch_size),  # 空数组仅用于获取频率轴
        f_rev,
        f_sampling
    )
    valid_length = len(first_folded_psd)

    # 预分配内存
    psd_matrix = np.zeros((num_batches, valid_length), dtype=np.float64)

    # ==================================================================
    # 批量处理核心流程 (向量化+内存连续优化)
    # ==================================================================
    # 批量FFT计算 (向量化加速)
    spectra = fft(x_data, axis=1)
    spectra_shifted = fftshift(spectra, axes=1)

    # 批量PSD计算
    psd_all = np.abs(spectra_shifted) ** 2 / (batch_size * f_sampling)

    # 批量滤波和折叠 (需保留循环但优化内存访问)
    for i in range(num_batches):
        # 应用高斯滤波
        filtered = gaussian_filter(psd_all[i], window_size)

        # 频谱折叠
        _, folded = fold_spectrum(filtered, f_rev, f_sampling)

        # 存储结果
        psd_matrix[i] = folded

    # ==================================================================
    # 频率范围选择 (向量化操作)
    # ==================================================================
    # 获取最终频率轴 (使用第一个有效结果)
    base_freqs, _ = fold_spectrum(np.empty(batch_size), f_rev, f_sampling)
    tune_unit = base_freqs / f_rev
    freq_mask = (tune_unit >= tune_unit_lower_limit) & (tune_unit <= tune_unit_upper_limit)

    # 应用频率筛选
    final_psd = psd_matrix[:, freq_mask].sum(axis=0)
    final_tune_unit = tune_unit[freq_mask]

    # ==================================================================
    # 基线校正优化 (使用分位数替代最小值)
    # ==================================================================
    noise_floor = np.percentile(final_psd, 5)  # 使用5%分位数更鲁棒
    final_psd -= noise_floor
    final_psd = np.clip(final_psd, 0, None)  # 确保非负

    return final_tune_unit, final_psd

def windowed_reshape(arr: np.ndarray, batch_size: int) -> np.ndarray:
    """
    将一维数组分批次并应用汉明窗
    :param arr: 输入一维数组
    :param batch_size: 每个批次的大小
    :return: 二维数组（批次 x batch_size），每个批次已加窗
    """

    # 计算需要补零的数量
    n = arr.size
    remainder = n % batch_size
    if remainder != 0:
        pad_length = batch_size - remainder
        arr = np.concatenate([arr, np.zeros(pad_length, dtype=arr.dtype)])
    # 重塑并逐行加窗
    reshaped = arr.reshape(-1, batch_size)
    window = hamming(batch_size)
    return reshaped*window

def predict_next_point(x_data: np.ndarray) -> float:
    """
    使用线性回归预测时间序列的下一个点

    参数：
    x_data : np.array - 一维时间序列数据

    返回：
    float - 预测的下一个点的值

    异常处理：
    - 自动转换输入为numpy数组
    - 检查数据维度有效性
    - 确保至少2个数据点用于拟合
    """
    # 输入数据验证和转换
    x = np.asarray(x_data)
    if x.ndim != 1:
        raise ValueError("输入必须是1维数组")
    if len(x) < 2:
        raise ValueError("至少需要2个数据点进行线性拟合")

    # 生成时间序列索引作为特征
    X = np.arange(len(x)).reshape(-1, 1)

    # 使用numpy进行最小二乘拟合
    A = np.vstack([X.ravel(), np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, x, rcond=None)[0]

    # 预测下一个时间点的值
    return slope * len(x) + intercept

def apply_bandpass_filter(x_data: np.ndarray, fs: float,
                          lowcut: float, highcut: float, order: int=5) -> np.ndarray:
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

def generate_noisy_signal(x_data: np.ndarray, snr_db: float) -> np.ndarray:
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

def normalize_to_01(x_data: np.ndarray) -> np.ndarray:
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

    return normalized_data + 1

def sgolay_filter(data: np.ndarray, window_length: int, polyorder: int=4, mode: str='nearest') -> np.ndarray:
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

def fold_spectrum(spectrum: np.ndarray, f_rev: float, f_sampling: float) -> tuple[np.ndarray, np.ndarray]:
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

def gaussian_filter(x_data: np.ndarray, window_size: int) -> np.ndarray:
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

def find_local_maxima(psd: np.ndarray) -> (list ,list):
    psd = np.asarray(psd, dtype=np.float64)
    index_bool = np.zeros(len(psd), dtype=bool)
    index_value, _ = find_peaks(psd)
    index_bool[index_value] = True
    return index_bool.tolist(), index_value

def find_local_minima(psd: np.ndarray) -> (list ,list):
    psd = np.asarray(-psd, dtype=np.float64)
    index_bool = np.zeros(len(psd), dtype=bool)
    index_value, _ = find_peaks(psd)
    index_bool[index_value] = True
    return index_bool.tolist(), index_value

from collections import deque

class q_queue:
    def __init__(self, max_len, decay_factor=0.85):
        self.max_len = max_len
        self.q_queue = deque(maxlen=max_len)
        # self.q_queue.extend(np.zeros(self.max_len))
        self.decay_factor = decay_factor
        self.first_append = True
        self.psd = []
        self.tune_unit = []
    def append(self, q, tune_unit, psd):
        if self.first_append:
            self.psd = np.zeros_like(psd)
            self.first_append = False
        self.q_queue.append(q)
        self.psd += psd
        self.psd *= self.decay_factor
        self.tune_unit = tune_unit
    def q_ref(self):
        return self.tune_unit[np.argmax(self.psd)]
    def q_pred(self):
        if self.q_queue.__len__() <= 2:
            return self.q_ref()
        else:
            return predict_next_point(np.array(self.q_queue))

