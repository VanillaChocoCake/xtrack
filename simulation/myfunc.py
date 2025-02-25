import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy import signal
from scipy.signal import savgol_filter, find_peaks
from scipy.signal.windows import hamming
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import bisect
import pickle
from scipy.interpolate import interp1d, CubicSpline, lagrange, UnivariateSpline

def fix_anomaly(data: list, value: float, max_len: int, outliers_threshold_coef) -> float:
    if len(data) >= max_len:
        diffs = np.diff(data)
        avg_diff = np.mean(np.abs(diffs))
        if np.abs(value - data[-1]) >= outliers_threshold_coef * avg_diff:
            value = data[-1] + np.sign(diffs[-1]) * avg_diff
    return value

def interpolation(data: np.ndarray, interpolate_method: str="cubic", interpolate_coef: float=1) -> np.ndarray:
    batch_size = len(data)
    if interpolate_method is not None:
        freq = np.linspace(0, batch_size - 1, num=batch_size)
        freq_new = np.linspace(0, batch_size - 1, num=int(interpolate_coef * batch_size))
        if interpolate_method == "linear":
            f = interp1d(freq, data)
        elif interpolate_method == "cubic":
            f = CubicSpline(freq, data)
        elif interpolate_method == "lagrange":
            f = lagrange(freq, data)
        elif interpolate_method == "univariate":
            f = UnivariateSpline(freq, data, s=np.mean(data))
        else:
            raise ValueError(f"Unknown interpolation method: {interpolate_method}")
        psd = f(freq_new)
    else:
        psd = data
    return psd

def exclude_coherent_spectrum(tune_unit: np.ndarray, spectrum: np.ndarray, side_point_num: int, threshold: float=0.4) -> np.ndarray:
    # 找到与 qx 最接近的两个值
    peak_idx = np.argmax(spectrum)
    peak_tune = tune_unit[peak_idx]
    left, right = find_closest_values(peak_tune, tune_unit)
    coherent_peak = left if np.abs(peak_tune - left) < np.abs(peak_tune - right) else right
    coherent_peak_index = np.where(tune_unit == coherent_peak)[0][0]
    # # 计算排除的范围
    # exclude_range = max(1, side_point_num//8)
    # start = int(max(0, coherent_peak_index - exclude_range))
    # stop = int(min(len(spectrum) - 1, coherent_peak_index + exclude_range))  # 修正索引范围
    coef = 1/np.max(spectrum)
    spectrum *= coef
    start_left = int(max(0, coherent_peak_index - side_point_num))
    stop_right = int(min(len(spectrum) - 1, coherent_peak_index + side_point_num))
    # fit_mask = ((tune_unit >= tune_unit[start_left]) & (tune_unit <= tune_unit[stop_right])) & ((tune_unit < tune_unit[start]) | (tune_unit > tune_unit[stop]))
    fit_mask = (spectrum <= threshold) & ((tune_unit >= tune_unit[start_left]) & (tune_unit <= tune_unit[stop_right]))
    fit_tune_unit = tune_unit[fit_mask]
    fit_spectrum = spectrum[fit_mask]
    a, x0, phi = gaussian_peak_fit(fit_tune_unit, fit_spectrum)
    scaled_spectrum = spectrum.copy()
    gaussian_fit = gaussian_function(tune_unit, a, x0, phi)
    new_mask = (spectrum > threshold) & ((tune_unit >= tune_unit[start_left]) & (tune_unit <= tune_unit[stop_right]))
    spectrum *= (spectrum <= threshold).astype(np.float64)
    spectrum += gaussian_fit*new_mask.astype(np.float64)
    # plt.figure()
    # plt.plot(tune_unit, scaled_spectrum, label="original")
    # plt.plot(tune_unit, spectrum, label="replaced")
    # plt.plot(tune_unit, gaussian_fit, label="fitted")
    # plt.legend()
    # plt.show()
    spectrum /= coef
    return spectrum

def exclude_coherent_signal(t: np.ndarray, y: np.ndarray, frequency: float, exclude_coherent:bool=False) -> np.ndarray:
    def harmonic_model(t, A, f, phi):
        return A * np.sin(2 * np.pi * f * t + phi)
    if exclude_coherent:
        p0 = [max(y), frequency, 0]
        params, _ = curve_fit(harmonic_model, xdata=t, ydata=y, p0=p0)
        A_fit, f_fit, phi_fit = params
        fitted_harmonic = harmonic_model(t, A_fit, f_fit, phi_fit)
        residual_signal = y - fitted_harmonic
        return residual_signal
    else:
        return y

def covered_by_detector(central_frequency: float,
                        bandwidth: float,
                        sideband_width: float,
                        tune: float, current_frequency: float) -> bool:
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
    qx = np.array(dic['qx'])
    failed_to_detect = np.array(dic['failed_to_detect'])
    q_ref = np.array(dic['q_ref'])
    q_predicted = np.array(dic['q_predicted'])
    q_measured = np.array(dic['q_measured'])
    peak_detection = np.array(dic['peak_detection'])
    cf = np.array(dic['curve_fitting'])
    weight_ref = np.array(dic['weight_ref'])
    weight_measured = np.array(dic['weight_measured'])
    plt.figure()
    plt.plot(qx, label='nominal')
    plt.plot(qx + 0.01, label='nominal + 0.01')
    plt.plot(qx - 0.01, label='nominal - 0.01')
    plt.plot(failed_to_detect, label='not covered area')
    plt.plot(q_ref, 'o', label='reference', markersize=1)
    plt.plot(q_predicted, 'v', label='predicted', markersize=1)
    plt.plot(q_measured, 's', label='measured', markersize=1)
    # plt.plot(peak_detection, '*', label='peak detection', markersize=1)
    plt.plot(cf, 'p', label='curve fitting', markersize=1)
    # plt.plot(weight_ref, label='weight_ref')
    # plt.plot(weight_measured, label='weight_measured')
    plt.title("Comparison")
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

def gaussian_function(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gaussian_peak_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    高斯函数拟合
    参数：
        x: 横坐标数组
        y: 纵坐标数组
    返回：
        float: 峰值位置x值
    """
    

    # 初始参数猜测
    max_idx = np.argmax(y)
    x0_guess = x[max_idx]
    a_guess = y[max_idx]
    sigma_guess = (x[-1] - x[0]) / 2  # 假设数据覆盖约4σ范围

    try:
        popt, _ = curve_fit(gaussian_function, x, y, p0=[a_guess, x0_guess, sigma_guess])
        return popt
    except:
        print("拟合失败，返回最大值位置")
        return np.array([a_guess, x0_guess, sigma_guess])

def generate_q_list(method: str, n_points: int, lower_limit: float, upper_limit: float) -> np.ndarray:
    x = np.linspace(0, 1, num=n_points)
    if method == "sin":
        q_list = (upper_limit - lower_limit)*0.5*np.sin(2*np.pi*x)
        q_list = q_list - min(q_list) + lower_limit
    elif method == "cos":
        q_list = (upper_limit - lower_limit)*0.5*np.cos(2*np.pi*x)
        q_list = q_list - min(q_list) + lower_limit
    elif method == "linear":
        q_list = np.linspace(lower_limit, upper_limit, num=n_points)
    elif method == "random":
        from scipy.interpolate import UnivariateSpline
        y = np.random.uniform(lower_limit, upper_limit, n_points)
        f = UnivariateSpline(x, y, s=5*(upper_limit + lower_limit)/2)
        y = f(x)
        y -= min(y)
        y = y/max(y)*(upper_limit - lower_limit) + lower_limit
        q_list = y
    else:
        q_list = (upper_limit + lower_limit)*0.5*np.ones_like(x)
    return q_list

def plot(x: np.ndarray, y: np.ndarray=None):
    plt.figure()
    if y is None:
        plt.plot(x)
    else:
        plt.plot(x, y)
    plt.show()

def cal_psd_normal(data: np.ndarray, f_sampling: float) -> tuple:
    full_freqs = fftshift(np.fft.fftfreq(len(data), 1/f_sampling))
    psd = fft(data)
    psd = fftshift(psd)
    psd = np.abs(psd) ** 2 / (len(psd) * f_sampling)
    noise_floor = np.percentile(psd, 5)  # 使用5%分位数更鲁棒
    psd -= noise_floor
    psd = np.clip(psd, 0, None)
    return full_freqs, psd

def cal_psd(x_data: np.ndarray, noise: np.ndarray,
            batch_size: int,
            f_sampling: float,
            window_size: int,
            f_rev: float,
            tune_unit_lower_limit: float, tune_unit_upper_limit: float,
            interpolate_method: str="cubic", interpolate_coef: float=1,
            snr: float=-20, side_point_num: int=None, exclude_coherent:bool=False,
            procedure: int=1) -> tuple:
    """
    procedure=1: fold -> sum -> filter
    procedure=2: fold -> filter -> sum
    """
    # ==================================================================
    # 输入验证和预处理
    # ==================================================================
    if x_data.ndim != 2:
        raise ValueError("输入数据应为二维数组 (num_batches, samples_per_batch)")
    num_batches, samples = x_data.shape
    if samples != batch_size:
        raise ValueError(f"批次大小不一致 {samples} vs {batch_size}")
    if interpolate_method is None:
        interpolate_coef = 1

    # ==================================================================
    # 预计算全局参数 (避免循环内重复计算)
    # ==================================================================
    # 生成完整频率轴 (只计算一次)
    base_freqs, _ = fold_spectrum(np.empty(batch_size), f_rev, f_sampling)
    tune_unit = base_freqs/f_rev
    freq_mask = (tune_unit >= tune_unit_lower_limit) & (tune_unit <= tune_unit_upper_limit)
    final_tune_unit = tune_unit[freq_mask]
    # 预分配内存
    psd_matrix = np.zeros((num_batches, len(tune_unit)), dtype=np.float64)

    # ==================================================================
    # 批量处理核心流程 (向量化+内存连续优化)
    # ==================================================================
    # 批量FFT计算 (向量化加速)
    spectra = fft(x_data, axis=1)
    spectra_shifted = fftshift(spectra, axes=1)

    # 批量PSD计算
    psd_all = np.abs(spectra_shifted) ** 2 / (batch_size * f_sampling)

    if exclude_coherent:
        spectra_noise = fft(noise, axis=1)
        spectra_noise_shifted = fftshift(spectra_noise, axes=1)
        psd_noise_all = np.abs(spectra_noise_shifted) ** 2 / (batch_size * f_sampling)
        # 批量滤波和折叠 (需保留循环但优化内存访问)
        for i in range(num_batches):
            # raise ValueError("You haven't debugged this part yet!")
            psd = psd_all[i]
            # psd = interpolation(psd, interpolate_method, interpolate_coef)
            _, folded = fold_spectrum(psd, f_rev, f_sampling)
            folded = exclude_coherent_spectrum(tune_unit, folded, side_point_num)
            psd_noise = psd_noise_all[i, 0: len(folded)]
            snr_linear = 10**(snr/10)
            P_noise_target = np.sum(folded)/snr_linear
            P_noise = np.sum(psd_noise)
            psd_noise *= (P_noise_target/P_noise)
            psd_matrix[i] = folded + psd_noise
    else:
        # 批量滤波和折叠 (需保留循环但优化内存访问)
        for i in range(num_batches):
            psd = psd_all[i]
            _, folded = fold_spectrum(psd, f_rev, f_sampling)
            if procedure == 1:
                # fold -> sum -> filter
                psd_matrix[i] = folded
            elif procedure == 2:
                # fold -> filter -> sum
                # filtered = gaussian_filter(folded, window_size)
                filtered = savgol_filter(psd, window_size, 5)
                filtered = interpolation(filtered, "univariate", 1.0)
                psd_matrix[i] = filtered

    if procedure == 1:
        # fold -> sum -> filter
        final_psd = psd_matrix.sum(axis=0)
        final_psd = gaussian_filter(final_psd, window_size)
        # final_psd = savgol_filter(final_psd, window_size, 5)
        # final_psd = interpolation(final_psd, "univariate", 1.0)
        final_psd = final_psd[freq_mask]
    elif procedure == 2:
        # fold -> filter -> sum
        final_psd = psd_matrix[:, freq_mask].sum(axis=0)
    final_psd /= np.min(final_psd)
    final_psd = interpolation(final_psd, interpolate_method, interpolate_coef)
    final_tune_unit = np.linspace(final_tune_unit[0], final_tune_unit[-1], num=len(final_psd))

    # final_psd = gaussian_filter(final_psd, window_size)

    # ==================================================================
    # 基线校正优化 (使用分位数替代最小值)
    # ==================================================================
    noise_floor = np.percentile(final_psd, 5)  # 使用5%分位数更鲁棒
    final_psd -= noise_floor
    final_psd = np.clip(final_psd, 0, None)  # 确保非负
    return final_tune_unit, final_psd

def windowed_reshape(arr: np.ndarray, batch_size: int) -> tuple:
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
    return reshaped*window, batch_size

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

def generate_noisy_signal(x_data: np.ndarray, snr_db: float, exclude_coherent:bool=False) -> tuple:
    """
    生成满足指定信噪比的高斯噪声，并计算缩放系数a
    :param x_data: 原始信号（一维数组）
    :param snr_db: 目标信噪比（单位：dB）
    :param exclude_coherent
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
    if exclude_coherent:
        return a * x_data, n
    else:
        return noisy_signal, n

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

def sgolay_filter(data: np.ndarray, window_length: int, polyorder: int=4, mode: str='mirror') -> np.ndarray:
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

def fold_spectrum(spectrum: np.ndarray, f_rev: float, f_sampling: float) -> tuple:
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

    n = len(spectrum) # 生成频率轴
    freqs = np.linspace(-f_sampling / 2, f_sampling / 2, n)
    # 计算基本参数
    bands_per_side = int(2*f_sampling / (2 * f_rev))  # 单边半频带数

    # 初始化输出数组
    folded_psd = []

    # 遍历所有频带
    for band_idx in range(bands_per_side):
        if band_idx != bands_per_side - 1:
            continue
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
        print(f"警告：窗口大小自动调整为奇数：{window_size}")

    # 转换为numpy数组
    x = np.asarray(x_data, dtype=np.float64)

    # 计算高斯核参数
    truncate = 2.0  # 覆盖95%能量
    sigma = (window_size - 1)/4

    # 执行滤波（边界处理模式可调整）
    return gaussian_filter1d(x, sigma=sigma, truncate=truncate, mode='mirror')

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

class AdaptiveKalmanFilter:
    def __init__(self, initial_state=0.5, initial_estimate_error=1, process_noise=0.06, measurement_noise=2.6**2):
        # 初始参数（可随意设置，滤波器会自动调整）
        # 状态量（标量）
        self.x = initial_state

        # 估计误差协方差（标量）
        self.P = initial_estimate_error

        # 过程噪声协方差（标量）
        self.Q = process_noise

        # 测量噪声协方差（标量）
        self.R = measurement_noise
        self.window = []  # 残差窗口

    def predict_update(self, z, alpha=0.2):
        """ 含参数自适应的预测-更新步骤 """
        # 预测阶段
        x_pred = self.x
        P_pred = self.P + self.Q

        # 计算卡尔曼增益
        K = P_pred / (P_pred + self.R)

        # 更新阶段
        residual = z - x_pred
        self.x = x_pred + K * residual
        self.P = (1 - K) * P_pred

        # 记录残差（用于调整参数）
        self.window.append(residual)
        if len(self.window) > 3:  # 滑动窗口大小
            self.window.pop(0)

        # 自适应调整R（基于残差方差）
        if len(self.window) >= 2:
            self.R = alpha * np.var(self.window) + (1 - alpha) * self.R

        # 自适应调整Q（基于残差绝对值）
        self.Q = alpha * abs(residual) + (1 - alpha) * self.Q

        return self.x

class DualDetectorAdaptiveKalmanFilter:
    def __init__(
            self,
            initial_state=0.5,
            initial_estimate_error=1,
            process_noise=0.06,
            measurement_noise=2.6 ** 2,  # 初始测量噪声同时赋给两个探测器
            max_len=8,
            alpha=0.3, alpha_range=(0.1, 0.5)
    ):
        # 状态估计初始化
        self.x = initial_state
        self.P = initial_estimate_error

        # 过程噪声协方差
        self.Q = process_noise

        # 两个探测器的测量噪声协方差
        self.R1 = measurement_noise  # 探测器1的初始噪声
        self.R2 = measurement_noise  # 探测器2的初始噪声

        self.w1 = 114514
        self.w2 = 1919810

        # 残差滑动窗口（每个探测器独立）
        self.window1 = deque(maxlen=max_len)  # 探测器1的残差窗口
        self.window2 = deque(maxlen=max_len)  # 探测器2的残差窗口
        self.residual_history = deque(maxlen=5)

        self.alpha = alpha
        self.alpha_min, self.alpha_max = alpha_range

    def predict_update(self, z1, z2):
        """基于双探测器的自适应预测-更新步骤"""
        # ----------- 预测阶段 -----------
        x_pred = self.x
        P_pred = self.P + self.Q  # 预测协方差

        # ----------- 测量融合 -----------
        # 计算加权综合测量值（噪声小的探测器权重更高）
        total_precision = 1 / self.R1 + 1 / self.R2
        self.w1 = (1 / self.R1) / total_precision  # 权重公式1
        self.w2 = (1 / self.R2) / total_precision  # 权重公式2

        # 加权融合测量值
        z_fused = self.w1 * z1 + self.w2 * z2
        R_fused = 1 / total_precision  # 融合后的等效测量噪声

        # ----------- 更新阶段 -----------
        # 计算卡尔曼增益
        K = P_pred / (P_pred + R_fused)

        # 更新状态估计
        residual_fused = z_fused - x_pred
        self.x = x_pred + K * residual_fused
        self.P = (1 - K) * P_pred

        # ----------- 参数自适应 -----------
        # 计算各探测器的残差（基于预测值）
        # 记录残差历史（用于alpha调整）
        self.residual_history.append(abs(residual_fused))

        # 动态调整alpha（基于最近残差均值）
        if len(self.residual_history) >= 3:
            avg_residual = np.mean(self.residual_history)
            # alpha与残差大小正相关（S型曲线调整）
            self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * \
                         (avg_residual / (avg_residual + 0.5))  # 0.5为平滑系数
        residual1 = z1 - x_pred
        residual2 = z2 - x_pred

        # 更新探测器1的噪声估计
        self.window1.append(residual1)
        if len(self.window1) >= 2:
            var1 = np.var(self.window1)
            self.R1 = self.alpha * var1 + (1 - self.alpha) * self.R1

        # 更新探测器2的噪声估计
        self.window2.append(residual2)
        if len(self.window2) >= 2:
            var2 = np.var(self.window2)
            self.R2 = self.alpha * var2 + (1 - self.alpha) * self.R2

        # 更新过程噪声（基于融合残差）
        self.Q = self.alpha * abs(residual_fused) + (1 - self.alpha) * self.Q

        return self.x

    def detector_weights(self):
        """获取标准化权重（保证和为1）"""
        total = self.w1 + self.w2
        return self.w1 / total, self.w2 / total  # 二次标准化确保精度


class SimpleKalmanFilter:
    def __init__(self, initial_state=0.5, initial_estimate_error=1, process_noise=0.06, measurement_noise=2.6 ** 2):
        """
        一维卡尔曼滤波器初始化
        :param initial_state: 初始状态估计值
        :param initial_estimate_error: 初始估计误差（协方差）
        :param process_noise: 过程噪声方差（Q）
        :param measurement_noise: 测量噪声方差（R）
        """
        # 状态量（标量）
        self.x = initial_state

        # 估计误差协方差（标量）
        self.P = initial_estimate_error

        # 过程噪声协方差（标量）
        self.Q = process_noise

        # 测量噪声协方差（标量）
        self.R = measurement_noise

        # 状态转移系数（标量）
        self.F = 1  # 假设系统为恒定模型

        # 观测系数（标量）
        self.H = 1  # 直接观测状态量

    def predict(self):
        """ 预测阶段 """
        # 状态预测（保持恒定模型）
        self.x = self.F * self.x
        # 协方差预测
        self.P = self.F * self.P * self.F + self.Q
        return self.x

    def update(self, z):
        """ 更新阶段 """
        # 计算卡尔曼增益
        K = self.P * self.H / (self.H * self.P * self.H + self.R)

        # 状态更新
        self.x = self.x + K * (z - self.H * self.x)

        # 协方差更新
        self.P = (1 - K * self.H) * self.P
        return self.x
