import matplotlib.pyplot as plt
import numpy as np

from myfunc import *

f_sampling = 29.75e6

noise = load_matlab_v73("noise.mat", "noise")
noise = noise.flatten()
t = np.linspace(0, len(noise)/f_sampling, len(noise))
freqs, psd = cal_psd_normal(noise ,f_sampling)
mask = (freqs >= 12.5e6) & (freqs <= 13e6)
freqs = freqs[mask]
psd_new = psd
psd = psd[mask]
while np.sum(np.array(psd_new[mask] >= 3e-6, dtype=int)):
    periodic_noise_freq = freqs[np.argmax(psd_new[mask])]
    y = exclude_coherent_signal(t, noise, periodic_noise_freq, True)
    _, psd_new = cal_psd_normal(y, f_sampling)
psd_new = psd_new[mask]
plt.figure()
plt.plot(freqs/1e6, psd, label="original")
plt.plot(freqs/1e6, psd_new, label="new")
plt.legend()
plt.show()

# 创建结构化数组用于保存数据
data_to_save = np.column_stack((
    freqs/1e6, psd, psd_new
))

# 生成带列标签的文件头
header = (
    "# frequency    original_psd    without_coherent_signal_psd"
)

# 保存到文本文件（科学计数法格式）
np.savetxt(
    f"rmsfit.txt",
    data_to_save,
    fmt='%.6e',  # 控制精度为6位小数
    delimiter='    ',  # 使用4空格分隔列
    header=header,
    comments=''  # 移除自动添加的注释符
)