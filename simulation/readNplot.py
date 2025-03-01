import matplotlib.pyplot as plt
import numpy as np

from Aegithalos_caudatus import *


dic = read_pkl("10MHz_gaussian_sin_-20_frev_7500000.0_without_coherent.pkl")
q_ref_original = dic['q_ref_original']
q_ref_filtered = dic['q_ref_filtered']
q_measured_original = dic['q_measured_original']
q_measured_filtered = dic['q_measured_filtered']
filtered = []
# 初始化滤波器
window_size = 10
traditional_mad = MADFilter(window_size=window_size)  # 与传统方法相同的参数

filtered_traditional_mad = []
q_list = []

# 并行处理数据流
# for q in q_ref_original:
for q in q_measured_original:
    q_list.append(q)
    # Kalman滤波+MAD动态阈值方法

    # 传统MAD处理方法（立即处理相同数据点）
    mad_result = traditional_mad.process(q)
    filtered_traditional_mad.append(mad_result)
# 可视化对比
plt.figure()
plt.plot(q_measured_original, alpha=0.5, label="Raw measurements")
# plt.plot(q_ref_original, alpha=0.5, label="Raw measurements")
plt.plot(filtered_traditional_mad, label="Traditional MAD filter", linewidth=1.2)
plt.plot(dic['qx'], label="Ground truth", alpha=0.8)
plt.xlabel("Time step")
plt.ylabel("Signal value")
plt.title("Comparative Performance: KF with Adaptive MAD vs Traditional MAD")
plt.legend()
plt.show()
