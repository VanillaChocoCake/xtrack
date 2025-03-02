import matplotlib.pyplot as plt
import numpy as np

from Aegithalos_caudatus import read_pkl, AdaptiveKalmanFilter




dic = read_pkl("3MHz_gaussian_constant_-20_frev_ramping_with_coherent.pkl")
q_ref = dic['q_ref']
q_measured = dic['q_measured']
filtered = []
# 初始化滤波器
window_size = 10
akf = AdaptiveKalmanFilter(transition_covariance_Q=0.01*np.eye(2)) # 与传统方法相同的参数
q_list = []

# 并行处理数据流
# for q in q_ref:
for q in q_measured:
    q_list.append(q)
    # Kalman滤波+MAD动态阈值方法

    # 传统MAD处理方法（立即处理相同数据点）
    res = akf.predict_update(q)[0]
    filtered.append(res)
# 可视化对比
plt.figure()
plt.plot(q_measured, label="meas")
plt.plot(q_ref, label="ref")
plt.plot(filtered, 'o-', label="filtered", markersize=0.1)
plt.plot(dic['qx'], label="Ground truth")
plt.plot(dic['failed_to_detect'].astype(int), label="Failed to detect")
plt.xlabel("Time step")
plt.ylabel("Signal value")
plt.title("Comparative Performance: KF with Adaptive MAD vs Traditional MAD")
plt.legend()
plt.show()
print(sum(abs(np.array(filtered[50:])-dic['qx'][50:])))
