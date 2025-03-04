import matplotlib.pyplot as plt
import numpy as np

from Aegithalos_caudatus import read_pkl, AdaptiveKalmanFilter, save_structured_txt




line_shape = "constant"
dic = read_pkl(f"10MHz_gaussian_{line_shape}_-20_frev_7500000.0_with_coherent.pkl")
q_ref = dic['q_ref']
q_measured = dic['q_measured']
filtered = []
x = np.linspace(0, 350, num=len(q_ref))
akf = AdaptiveKalmanFilter(transition_covariance_Q=0.001*np.eye(2)) # 与传统方法相同的参数
q_list = []

# 并行处理数据流
# for q in q_ref:
for q in q_measured:
    q_list.append(q)
    # Kalman滤波+MAD动态阈值方法
    q = 1-q
    # 传统MAD处理方法（立即处理相同数据点）
    res = akf.predict_update(q)[0]
    filtered.append(res)
# 可视化对比
plt.figure()
plt.plot(1-np.array(q_measured), label="meas")
plt.plot(1-np.array(q_ref), label="ref")
plt.plot(filtered, 'o-', label="filtered", markersize=0.1)
plt.plot(1-np.array(dic['qx']), label="Ground truth")
# plt.plot(dic['failed_to_detect'].astype(int), label="Failed to detect")
plt.xlabel("Time step")
plt.ylabel("Signal value")
plt.title("Comparative Performance: KF with Adaptive MAD vs Traditional MAD")
plt.legend()
plt.show()
save_structured_txt([x, 1-dic['qx'], 1-np.array(dic['q_ref_filtered']), 1-np.array(dic['q_measured_filtered']), 1-np.array(dic['q_predicted']), np.array(dic['weight_ref']), np.array(dic['weight_measured'])],
                    ['time(ms)', 'ground_truth', 'reference_filtered', 'measured_filtered', 'predicted', 'w1_ref', 'w2_measured'],
                    f"fusion_{line_shape}.txt")
