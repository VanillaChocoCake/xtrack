import matplotlib.pyplot as plt

from Aegithalos_caudatus import *

dic = read_pkl("linear_-10_frev_7500000.0_without_coherent.pkl")
kf = RobustAdaptiveKalmanFilter()
q_ref_original = dic['q_ref_original']
q_ref_filtered = dic['q_ref_filtered']
q_measured_original = dic['q_measured_original']
q_measured_filtered = dic['q_measured_filtered']
filtered = []
for q in q_ref_original:
    filtered.append(kf.predict_update(q))
# for q in q_measured:
#     filtered.append(kf.predict_update(q))
plt.figure()
plt.plot(q_ref_original, label="original")
# plt.plot(q_measured, label="original")
plt.plot(filtered, label="filtered")
plt.plot(dic['qx'], label="true")
plt.legend()
plt.show()