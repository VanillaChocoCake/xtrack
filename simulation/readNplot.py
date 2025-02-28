import matplotlib.pyplot as plt

from Aegithalos_caudatus import *

dic = read_pkl("cos_-20_frev_7500000.0_without_coherent.pkl")
kf = RobustAdaptiveKalmanFilter()
q_ref = dic['q_ref']
q_measured = dic['q_measured']
filtered = []
# for q in q_ref:
#     filtered.append(kf.predict_update(q))
for q in q_measured:
    filtered.append(kf.predict_update(q))
plt.figure()
plt.plot(q_measured, label="original")
plt.plot(filtered, label="filtered")
plt.plot(dic['qx'], label="true")
plt.legend()
plt.show()