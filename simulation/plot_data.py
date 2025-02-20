import pickle
import matplotlib.pyplot as plt
with open(f"sin_sum.pkl", "rb") as f:
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