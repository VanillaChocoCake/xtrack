import matplotlib.pyplot as plt
import numpy as np

from Aegithalos_caudatus import read_pkl, save_structured_txt, AdaptiveKalmanFilter, DualDetectorKalmanFilter
from Aegithalos_caudatus import AdaptiveSensorFusionKalmanFilter, plot

line_shape = "random"
dic = read_pkl(f"10MHz_gaussian_{line_shape}_-20_frev_ramping_with_coherent.pkl")
q_predicted = np.array(dic['q_predicted'])
qm = np.array(dic['q_measured'])
qr = np.array(dic['acquire_q_ref'])
pd = np.array(dic['peak_detection'])
gt = np.array(dic['qx'])
x = np.linspace(0, 350, num=len(q_predicted))
save_structured_txt([x, 1-gt, 1-q_predicted, 1-pd],
                    ["time(ms)", "ground_truth", "predicted", "peak_detection"],
                    f"s1.3.{line_shape}.txt")
akf = AdaptiveKalmanFilter()
pred = akf.kf.filter(qr)[0][:, 0]
qp = q_predicted[50:]
pd = pd[50:]
gt = gt[50:]
print()
print(f"& {np.mean(np.abs(pd-gt)):.4f} & {np.std(np.abs(pd-gt)):.4f} & {np.sum(np.abs(pd-gt)<=0.001)/qp.shape[0]*100:.2f} & {np.sum(np.abs(pd-gt)<=0.01)/qp.shape[0]*100:.2f}")
print(f"& {np.mean(np.abs(qp-gt)):.4f} & {np.std(np.abs(qp-gt)):.4f} & {np.sum(np.abs(qp-gt)<=0.001)/qp.shape[0]*100:.2f} & {np.sum(np.abs(qp-gt)<=0.01)/qp.shape[0]*100:.2f}")

# q_ref_list = dic['acquire_q_ref']
# q_measured_list = dic['q_measured']
# filtered = []
# x = np.linspace(0, 350, num=len(q_ref_list))
# akf_ref = AdaptiveKalmanFilter(transition_covariance_Q=0.1*np.eye(2)) # 与传统方法相同的参数
# akf_meas = AdaptiveKalmanFilter()
# msf = AdaptiveSensorFusionKalmanFilter()
#
# ref_fil = []
# measured_fil = []
# predicted = []
# dual = []
#
# # for q in acquire_q_ref:
# # for q in q_measured:
# for i in range(len(q_ref_list)):
#     acquire_q_ref = q_ref_list[i]
#     q_measured = q_measured_list[i]
#     acquire_q_ref = akf_ref.predict_update(acquire_q_ref)[0]
#     ref_fil.append(acquire_q_ref)
#     q_measured = akf_meas.predict_update(q_measured)[0]
#     measured_fil.append(q_measured)
#     q_predicted = msf.predict_update(acquire_q_ref, q_measured)
#     predicted.append(q_predicted)
# # 可视化对比
# plt.figure()
# plt.plot(measured_fil, label="meas")
# plt.plot(ref_fil, label="ref")
# plt.plot(predicted, label="pred")
# plt.plot(np.array(dic['qx']), label="Ground truth")
# # plt.plot(dic['failed_to_detect'].astype(int), label="Failed to detect")
# plt.xlabel("Time step")
# plt.ylabel("Signal value")
# plt.legend()
# plt.show()
# print(np.mean(np.abs(np.array(predicted[50:])-np.array(dic['qx'])[50:])))
# # save_structured_txt([x, 1-dic['qx'], 1-np.array(dic['q_ref_filtered']), 1-np.array(dic['q_measured_filtered']), 1-np.array(dic['q_predicted']), np.array(dic['weight_ref']), np.array(dic['weight_measured'])],
# #                     ['time(ms)', 'ground_truth', 'reference_filtered', 'measured_filtered', 'predicted', 'w1_ref', 'w2_measured'],
# #                     f"fusion_{line_shape}.txt")
