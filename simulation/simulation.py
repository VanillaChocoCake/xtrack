import matplotlib.pyplot as plt

import xtrack as xt
import xpart as xp
import xobjects as xo
from parameters import SynchrotronConfiguration, DetectorConfiguration
from myfunc import *
import warnings
import pickle
import scipy.constants as sc

warnings.filterwarnings("ignore")
context = xo.ContextCpu(omp_num_threads="auto")
config = SynchrotronConfiguration()
detector = DetectorConfiguration()
f_rev_range = np.linspace(0, 3.5, num=351)
f_rev_increase_rate = 10e6
min_track_turns = 100
snr = -20
min_freq_res = 10e3
simulation_time = 0.001
max_len = 10
q_prev_queue = q_queue(max_len=max_len, decay_factor=0.8)
method = "sin"
if config.qx > 0.5:
    qx = 1 - config.qx
else:
    qx = config.qx
if config.qy > 0.5:
    qy = 1 - config.qy
else:
    qy = config.qy
qx_list = generate_q_list(method, 351, qx - 0.09, qx + 0.09)
qy_list = generate_q_list(method, 351, qy - 0.09, qy + 0.09)
q_ref_list = []
q_measured_list = []
q_predicted_list = []
peak_detection_list = []
cf_list = []
alpha = 0.45
upper_limit = qx + 0.1
lower_limit = qx - 0.1
# for i in range(len(f_rev_range)):
for i in range(len(qx_list)):
    qy = qy_list[i]
    qx = qx_list[i]
    lmap = xt.LineSegmentMap(length=config.length,
                             qx=qx, qy=qy,
                             betx=config.betx, bety=config.bety,
                             alfx=config.alfx, alfy=config.alfy,
                             dx=config.dx, dy=config.dy,
                             dpx=config.dpx, dpy=config.dpy,
                             longitudinal_mode=config.longitudinal_mode,
                             qs=config.qs,
                             bets=config.bets,
                             dqx=config.dqx, dqy=config.dqy,
                             )
    # f_rev = (4 + f_rev_range[i]) * 1e6
    f_rev = 7.5e6
    # schottky_harmonic = int(np.round(detector.fc / f_rev))
    schottky_harmonic = 6
    f_sampling = 2 * (1 + 1) * f_rev
    # batch_size = int(2 ** np.ceil(np.log2(f_sampling / min_freq_res)))
    batch_size = 4096
    band_width = detector.bandwidth / f_rev
    freq_res = f_sampling / batch_size
    deltaQ = freq_res / f_rev
    side_point_num = np.ceil(500e3 / (2 * freq_res))
    window_size = max(3, 2 * np.floor(side_point_num/6) - 1)
    n_turns = int(np.floor(f_rev * simulation_time / min_track_turns) * min_track_turns)

    line = xt.Line(elements=[lmap])
    line.discard_tracker()
    schottky_monitor = xt.SchottkyMonitor(f_rev=f_rev, schottky_harmonic=schottky_harmonic, n_taylor=32)
    BPM = xt.BeamPositionMonitor(frev=f_rev,
                                 start_at_turn=0, stop_at_turn=n_turns,
                                 sampling_frequency=f_sampling)
    line.append_element(element=schottky_monitor, name="SchottkyMonitor")
    line.append_element(element=BPM, name=f"bpm")
    line.build_tracker()

    beta = config.length * f_rev / sc.c
    gamma = (1 / (1 - beta ** 2)) ** 0.5
    energy0 = gamma * xt.PROTON_MASS_EV
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, _context=context, energy0=energy0)
    bunch = xp.generate_matched_gaussian_bunch(num_particles=int(1e4),
                                               nemitt_x=2 * np.pi * 1e-6, nemitt_y=2 * np.pi * 1e-6,
                                               line=line,
                                               total_intensity_particles=int(1e11),
                                               sigma_z=config.length / 8
                                               )
    line.track(bunch, num_turns=n_turns, with_progress=min_track_turns)

    # In order to take the fc and bandwidth of the detector into consideration,
    # Qx, Qy and band_width(in revolution frequency unit) need to be adjusted to fit fc

    band_width = 0.2
    deltaQ = 1e-3

    # schottky_monitor.process_spectrum(inst_spectrum_len=int(n_turns / 1), deltaQ=deltaQ,
    #                                   band_width=band_width,
    #                                   Qx=qx, Qy=qy,
    #                                   x=True, y=False, z=True,
    #                                   flattop_window=True)
    #
    # plt.figure(figsize=(20, 16))
    # ax1 = plt.subplot(1, 3, 1)
    # ax2 = plt.subplot(1, 3, 2)
    # ax3 = plt.subplot(1, 3, 3)
    # for ax, region in zip([ax1, ax2, ax3], ['lowerH', 'center', 'upperH']):
    #     PSD = schottky_monitor.PSD_avg[region]
    #     ax.plot(schottky_monitor.frequencies[region], schottky_monitor.PSD_avg[region], color='b')
    #     ax.set_xlabel(f'Frequency [$f_0$]')
    #     ax.set_ylabel(f'PSD [arb. units]')
    #     # ax.set_yscale('log')
    # plt.tight_layout()
    # plt.savefig(f"schottky_n_turns_{n_turns}.png")
    # plt.show()

    x_data = BPM.x_mean
    mask_nan = np.isnan(x_data)
    # x_data = BPM.y_mean
    x_data = np.nan_to_num(x_data, nan=0)
    x_data_noisy = generate_noisy_signal(x_data, snr)
    x_data_noisy = windowed_reshape(x_data_noisy, batch_size)
    tune_unit, psd = cal_psd(x_data_noisy, batch_size, f_sampling, window_size, f_rev, lower_limit, upper_limit)
    index_bool_maxima, index_value_maxima= find_local_maxima(psd)
    index_bool_minima, index_value_minima= find_local_minima(psd)
    weight_amplitude = normalize_to_01(psd[index_bool_maxima]) - 1
    try:
        q_ref = q_prev_queue.q_ref()
        assert q_ref > 0
    except:
        q_ref = np.mean(tune_unit[index_bool_maxima])
    q_pred = q_prev_queue.q_pred()
    q_ref_list.append(q_ref)
    q_predicted_list.append(q_pred)
    distance = normalize_to_01(abs(tune_unit[index_bool_maxima] - (q_ref + q_pred)/2)) - 1
    weight_distance = 1 - distance
    confidence = alpha * weight_amplitude + (1 - alpha) * weight_distance
    q_measured_index = np.argmax(confidence)
    q_measured = tune_unit[index_bool_maxima][q_measured_index]
    q_confidence = max(confidence)
    if q_confidence >= 0.95:
        alpha = max(0.1, alpha - 0.01)
    q_prev_queue.append(q_measured, tune_unit, psd)
    q_measured_list.append(q_measured)
    q_peak_detection = tune_unit[index_bool_maxima][np.argmax(weight_amplitude)]
    peak_detection_list.append(q_peak_detection)
    closest_minima = find_closest_values(index_value_maxima[q_measured_index], index_value_minima)
    cf_start = int(max([0, # Should be bigger than 0
                        index_value_maxima[q_measured_index] - np.floor(side_point_num/2), # Expected span
                        closest_minima[0]])) # Stop at the first minima on the left
    cf_end = int(min([len(tune_unit) - 1,
                      index_value_maxima[q_measured_index] + np.floor(side_point_num/2),
                      closest_minima[1]]))
    q_curve_fitting = gaussian_peak_fit(tune_unit[cf_start:cf_end], psd[cf_start:cf_end])
    cf_list.append(q_curve_fitting)
    # plt.figure()
    # plt.plot(tune_unit, normalize_to_01(psd), label="sum")
    # plt.plot(tune_unit, normalize_to_01(q_prev_queue.psd), label="ref")
    # plt.legend()
    # plt.show()
    print(f"qx:{qx}, q_ref:{q_ref}, q_predicted:{q_pred}, q_measured:{q_measured}, confidence:{q_confidence * 100}%, peak_detection:{q_peak_detection}, curve_fitting:{q_curve_fitting}")

plt.figure()
plt.plot(qx_list, label='nominal')
plt.plot(q_ref_list, label='reference')
plt.plot(q_predicted_list, label='predicted')
plt.plot(q_measured_list, label='measured')
plt.plot(peak_detection_list, label='peak detection')
plt.plot(cf_list, label='curve fitting')
plt.legend()
plt.show()
res_dic = {'qx': qx_list,
           'q_ref': q_ref_list,
           'q_predicted': q_predicted_list,
           'q_measured': q_measured_list,
           'peak_detection': peak_detection_list,
           'curve_fitting': cf_list}
with open(f"{method}_sum.pkl", "wb") as f:
    pickle.dump(res_dic, f, protocol=pickle.HIGHEST_PROTOCOL)

