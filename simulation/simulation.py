import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
import xpart as xp
import xobjects as xo
from parameters import SynchrotronConfiguration, DetectorConfiguration
from myfunc import *
import shutup
import pickle
import scipy.constants as sc

shutup.please()
context = xo.ContextCpu(omp_num_threads="auto")
config = SynchrotronConfiguration()
detector = DetectorConfiguration()

detector.bandwidth = 10e6
sideband_width = 500e3
f_rev_increase_rate = 10e6
min_track_turns = 10
snr = -20
min_freq_res = 10e3
simulation_time = 0.001
max_len = 10
decay_factor = 0.8
method = "cos"
f_rev_mode = 7.5e6
alpha = 0.45
exclude_coherent = False
schottky_harmonic = 2
interpolate_method = "cubic"
interpolate_coef = 2

q_prev_queue = q_queue(max_len=max_len, decay_factor=0.8)
if config.qx > 0.5:
    qx = 1 - config.qx
else:
    qx = config.qx
if config.qy > 0.5:
    qy = 1 - config.qy
else:
    qy = config.qy
upper_limit = qx + 0.1
lower_limit = qx - 0.1
if f_rev_mode == "ramping":
    f_rev_range, covered_frequency_bands \
        = covered_frequency_bands_minmax(central_frequency=detector.fc, bandwidth=detector.bandwidth,
                                         tune_min=lower_limit, tune_max=upper_limit,
                                         sideband_width=sideband_width,
                                         start_frequency=4e6, end_frequency=7.5e6, step=0.01e6)
else:
    f_rev_range = f_rev_mode*np.ones(351)
qx_list = generate_q_list(method, len(f_rev_range), qx - 0.09, qx + 0.09)
qy_list = generate_q_list(method, len(f_rev_range), qy - 0.09, qy + 0.09)
plot(qx_list)
q_ref_list = []
q_measured_list = []
q_predicted_list = []
peak_detection_list = []
cf_list = []
w1_list = []
w2_list = []
failed_to_detect = np.zeros_like(f_rev_range, dtype=bool)
batch_size = int(2**np.ceil(np.log2(2*schottky_harmonic*np.max(f_rev_range)/min_freq_res)))
# kf = AdaptiveKalmanFilter(initial_state=qx)
kf = DualDetectorAdaptiveKalmanFilter(initial_state=qx)
q_measured = 0.3
for i in range(len(qx_list)):
    qy = qy_list[i]
    qx = qx_list[i]
    qx = float(qx)
    qy = float(qy)
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
    f_rev = f_rev_range[i]
    print("f_rev:", f_rev)
    f_sampling = 2 * schottky_harmonic * f_rev
    band_width = detector.bandwidth / f_rev
    freq_res = f_sampling / batch_size
    deltaQ = freq_res / f_rev
    side_point_num = sideband_width//(2 * freq_res) + 1
    window_size = int(max(3, side_point_num//4))
    # window_size = batch_size // (2*schottky_harmonic*2) // 20
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
    bunch = xp.generate_matched_gaussian_bunch(num_particles=int(1e3),
                                               nemitt_x=2 * np.pi * 1e-6, nemitt_y=2 * np.pi * 1e-6,
                                               line=line,
                                               total_intensity_particles=int(1e11),
                                               sigma_z=config.length / 8
                                               )
    line.track(bunch, num_turns=n_turns, with_progress=min_track_turns)

    # In order to take the fc and bandwidth of the detector into consideration,
    # Qx, Qy and band_width(in revolution frequency unit) need to be adjusted to fit fc

    # band_width = 0.2
    # deltaQ = 1e-5
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
    # freq, clean_spectrum = cal_psd_normal(x_data, f_sampling)
    # plot(freq, clean_spectrum)
    x_data, noise = generate_noisy_signal(x_data, snr, exclude_coherent=exclude_coherent)
    # x_data_reshaped = windowed_reshape(x_data_reshaped, batch_size)
    x_data_reshaped, batch_size_processed = windowed_reshape(x_data, batch_size)
    noise_reshaped, _ = windowed_reshape(noise, batch_size)
    if not covered_by_detector(central_frequency=detector.fc, bandwidth=detector.bandwidth,
                               sideband_width=sideband_width,
                               tune=qx, current_frequency=f_rev):
        tune_unit, psd = cal_psd(x_data=noise_reshaped, noise=noise_reshaped, batch_size=batch_size_processed,
                                 f_sampling=f_sampling, window_size=window_size, f_rev=f_rev,
                                 tune_unit_lower_limit=lower_limit, tune_unit_upper_limit=upper_limit,
                                 side_point_num=side_point_num, snr=snr,
                                 interpolate_method=interpolate_method, interpolate_coef=interpolate_coef,
                                 exclude_coherent=False)
        failed_to_detect[i] = True
        print(f"Betatron tune can not be measured at this frequency.")
    else:
        tune_unit, psd = cal_psd(x_data=x_data_reshaped, noise=noise_reshaped, batch_size=batch_size_processed,
                                 f_sampling=f_sampling, window_size=window_size, f_rev=f_rev,
                                 tune_unit_lower_limit=lower_limit, tune_unit_upper_limit=upper_limit,
                                 side_point_num=side_point_num, snr=snr,
                                 interpolate_method=interpolate_method, interpolate_coef=interpolate_coef,
                                 exclude_coherent=exclude_coherent)
    index_bool_maxima, index_value_maxima= find_local_maxima(psd)
    index_bool_minima, index_value_minima= find_local_minima(psd)
    weight_amplitude = normalize_to_01(psd[index_bool_maxima]) - 1
    try:
        q_ref = q_prev_queue.q_ref()
        assert q_ref > 0
        # q_pred = q_prev_queue.q_pred()
        # q_pred = kf.predict_update(0.5*q_measured + 0.5*q_ref)
    except:
        q_ref = np.mean(tune_unit[index_bool_maxima])
        # q_pred = q_ref
        # q_pred = kf.predict_update(q_ref)
    q_pred = kf.predict_update(q_ref, q_measured)
    w1, w2 = kf.detector_weights()
    w1_list.append(w1)
    w2_list.append(w2)
    # kf.predict()
    # q_pred = kf.update(0.5*q_measured + 0.5*q_ref)[0][0]
    q_ref_list.append(q_ref)
    q_predicted_list.append(q_pred)
    distance = normalize_to_01(abs(tune_unit[index_bool_maxima] - (q_ref + q_pred)/2)) - 1
    weight_distance = 1 - distance
    confidence = alpha * weight_amplitude + (1 - alpha) * weight_distance
    q_measured_index = np.argmax(confidence)
    q_measured = tune_unit[index_bool_maxima][q_measured_index]
    # q_measured = q_measured if np.abs(q_measured - q_pred) < 0.1 else q_pred
    # q_measured = 0.5*q_measured + 0.5*q_ref
    q_confidence = max(confidence)
    q_prev_queue.append(q_measured, tune_unit, psd)
    q_measured_list.append(q_measured)
    q_peak_detection = tune_unit[index_bool_maxima][np.argmax(weight_amplitude)]
    peak_detection_list.append(q_peak_detection)
    closest_minima = find_closest_values(index_value_maxima[q_measured_index], index_value_minima)
    cf_start = int(max(0, index_value_maxima[q_measured_index] - side_point_num*interpolate_coef//2))
    cf_end = int(min(len(tune_unit) - 1, index_value_maxima[q_measured_index] + side_point_num*interpolate_coef//2))
    cf_params = gaussian_peak_fit(tune_unit[cf_start:cf_end + 1], psd[cf_start:cf_end + 1])
    q_curve_fitting = cf_params[1]
    cf_list.append(q_curve_fitting)
    plt.figure()
    plt.plot(tune_unit, normalize_to_01(psd), label="sum")
    plt.plot(tune_unit, normalize_to_01(q_prev_queue.psd), label="ref")
    plt.legend()
    plt.show()
    print(f"qx:{qx: .4f}, q_ref:{q_ref: .4f}, q_predicted:{q_pred: .4f}, q_measured:{q_measured: .4f}, confidence:{q_confidence * 100: .2f}%, peak_detection:{q_peak_detection: .4f}, curve_fitting:{q_curve_fitting: .4f}")

dic = {'qx': qx_list,
       'q_ref': q_ref_list,
       'q_predicted': q_predicted_list,
       'q_measured': q_measured_list,
       'peak_detection': peak_detection_list,
       'curve_fitting': cf_list,
       'failed_to_detect': failed_to_detect,
       'weight_ref': w1_list,
       'weight_measured': w2_list}
plot_measured_results(dic=dic)
with open(f"{method}_{snr}_frev_{f_rev_mode}_{'without' if exclude_coherent else 'with'}_coherent.pkl", "wb") as f:
    pickle.dump(dic, f, protocol=pickle.HIGHEST_PROTOCOL)
plt.figure()
plt.plot(f_rev_range, w1_list, label="w1")
plt.plot(f_rev_range, w2_list, label="w2")
plt.legend()
plt.show()

