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
q_measured_list = []
peak_detection_list = []
cf_list = []
alpha = 0.45
# for i in range(len(f_rev_range)):
for i in range(len(qx_list)):
    qy = qy_list[i]
    qx = qx_list[i]
    q = qx
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
    # x_data_noisy[mask_nan] = 0
    # x_data_noisy = noise_reduction_gate(x_data_noisy, f_rev, f_sampling)
    # plot([x_data, x_data_noisy])
    # x_data_noisy = apply_bandpass_filter(x_data_noisy, fs=f_sampling, lowcut=detector.fl, highcut=detector.fh)
    x_data_noisy = windowed_reshape(x_data_noisy, batch_size)
    # N = batch_size
    # psd_s = 0
    # # psd_p = 1
    # for j in range(x_data_noisy.shape[0]):
    #     # 计算FFT
    #     X = fft(x_data_noisy[j, :])
    #     # fftshift将零频分量移到中心
    #     X_shifted = fftshift(X)
    #     # 生成频率轴，从-f_sampling/2到f_sampling/2
    #     # 计算PSD（归一化方法可根据实际需要调整）
    #     psd_j = np.abs(X_shifted) ** 2 / (N * f_sampling)
    #     # psd_j = sgolay_filter(psd_j, window_size-2, 4)
    #     psd_j = gaussian_filter(psd_j, window_size)
    #     freqs, psd_j = fold_spectrum(psd_j, f_rev, f_sampling)
    #     psd_j = psd_j[(freqs / f_rev >= 0.22) & (freqs / f_rev <= 0.42)]
    #     # psd_j[0] = 0
    #     # psd_j[-1] = 0
    #     psd_s += psd_j
    #     # psd_p *= (normalize_to_01(psd_j) + 1)
    #     # psd_p = normalize_to_01(psd_p) + 1
    #     # psd_p *= normalize_to_01(psd_j)
    #     # psd_p = normalize_to_01(psd_p)
    # psd_s -= min(psd_s)
    # tune_unit = freqs[(freqs / f_rev >= 0.22) & (freqs / f_rev <= 0.42)] / f_rev
    # # psd_s = normalize_to_01(psd_s)
    # psd = psd_s
    tune_unit, psd = cal_psd(x_data_noisy, batch_size, f_sampling, window_size, f_rev, q - 0.01, q + 0.01)
    index = find_local_maxima(psd)
    weight_amplitude = normalize_to_01(psd[index]) - 1
    try:
        q_ref = q_prev_queue.q_ref()
        assert q_ref > 0
    except:
        q_ref = np.mean(tune_unit[index])
    q_pred = q_prev_queue.q_pred()
    distance = normalize_to_01(abs(tune_unit[index] - (q_ref + q_pred)/2)) - 1
    weight_distance = 1 - distance
    confidence = alpha * weight_amplitude + (1 - alpha) * weight_distance
    q_measured_index = np.argmax(confidence)
    q_measured = tune_unit[index][q_measured_index]
    q_confidence = max(confidence)
    if q_confidence >= 0.95:
        alpha = max(0.1, alpha - 0.01)
    q_prev_queue.append(q_measured, tune_unit, psd)
    q_measured_list.append(q_measured)
    peak_detection_list.append(tune_unit[index][np.argmax(weight_amplitude)])
    plt.figure()
    plt.plot(tune_unit, normalize_to_01(psd), label="sum")
    plt.plot(tune_unit, normalize_to_01(q_prev_queue.psd), label="ref")
    # plt.plot(tune_unit, normalize_to_01(psd_p), label="prod")
    plt.legend()
    plt.show()
    print(f"qx:{qx}, q_ref:{q_ref}, q_measured:{q_measured}, confidence:{q_confidence * 100}%")

plot([qx_list, q_measured_list, peak_detection_list])
res_dic = {'qx': qx_list,
           'q_measured': q_measured_list,
           'peak_detection': peak_detection_list}
with open(f"random_sum.pkl", "wb") as f:
    pickle.dump(res_dic, f, protocol=pickle.HIGHEST_PROTOCOL)

