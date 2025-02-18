import random

import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo
from parameters import SynchrotronConfiguration, DetectorConfiguration
from myfunc import *

random.seed(114514)
context = xo.ContextCpu(omp_num_threads="auto")
config = SynchrotronConfiguration()
detector = DetectorConfiguration()
# if config.longitudinal_mode == "linear_fixed_qs":
#     lmap = xt.LineSegmentMap(length=config.length,
#                              qx=config.qx, qy=config.qy,
#                              betx=config.betx, bety=config.bety,
#                              alfx=config.alfx, alfy=config.alfy,
#                              dx=config.dx, dy=config.dy,
#                              dpx=config.dpx, dpy=config.dpy,
#                              longitudinal_mode=config.longitudinal_mode,
#                              qs=config.qs,
#                              bets=config.bets,
#                              dqx=config.dqx, dqy=config.dqy,
#                              )
# else:
#     error()

f_rev_range = np.linspace(0, 3.5, num=351)
# f_rev_range = np.linspace(2, 2.05, num=6)
f_rev_increase_rate = 10e6
min_track_turns = 100
snr = -20
min_freq_res = 10e3
simulation_time = 0.001
max_len = 8
q_prev_queue = q_queue(max_len=max_len)
qx_list = (1 - config.qx) + np.linspace(-0.09, 0.09, num=351)
qy_list = config.qy + np.linspace(-0.09, 0.09, num=351)
for i in range(len(f_rev_range)):
    qy = qy_list[i]
    qx = qx_list[i]
    qx = 0.33
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
    f_rev = (4 + f_rev_range[i]) * 1e6
    schottky_harmonic = int(np.round(detector.fc / f_rev))
    schottky_harmonic = 6
    f_sampling = 2 * (1 + 1) * f_rev
    batch_size = int(2 ** np.ceil(np.log2(f_sampling / min_freq_res)))
    band_width = detector.bandwidth / f_rev
    freq_res = f_sampling / batch_size
    deltaQ = freq_res / f_rev
    side_point_num = np.ceil(500e3 / (2 * freq_res))
    window_size = max(3, 2 * np.floor(side_point_num/4) - 1)
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

    schottky_monitor.process_spectrum(inst_spectrum_len=int(n_turns / 1), deltaQ=deltaQ,
                                      band_width=band_width,
                                      Qx=qx, Qy=qy,
                                      x=True, y=False, z=True,
                                      flattop_window=True)

    plt.figure(figsize=(20, 16))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    for ax, region in zip([ax1, ax2, ax3], ['lowerH', 'center', 'upperH']):
        PSD = schottky_monitor.PSD_avg[region]
        ax.plot(schottky_monitor.frequencies[region], schottky_monitor.PSD_avg[region], color='b')
        ax.set_xlabel(f'Frequency [$f_0$]')
        ax.set_ylabel(f'PSD [arb. units]')
        # ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f"schottky_n_turns_{n_turns}.png")
    plt.show()

    x_data = BPM.x_mean
    mask_nan = np.isnan(x_data)
    # x_data = BPM.y_mean
    x_data = np.nan_to_num(x_data, nan=0)
    x_data_noisy = generate_noisy_signal(x_data, snr)
    # x_data_noisy[mask_nan] = 0
    # x_data_noisy = noise_reduction_gate(x_data_noisy, f_rev, f_sampling)
    # plot([x_data, x_data_noisy])
    # x_data_noisy = apply_bandpass_filter(x_data_noisy, fs=f_sampling, lowcut=detector.fl, highcut=detector.fh)
    x_data_noisy = reshape_with_padding(x_data_noisy, batch_size)
    # N = len(x_data)  # 数据点数
    N = batch_size
    psd_s = 0
    psd_p = 1
    for i in range(x_data_noisy.shape[0]):
        # 计算FFT
        X = fft(x_data_noisy[i, :])
        # fftshift将零频分量移到中心
        X_shifted = fftshift(X)
        # 生成频率轴，从-f_sampling/2到f_sampling/2
        # 计算PSD（归一化方法可根据实际需要调整）
        psd_i = np.abs(X_shifted) ** 2 / (N * f_sampling)

        # psd_i = sgolay_filter(psd_i, window_size, 3)
        psd_i = gaussian_filter(psd_i, window_size)
        freqs, psd_i = fold_spectrum(psd_i, f_rev, f_sampling)
        half_frev = (freqs / f_rev >= 0.22) & (freqs / f_rev <= 0.42)
        psd_i = psd_i[half_frev]
        psd_i[0] = 0
        psd_i[-1] = 0
        psd_s += psd_i
        # psd_p *= (normalize_to_01(psd_i) + 1)
        # psd_p = normalize_to_01(psd_p) + 1
        psd_p *= normalize_to_01(psd_i)
        psd_p = normalize_to_01(psd_p)
    psd_s[0] = psd_s[1]
    psd_s[-1] = psd_s[-2]
    psd_s -= min(psd_s)
    psd_s = normalize_to_01(psd_s)
    psd_p  = normalize_to_01(psd_p)
    plt.figure()
    tune_unit = freqs[half_frev] / f_rev
    plt.plot(tune_unit, psd_s)
    plt.plot(tune_unit, psd_p)
    plt.show()
    psd = psd_p
    index = find_local_maxima(psd)
    weight_amplitude = normalize_to_01(psd[index])
    try:
        confidence_prev = np.array(q_prev_queue.weight_decay) * np.array(q_prev_queue.confidence_queue)
        q_ref = q_prev_queue.q_queue[np.argmax(confidence_prev)]
        distance = normalize_to_01(abs(tune_unit[index] - q_ref))
        weight_distance = 1 - distance
        confidence = 0.4 * weight_amplitude + 0.6 * weight_distance
        q_measured_index = np.argmax(confidence)
        q_measured = tune_unit[index][q_measured_index]
    except:
        q_measured_index = np.argmax(weight_amplitude)
        q_measured = tune_unit[index][q_measured_index]
        confidence = [0.5*max(weight_amplitude)]
    q_confidence = max(confidence)
    q_prev_queue.append(q_measured, q_confidence)
    print(f"{q_measured}, {q_confidence*100}%")
