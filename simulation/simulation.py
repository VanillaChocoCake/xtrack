import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
import xpart as xp
import xobjects as xo
from parameters import SynchrotronConfiguration, DetectorConfiguration, AlgorithmConfiguration, calculate_slip_factor, calculate_bets
from Aegithalos_caudatus import *
import shutup
import pickle
import scipy.constants as sc

def tune_measurement_algorithm(synchrotron_parameters: SynchrotronConfiguration,
                               detector_parameters: DetectorConfiguration,
                               algorithm_parameters: AlgorithmConfiguration) -> None:
    context = xo.ContextCpu(omp_num_threads="auto")
    sideband_width = algorithm_parameters.sideband_width
    f_rev_increase_rate = algorithm_parameters.f_rev_increase_rate
    min_track_turns = algorithm_parameters.min_track_turns
    snr = algorithm_parameters.snr
    min_freq_res = algorithm_parameters.min_freq_res
    simulation_time = algorithm_parameters.simulation_time
    max_len = algorithm_parameters.max_len
    decay_factor = algorithm_parameters.decay_factor
    line_shape = algorithm_parameters.line_shape
    f_rev_mode = algorithm_parameters.f_rev_mode
    alpha = algorithm_parameters.alpha
    exclude_coherent = algorithm_parameters.exclude_coherent
    schottky_harmonic = algorithm_parameters.schottky_harmonic
    interpolate_method = algorithm_parameters.interpolate_method
    interpolate_coef = algorithm_parameters.interpolate_coef
    smoothing_method = algorithm_parameters.smoothing_method

    ema_psd = EMA_PSD(max_len=max_len, decay_factor=decay_factor)
    if synchrotron_parameters.qx > 0.5:
        qx = 1 - synchrotron_parameters.qx
    else:
        qx = synchrotron_parameters.qx
    if synchrotron_parameters.qy > 0.5:
        qy = 1 - synchrotron_parameters.qy
    else:
        qy = synchrotron_parameters.qy
    upper_limit = qx + 0.1
    lower_limit = qx - 0.1
    if f_rev_mode == "ramping":
        f_rev_range, covered_frequency_bands \
            = covered_frequency_bands_minmax(central_frequency=detector_parameters.fc, bandwidth=detector_parameters.bandwidth,
                                             tune_min=lower_limit, tune_max=upper_limit,
                                             sideband_width=sideband_width,
                                             start_frequency=4e6, end_frequency=7.5e6, step=0.01e6)
    else:
        f_rev_range = f_rev_mode * np.ones(351)
    qx_list = generate_q_list(line_shape, len(f_rev_range), qx - 0.09, qx + 0.09)
    qy_list = generate_q_list(line_shape, len(f_rev_range), qy - 0.09, qy + 0.09)
    # plot(qx_list)
    q_ref_list = []
    q_ref_filtered = []
    q_measured_list = []
    q_measured_filtered = []
    q_predicted_list = []
    peak_detection_list = []
    cf_list = []
    w1_list = []
    w2_list = []
    failed_to_detect = np.zeros_like(f_rev_range, dtype=bool)
    min_freq = np.min(f_rev_range)
    max_freq = np.max(f_rev_range)
    min_exponential = np.log2(2*schottky_harmonic*min_freq/min_freq_res)
    max_exponential = np.log2(2*schottky_harmonic*max_freq/min_freq_res)
    exponential = np.round((min_exponential + max_exponential)/2)
    batch_size = int(2 ** exponential)
    dkf = AdaptiveSensorFusionKalmanFilter(initial_state=qx)
    akf_ref = AdaptiveKalmanFilter(transition_covariance_Q=0.1*np.eye(2))
    akf_meas = AdaptiveKalmanFilter(transition_covariance_Q=0.001*np.eye(2))
    q_measured = 0.3
    for i in range(len(qx_list)):
        print(i)
        # Part Simulation
        qy = qy_list[i]
        qx = qx_list[i]
        qx = float(qx)
        qy = float(qy)
        f_rev = f_rev_range[i]
        synchrotron_parameters.slip_factor = calculate_slip_factor(f_rev)
        synchrotron_parameters.bets = calculate_bets(synchrotron_parameters.slip_factor)
        lmap = xt.LineSegmentMap(length=synchrotron_parameters.length,
                                 qx=qx, qy=qy,
                                 betx=synchrotron_parameters.betx, bety=synchrotron_parameters.bety,
                                 alfx=synchrotron_parameters.alfx, alfy=synchrotron_parameters.alfy,
                                 dx=synchrotron_parameters.dx, dy=synchrotron_parameters.dy,
                                 dpx=synchrotron_parameters.dpx, dpy=synchrotron_parameters.dpy,
                                 longitudinal_mode=synchrotron_parameters.longitudinal_mode,
                                 qs=synchrotron_parameters.qs,
                                 bets=synchrotron_parameters.bets,
                                 dqx=synchrotron_parameters.dqx, dqy=synchrotron_parameters.dqy,
                                 )
        print("f_rev:", f_rev)
        f_sampling = 2 * schottky_harmonic * f_rev
        freq_res = f_sampling / batch_size
        side_point_num = sideband_width // (2 * freq_res) + 1
        window_size = int(max(3, 2*side_point_num//2 + 1))
        n_turns = int(np.floor(f_rev * simulation_time / min_track_turns) * min_track_turns)

        line = xt.Line(elements=[lmap])
        line.discard_tracker()
        # monitor = xt.ParticlesMonitor(_context=context, start_at_turn=1, stop_at_turn=n_turns, num_particles=int(1e3))
        # schottky_monitor = xt.SchottkyMonitor(f_rev=f_rev, schottky_harmonic=np.ceil(schottky_harmonic), n_taylor=32)
        BPM = xt.BeamPositionMonitor(frev=f_rev,
                                     start_at_turn=0, stop_at_turn=n_turns,
                                     sampling_frequency=f_sampling)
        # line.append_element(element=schottky_monitor, name="SchottkyMonitor")
        line.append_element(element=BPM, name="BPM")
        # line.append_element(element=monitor, name="normal_monitor")
        line.build_tracker()

        beta = synchrotron_parameters.length * f_rev / sc.c
        gamma = (1 / (1 - beta ** 2)) ** 0.5
        energy0 = gamma * xt.PROTON_MASS_EV
        line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, _context=context, energy0=energy0)
        bunch = xp.generate_matched_gaussian_bunch(num_particles=int(1e3),
                                                   nemitt_x=SynchrotronConfiguration.nemitt,
                                                   nemitt_y=SynchrotronConfiguration.nemitt,
                                                   line=line,
                                                   total_intensity_particles=synchrotron_parameters.intensity,
                                                   sigma_z=synchrotron_parameters.length / 8
                                                   )
        line.track(bunch, num_turns=n_turns, with_progress=min_track_turns)

        # # Another way to generate Schottky signal, but fixed sampling frequency
        # freqs, x_psd = cal_psd(monitor.x, f_rev)
        # _, y_psd = cal_psd(monitor.y, f_rev)
        # _, z_psd = cal_psd(monitor.zeta, f_rev)
        # freqs /= f_rev
        # plot(freqs, x_psd)
        # plot(freqs, y_psd)
        # plot(freqs, z_psd)


        # band_width = 0.05
        # deltaQ = 1e-4
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
        #     ax.plot(schottky_monitor.frequencies[region], schottky_monitor.PSD_avg[region], color='b')
        #     ax.set_xlabel(f'Frequency [$f_0$]')
        #     ax.set_ylabel(f'PSD [arb. units]')
        #     # ax.set_yscale('log')
        # plt.tight_layout()
        # plt.savefig(f"schottky_n_turns_{n_turns}.png")
        # plt.show()
        #
        # # 创建结构化数组用于保存数据
        # data_to_save = np.column_stack((
        #     schottky_monitor.frequencies['lowerH'],
        #     schottky_monitor.PSD_avg['lowerH'],
        #     schottky_monitor.frequencies['center'],
        #     schottky_monitor.PSD_avg['center'],
        #     schottky_monitor.frequencies['upperH'],
        #     schottky_monitor.PSD_avg['upperH']
        # ))
        #
        # # 生成带列标签的文件头
        # header = (
        #     "# lowerH_frequency    lowerH_PSD    center_frequency    center_PSD    upperH_frequency    upperH_PSD"
        # )
        #
        # # 保存到文本文件（科学计数法格式）
        # np.savetxt(
        #     f"schottky_data.txt",
        #     data_to_save,
        #     fmt='%.6e',  # 控制精度为6位小数
        #     delimiter='    ',  # 使用4空格分隔列
        #     header=header,
        #     comments=''  # 移除自动添加的注释符
        # )

        # Part Algorithm
        x_data = BPM.x_mean
        # x_data = BPM.y_mean
        x_data = np.nan_to_num(x_data, nan=0)

        # f, psd = cal_psd(x_data, f_sampling)
        # f /= f_rev
        # mask = f >= (schottky_harmonic - 2)
        # f = f[mask] - (schottky_harmonic - 1)
        # psd = psd[mask]
        # data_to_save = np.column_stack((f, psd))
        # header = (
        #     "# Tune_unit    psd"
        # )
        # np.savetxt(
        #     f"folding.txt",
        #     data_to_save,
        #     fmt='%.6e',  # 控制精度为6位小数
        #     delimiter='    ',  # 使用4空格分隔列
        #     header=header,
        #     comments=''  # 移除自动添加的注释符
        # )

        t = np.linspace(0, len(x_data)/f_sampling, len(x_data))
        if exclude_coherent:

            # f, psd_ori = cal_psd(x_data, f_sampling)
            _, noise = generate_noisy_signal(x_data, snr)
            x_data = exclude_coherent_signal(t, x_data, f_sampling, np.array([0.21, 0.49])*f_rev)
            x_data += noise
            # _, psd_excluded = cal_psd(x_data, f_sampling)
            # f /= f_rev
            # mask = (f > 0.22) & (f < 0.42)
            # f = f[mask]
            # psd_ori = psd_ori[mask]
            # psd_excluded = psd_excluded[mask]
            # data_to_save = np.column_stack((
            #     f, psd_ori, psd_excluded
            # ))
            # header = (
            #     "# Tune_unit    original_psd    without_coherent_signal_psd"
            # )
            # np.savetxt(
            #     f"rmsfit_transverse_coherent_excluded.txt",
            #     data_to_save,
            #     fmt='%.6e',  # 控制精度为6位小数
            #     delimiter='    ',  # 使用4空格分隔列
            #     header=header,
            #     comments=''  # 移除自动添加的注释符
            # )
        else:
            x_data, noise = generate_noisy_signal(x_data, snr)

        x_data_reshaped = windowed_reshape(x_data, batch_size)
        noise_reshaped = windowed_reshape(noise, batch_size)

        common_spectral_params = {
            'batch_size': batch_size,
            'f_sampling': f_sampling,
            'window_size': window_size,
            'f_rev': f_rev,
            'tune_unit_lower_limit': lower_limit,
            'tune_unit_upper_limit': upper_limit,
            'smoothing_method': smoothing_method,
            'interpolate_method': interpolate_method,
            'interpolate_coef': interpolate_coef,
        }
        if not covered_by_detector(central_frequency=detector_parameters.fc, bandwidth=detector_parameters.bandwidth,
                                   sideband_width=sideband_width,
                                   tune=qx, current_frequency=f_rev):
            tune_unit, psd = spectral_processing(x_data=noise_reshaped, **common_spectral_params)
            failed_to_detect[i] = True
            print(f"Betatron tune can not be measured at this frequency.")
        else:
            tune_unit, psd = spectral_processing(x_data=x_data_reshaped, **common_spectral_params)

        # Determination of reference tune (Sensor 1)
        index_bool_maxima, index_value_maxima = find_local_maxima(psd)
        weight_amplitude = normalize_to_0_1(psd[index_bool_maxima])
        try:
            q_ref = ema_psd.q_ref()
            assert (q_ref > lower_limit) and (q_ref < upper_limit)
        except:
            q_ref = np.mean(tune_unit[index_bool_maxima])
            q_pred = q_ref
        q_ref_list.append(q_ref)
        q_ref = akf_ref.predict_update(q_ref)[0]
        q_ref_filtered.append(q_ref)


        # Determination of measured tune (Sensor 2) using weighted linear combination
        # distance = normalize_to_0_1(abs(tune_unit[index_bool_maxima] - (q_ref + q_pred) / 2))
        distance = normalize_to_0_1(abs(tune_unit[index_bool_maxima] - q_pred))
        weight_distance = 1 - distance
        confidence = alpha * weight_amplitude + (1 - alpha) * weight_distance
        q_measured_index = np.argmax(confidence)
        q_measured = tune_unit[index_bool_maxima][q_measured_index]
        q_confidence = max(confidence)
        q_measured_list.append(q_measured)
        q_measured = akf_meas.predict_update(q_measured)[0]
        q_measured_filtered.append(q_measured)

        # Predict the tune value using adaptive dual sensor Kalman filter
        q_pred = dkf.predict_update(q_ref, q_measured)
        w1, w2 = dkf.detector_weights()
        w1_list.append(w1)
        w2_list.append(w2)
        q_predicted_list.append(q_pred)

        # Update previous PSD
        ema_psd.append(tune_unit, psd)


        # Peak detection method, commonly used to measure coherent tune
        q_peak_detection = tune_unit[index_bool_maxima][np.argmax(weight_amplitude)]
        peak_detection_list.append(q_peak_detection)

        # # Curve fitting method, commonly used to measure incoherent tune and chromaticity, which is
        # # not suitable for low SNR and limited frequency resolution scenarios
        # cf_start = int(max(0, index_value_maxima[q_measured_index] - side_point_num * interpolate_coef // 2))
        # cf_end = int(
        #     min(len(tune_unit) - 1, index_value_maxima[q_measured_index] + side_point_num * interpolate_coef // 2))
        # cf_params = gaussian_peak_fit(tune_unit[cf_start:cf_end + 1], psd[cf_start:cf_end + 1])
        # q_curve_fitting = cf_params[1]
        # cf_list.append(q_curve_fitting)

        plt.figure()
        plt.plot(tune_unit, normalize_to_0_1(psd), label="sum")
        plt.plot(tune_unit, normalize_to_0_1(ema_psd.psd), label="ref")
        plt.legend()
        plt.show()
        print(
            f"qx:{qx: .4f}, "
            f"q_ref:{q_ref: .4f}, "
            f"q_predicted:{q_pred: .4f}, "
            f"q_measured:{q_measured: .4f}, "
            f"confidence:{q_confidence * 100: .2f}%, "
            f"peak_detection:{q_peak_detection: .4f}, "
            # f"curve_fitting:{q_curve_fitting: .4f}"
        )

    dic = {'qx': qx_list,
           'q_ref': q_ref_list,
           'q_ref_filtered': q_ref_filtered,
           'q_predicted': q_predicted_list,
           'q_measured': q_measured_list,
           'q_measured_filtered': q_measured_filtered,
           'peak_detection': peak_detection_list,
           'curve_fitting': cf_list,
           'failed_to_detect': failed_to_detect,
           'weight_ref': w1_list,
           'weight_measured': w2_list}
    plot_measured_results(dic=dic,
                          title=f"{int(detector_parameters.bandwidth/1e6)}MHz_{smoothing_method}_{line_shape}_{snr}_frev_{f_rev_mode}_{'without' if exclude_coherent else 'with'}_coherent.pkl")
    with open(f"{int(detector_parameters.bandwidth/1e6)}MHz_{smoothing_method}_{line_shape}_{snr}_frev_{f_rev_mode}_{'without' if exclude_coherent else 'with'}_coherent.pkl", "wb") as f:
        pickle.dump(dic, f, protocol=pickle.HIGHEST_PROTOCOL)
    plt.figure()
    plt.plot(w1_list, label="w1")
    plt.plot(w2_list, label="w2")
    plt.legend()
    plt.show()
    print(1)

if __name__ == "__main__":
    synchrotron_parameters = SynchrotronConfiguration()
    detector_parameters = DetectorConfiguration(bandwidth=10e6)
    smoothing_method_list = ["gaussian", "sgolay"]
    # smoothing_method_list = ["gaussian"]
    snr_list = [-20, -15, -10]
    # snr_list = [-20]
    line_shape_list = ["constant", "linear", "random", "sin", "cos"]
    # line_shape_list = ["constant"]
    exclude_coherent_list = [False, True]
    shutup.please()
    for smoothing_method in smoothing_method_list:
        for snr in snr_list:
            for line_shape in line_shape_list:
                for exclude_coherent in exclude_coherent_list:
                    algorithm_parameters = AlgorithmConfiguration(snr=snr,
                                                                  line_shape=line_shape,
                                                                  exclude_coherent=exclude_coherent,
                                                                  smoothing_method=smoothing_method,
                                                                  f_rev_mode=7.5e6)
                    tune_measurement_algorithm(synchrotron_parameters, detector_parameters, algorithm_parameters)
                
