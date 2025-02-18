import xtrack as xt
import xpart as xp
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftshift
from myfunc import *

lmap = xt.LineSegmentMap(length=26658.8831999989, qx=0.27, qy=0.295, dqx=15, dqy=15, longitudinal_mode='nonlinear',
    voltage_rf=4e6, frequency_rf=400e6, lag_rf=180, momentum_compaction_factor=3.225e-04, betx=1, bety=1)
line = xt.Line(elements=[lmap])
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=450e9)
twiss = line.twiss()
f_rev = 1/twiss.T_rev0
f_sampling = np.ceil(1e6/f_rev)*f_rev
schottky_monitor = xt.monitors.SchottkyMonitor(f_rev=f_rev, schottky_harmonic=427_725, n_taylor=30)
BPM = xt.BeamPositionMonitor(frev=f_rev,
                             start_at_turn=0, stop_at_turn=20000,
                             sampling_frequency=f_sampling)
line.discard_tracker()
line.append_element(element=schottky_monitor, name='Schottky monitor')
line.append_element(element=BPM, name='Beam position monitor')
line.build_tracker()

bunch = xp.generate_matched_gaussian_bunch(num_particles=int(1e4), nemitt_x=1.5e-6, nemitt_y=1.5e-6, line=line, total_intensity_particles=1e11, sigma_z=7e-2)
line.track(bunch, num_turns=20_000, with_progress=True)
schottky_monitor.process_spectrum(inst_spectrum_len=2_000, deltaQ=0.001, band_width=0.3, Qx=0.27, Qy=0.295, x=True, y=False, z=True)
plt.figure(figsize=(12,4))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
for ax, region in zip([ax1, ax2, ax3], ['lowerH', 'center', 'upperH']):
    ax.plot(schottky_monitor.frequencies[region], schottky_monitor.PSD_avg[region])
    ax.set_xlabel(f'Frequency [$f_0$]')
    ax.set_ylabel(f'PSD [arb. units]')
    # ax.set_yscale('log')
plt.tight_layout()
plt.show()

batch_size = 32768
x_data = BPM.x_mean
# x_data = BPM.y_mean
x_data = np.nan_to_num(x_data, nan=0)
x_data_noisy = generate_noisy_signal(x_data, 0)
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

    # psd_i = sgolay_filter(psd_i, window_size, min(4, window_size - 1))
    psd_s += psd_i
    psd_p *= (normalize_to_01(psd_i) + 1)
    psd_p = normalize_to_01(psd_p) + 1
psd_s -= min(psd_s)
coef = max(psd_s)/max(psd_p)
psd_p *= coef
freqs = np.linspace(-f_sampling / 2, f_sampling / 2, N, endpoint=False)
plt.figure()
plt.plot(freqs/f_rev, psd_s)
plt.plot(freqs/f_rev, psd_p)
plt.show()
print(1)
