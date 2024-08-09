import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np
from parameters import SynchrotronConfiguration, DetectorConfiguration
import matplotlib.pyplot as plt
from tiny_function import cal_weight

context = xo.ContextCpu(omp_num_threads="auto")
config = SynchrotronConfiguration()
detector = DetectorConfiguration()
if config.longitudinal_mode == "nonlinear" or config.longitudinal_mode == "linear_fixed_rf":
    lmap = xt.LineSegmentMap(length=config.length,
                             qx=config.qx, qy=config.qy,
                             betx=config.betx, bety=config.bety,
                             alfx=config.alfx, alfy=config.alfy,
                             dx=config.dx, dy=config.dy,
                             dpx=config.dpx, dpy=config.dpy,
                             x_ref=config.x_ref, y_ref=config.y_ref,
                             px_ref=config.px_ref, py_ref=config.py_ref,
                             longitudinal_mode=config.longitudinal_mode,
                             momentum_compaction_factor=config.momentum_compaction_factor,
                             voltage_rf=config.voltage_rf, frequency_rf=config.frequency_rf, lag_rf=config.lag_rf,
                             dqx=config.dqx, dqy=config.dqy,
                             )
elif config.longitudinal_mode == "linear_fixed_qs":
    lmap = xt.LineSegmentMap(length=config.length,
                             qx=config.qx, qy=config.qy,
                             betx=config.betx, bety=config.bety,
                             alfx=config.alfx, alfy=config.alfy,
                             dx=config.dx, dy=config.dy,
                             dpx=config.dpx, dpy=config.dpy,
                             # x_ref=config.x_ref, y_ref=config.y_ref,
                             # px_ref=config.px_ref, py_ref=config.py_ref,
                             longitudinal_mode=config.longitudinal_mode,
                             qs=config.qs,
                             bets=config.bets,
                             dqx=config.dqx, dqy=config.dqy,
                             )

line = xt.Line(elements=[lmap])
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, _context=context)

# Compute the revolution period needed by the Schottky monitor

twiss = line.twiss()
# f_rev = 1 / twiss.T_rev0
f_rev = 6.5e6
schottky_harmonic = np.array([np.floor(detector.fc / f_rev), np.ceil(detector.fc / f_rev)])
slow_fc_x = (schottky_harmonic - config.qx) * f_rev
fast_fc_x = (schottky_harmonic + config.qx) * f_rev
slow_fc_y = (schottky_harmonic - config.qy) * f_rev
fast_fc_y = (schottky_harmonic + config.qy) * f_rev
for harmonic, slow, fast in zip(schottky_harmonic, slow_fc_x, fast_fc_x):
    if (detector.fl + 0.05 * detector.bandwidth < slow < detector.fh - 0.05 * detector.bandwidth
            or
            detector.fl + 0.05 * detector.bandwidth < fast < detector.fh - 0.05 * detector.bandwidth):
        schottky_harmonic = harmonic
        slow_fc_x = slow
        fast_fc_x = fast
        break
try:
    if len(schottky_harmonic) == 2:
        raise ValueError("Schottky monitor can't detect sidebands under given fc and bandwidth!")
except TypeError:
    pass



schottky_monitor = xt.SchottkyMonitor(f_rev=f_rev, schottky_harmonic=schottky_harmonic, n_taylor=4)
# BPM = xt.BeamPositionMonitor(frev=f_rev,
#                              start_at_turn=1, stop_at_turn=int(1e4),
#                              sampling_frequency=200e6)

line.discard_tracker()
line.append_element(element=schottky_monitor, name='Schottky monitor')
# line.append_element(element=BPM, name='Beam position monitor')
line.build_tracker()

bunch = xp.generate_matched_gaussian_bunch(num_particles=int(1e4),
                                           nemitt_x=1.5e-6, nemitt_y=1.5e-6,
                                           line=line,
                                           total_intensity_particles=int(1e11),
                                           sigma_z=7e-2
                                           )

n_turns = int(np.power(2, 11) * 100)
line.track(bunch, num_turns=n_turns, with_progress=True)

# In order to take the fc and bandwidth of the detector into consideration,
# Qx, Qy and band_width(in revolution frequency unit) need to be adjusted to fit fc
band_width = detector.bandwidth / f_rev
adjusted_qx = abs(detector.fc - slow_fc_x) / (schottky_harmonic * f_rev)
adjusted_qy = abs(detector.fc - slow_fc_y) / (schottky_harmonic * f_rev)

schottky_monitor.process_spectrum(inst_spectrum_len=int(n_turns / 100), deltaQ=1e-5,
                                  band_width=band_width,
                                  Qx=adjusted_qx, Qy=config.qy,
                                  x=True, y=False, z=True)
"""
schottky_monitor.process_spectrum(inst_spectrum_len=int(n_turns / 100), deltaQ=1e-5,
                                  band_width=5e-2,
                                  Qx=config.qx, Qy=config.qy,
                                  x=True, y=False, z=True)
"""

gain = 100
steepness = 5
weight = cal_weight(num=len(schottky_monitor.PSD_avg['center']), gain=gain, steepness=steepness)
plt.figure(figsize=(20, 16))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)
for ax, region in zip([ax1, ax2, ax3], ['lowerH', 'center', 'upperH']):
    PSD = np.multiply(schottky_monitor.PSD_avg[region], weight)
    ax.plot(schottky_monitor.frequencies[region], 20 * np.log10(PSD), color='r')
    ax.plot(schottky_monitor.frequencies[region], 20 * np.log10(schottky_monitor.PSD_avg[region]), color='g')
    ax.set_xlabel(f'Frequency [$f_0$]')
    ax.set_ylabel(f'PSD [arb. units]')
plt.tight_layout()
plt.savefig(f"schottky_n_turns_{n_turns}.png")
plt.show()
print(f"Original tune: {config.qx}, adjusted tune: {adjusted_qx}")
