import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np
from parameters import Synchrotron_configuration
import matplotlib.pyplot as plt

context = xo.ContextCpu(omp_num_threads="auto")
config = Synchrotron_configuration()
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
f_rev = 1 / twiss.T_rev0
# f_rev = 7e6
schottky_monitor = xt.SchottkyMonitor(f_rev=f_rev, schottky_harmonic=5, n_taylor=4)
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

n_turns = int(np.power(2,  12) * 100)
"""
# stand-alone mode
for i in range(n_turns):
    schottky_monitor.track(bunch)
    line.track(bunch)
"""
line.track(bunch, num_turns=n_turns, with_progress=True)

schottky_monitor.process_spectrum(inst_spectrum_len=int(n_turns / 100), deltaQ=1e-5, band_width=5e-2,
                                  Qx=config.qx, Qy=config.qy,
                                  x=True, y=False, z=True)

plt.figure(figsize=(12, 4))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
for ax, region in zip([ax1, ax2, ax3], ['lowerH', 'center', 'upperH']):
    ax.plot(schottky_monitor.frequencies[region], schottky_monitor.PSD_avg[region])
    ax.set_xlabel(f'Frequency [$f_0$]')
    ax.set_ylabel(f'PSD [arb. units]')
    ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f"schottky_n_turns_{n_turns}.png")
plt.show()

print(1)
