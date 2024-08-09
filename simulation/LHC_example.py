import xtrack as xt
import xpart as xp
import matplotlib.pyplot as plt
import numpy as np
from tiny_function import cal_weight

lmap = xt.LineSegmentMap(length=26658.8831999989,
                         qx=0.27, qy=0.295,
                         dqx=15, dqy=15,
                         longitudinal_mode='nonlinear',
                         voltage_rf=[4e6], frequency_rf=[400e6], lag_rf=[180],
                         momentum_compaction_factor=3.225e-04,
                         betx=1, bety=1)
line = xt.Line(elements=[lmap])
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=450e9)
twiss = line.twiss()
schottky_monitor = xt.monitors.SchottkyMonitor(f_rev=1/twiss.T_rev0, schottky_harmonic=427_725, n_taylor=4)
line.discard_tracker()
line.append_element(element=schottky_monitor, name='Schottky monitor')
line.build_tracker()
bunch = xp.generate_matched_gaussian_bunch(
    num_particles=int(1e4),
    nemitt_x=1.5e-6, nemitt_y=1.5e-6,
    line=line,
    total_intensity_particles=1e11,
    sigma_z=7e-2)
n_turns = int(np.power(2,  11) * 100)
line.track(bunch, num_turns=n_turns, with_progress=True)
schottky_monitor.process_spectrum(inst_spectrum_len=int(n_turns / 100),
                                  deltaQ=5e-5, band_width=0.5,
                                  Qx=0.35, Qy=0.295,
                                  x=True, y=False, z=True)
"""
schottky_monitor.process_spectrum(inst_spectrum_len=int(n_turns / 100),
                                  deltaQ=5e-5, band_width=0.5,
                                  Qx=0.27, Qy=0.295,
                                  x=True, y=False, z=True)
"""
gain = 100
weight = cal_weight(num=len(schottky_monitor.PSD_avg['center']), steepness=5, gain=gain)
plt.figure(figsize=(20, 16))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)
for ax, region in zip([ax1, ax2, ax3], ['lowerH', 'center', 'upperH']):
    PSD = np.multiply(schottky_monitor.PSD_avg[region], weight)
    ax.plot(schottky_monitor.frequencies[region], 20 * np.log10(PSD) - gain, color='r')
    ax.plot(schottky_monitor.frequencies[region], 20 * np.log10(schottky_monitor.PSD_avg[region]), color='g')
    ax.set_xlabel(f'Frequency [$f_0$]')
    ax.set_ylabel(f'PSD [arb. units]')
    # ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f"schottky_LHC_n_turns_{n_turns}.png")
plt.show()
print(1)
