import random
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
import xpart as xp
import xobjects as xo
from parameters import SynchrotronConfiguration, calculate_slip_factor, calculate_bets
from utils import *
import shutup
import pickle
import scipy.constants as sc

shutup.please()
# np.random.seed(114514)
# random.seed(114514)

def Schottky_spectra(synchrotron_parameters: SynchrotronConfiguration,
                               frev: float,
                               schottky_harmonic: int=5,
                               n_turns: int=3000,
                               band_width: float=0.05,
                               deltaQ: float=1e-4) -> None:
    context = xo.ContextCpu(omp_num_threads="auto")
    qx = synchrotron_parameters.qx
    qy = synchrotron_parameters.qy
    synchrotron_parameters.slip_factor = calculate_slip_factor(frev)
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
    line = xt.Line(elements=[lmap])
    line.discard_tracker()
    schottky_monitor = xt.SchottkyMonitor(f_rev=frev, schottky_harmonic=np.ceil(schottky_harmonic), n_taylor=32)
    line.append_element(element=schottky_monitor, name="SchottkyMonitor")
    line.build_tracker()

    beta = synchrotron_parameters.length * frev / sc.c
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
    line.track(bunch, num_turns=n_turns)
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
        ax.plot(schottky_monitor.frequencies[region], schottky_monitor.PSD_avg[region], color='b')
        ax.set_xlabel(f'Frequency [$f_0$]')
        ax.set_ylabel(f'PSD [arb. units]')
        # ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f"schottky_n_turns_{n_turns}.png")
    plt.show()

if __name__ == "__main__":
    frev = 7.5e6
    Schottky_spectra(SynchrotronConfiguration(), frev, 5, int(1e-3*frev), 0.05, 1e-4)