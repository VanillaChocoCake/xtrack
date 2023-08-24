import numpy as np
import xtrack as xt

line = xt.Line.from_json('fccee_p_ring_thin.json')

line.build_tracker()

tw_no_rad = line.twiss(method='4d')

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad_wig_off = line.twiss(eneloss_and_damping=True)

line.vars['on_wiggler_v'] = 0.1
line.compensate_radiation_energy_loss()
opt = line.match(
    solve=False,
    eneloss_and_damping=True,
    compensate_radiation_energy_loss=True,
    targets=[
        xt.Target(eq_gemitt_y=1e-12, tol=1e-15, optimize_log=True)],
    vary=xt.Vary('on_wiggler_v', step=0.01, limits=(0.1, 2))
)

opt.solve()

tw_rad = line.twiss(eneloss_and_damping=True)
