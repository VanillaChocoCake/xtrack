import numpy as np

length = 24.6
qx = 0.68
qy = 0.4
betx = 6
bety = 1.1
alfx = 1.359
alfy = -0.416
dx = 0.5285
dpx = -0.26
dy = 0
dpy = 0
x_ref = 0
px_ref = 0
y_ref = 0
py_ref = 0
longitudinal_mode = "linear_fixed_qs"
qs = 0.001
slip_factor = 0.33
bets = slip_factor * length / (2 * np.pi * qs)
momentum_compaction_factor = 0.3175
slippage_length = None
voltage_rf = 1.5e3
frequency_rf = 7.48e6
lag_rf = 180
dqx = -1.46
dqy = -1.34


class SynchrotronConfiguration:
    def __init__(self,
                 length=length,
                 qx=qx,
                 qy=qy,
                 betx=betx,
                 bety=bety,
                 alfx=alfx,
                 alfy=alfy,
                 dx=dx,
                 dpx=dpx,
                 dy=dy,
                 dpy=dpy,
                 x_ref=x_ref,
                 y_ref=y_ref,
                 px_ref=px_ref,
                 py_ref=py_ref,
                 longitudinal_mode=longitudinal_mode,
                 qs=qs,
                 bets=bets,
                 slip_factor=slip_factor,
                 momentum_compaction_factor=momentum_compaction_factor,
                 slippage_length=slippage_length,
                 voltage_rf=voltage_rf,
                 frequency_rf=frequency_rf,
                 lag_rf=lag_rf,
                 dqx=dqx,
                 dqy=dqy):
        self.length = length
        self.qx = qx
        self.qy = qy
        self.betx = betx
        self.bety = bety
        self.alfx = alfx
        self.alfy = alfy
        self.dx = dx
        self.dpx = dpx
        self.dy = dy
        self.dpy = dpy
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.px_ref = px_ref
        self.py_ref = py_ref
        self.longitudinal_mode = longitudinal_mode
        self.qs = qs
        self.bets = bets
        self.slip_factor = slip_factor
        self.momentum_compaction_factor = momentum_compaction_factor
        self.slippage_length = slippage_length
        self.voltage_rf = voltage_rf
        self.frequency_rf = frequency_rf
        self.lag_rf = lag_rf
        self.dqx = dqx
        self.dqy = dqy


bandwidth = 3e6  # Hz
fc = 38.5e6  # Hz


class DetectorConfiguration:
    def __init__(self, bandwidth=bandwidth, fc=fc):
        self.bandwidth = bandwidth
        self.fc = fc
        self.fl = self.fc - self.bandwidth / 2
        self.fh = self.fc + self.bandwidth / 2

sideband_width = 500e3
f_rev_increase_rate = 10e6
min_track_turns = 10
snr = -20
min_freq_res = 10e3
simulation_time = 0.001
max_len = 10
decay_factor = 0.8
line_shape = "cos"
f_rev_mode = 7.5e6
alpha = 0.45
exclude_coherent = False
schottky_harmonic = 2
interpolate_method = "cubic"
interpolate_coef = 2
outliers_len = 20
outliers_threshold_coef = 2
filter = "gaussian"

class AlgorithmConfiguration:
    def __init__(self,
                 sideband_width=sideband_width,
                 f_rev_increase_rate=f_rev_increase_rate,
                 min_track_turns=min_track_turns,
                 snr=snr,
                 min_freq_res=min_freq_res,
                 simulation_time=simulation_time,
                 max_len=max_len,
                 decay_factor=decay_factor,
                 line_shape=line_shape,
                 f_rev_mode=f_rev_mode,
                 alpha=alpha,
                 exclude_coherent=exclude_coherent,
                 schottky_harmonic=schottky_harmonic,
                 interpolate_method=interpolate_method,
                 interpolate_coef=interpolate_coef,
                 outliers_threshold_coef=outliers_threshold_coef,
                 filter=filter):
        self.sideband_width = sideband_width
        self.f_rev_increase_rate = f_rev_increase_rate
        self.min_track_turns = min_track_turns
        self.snr = snr
        self.min_freq_res = min_freq_res
        self.simulation_time = simulation_time
        self.max_len = max_len
        self.decay_factor = decay_factor
        self.line_shape = line_shape
        self.f_rev_mode = f_rev_mode
        self.alpha = alpha
        self.exclude_coherent = exclude_coherent
        self.schottky_harmonic = schottky_harmonic
        self.interpolate_method = interpolate_method
        self.interpolate_coef = interpolate_coef
        self.outliers_len = outliers_len
        self.outliers_threshold_coef = outliers_threshold_coef
        self.filter = filter