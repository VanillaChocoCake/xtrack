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
fc = 35.8e6  # Hz


class DetectorConfiguration:
    def __init__(self, bandwidth=bandwidth, fc=fc):
        self.bandwidth = bandwidth
        self.fc = fc
        self.fl = self.fc - self.bandwidth / 2
        self.fh = self.fc + self.bandwidth / 2
