from dataclasses import dataclass, field
import numpy as np

# 全局常量定义
LENGTH: float = 24.6
QX: float = 0.68
QY: float = 0.4
BETX: float = 6
BETY: float = 1.1
ALFX: float = 1.359
ALFY: float = -0.416
DX: float = 0.5285
DPX: float = -0.26
DY: float = 0
DPY: float = 0
LONGITUDINAL_MODE: str = "linear_fixed_qs"
QS: float = 0.001
MOMENTUM_COMPACTION_FACTOR: float = 0.3175
VOLTAGE_RF: float = 1.5e3
DQX: float = -1.46
DQY: float = -1.34
NEMITT: float = 2 * np.pi * 1e-6
INTENSITY: int = int(1e11)

import scipy.constants as const
def calculate_slip_factor(f_rev: float,
                          length: float = LENGTH,
                          momentum_compaction_factor: float = MOMENTUM_COMPACTION_FACTOR) -> float:
    velocity = length*f_rev
    beta = velocity/const.c
    gamma = 1/np.sqrt(1-beta**2)
    slip_factor = 1/gamma**2 - momentum_compaction_factor
    return slip_factor


def calculate_bets(slip_factor: float, length: float = LENGTH, qs: float = QS) -> float:
    return slip_factor * length / (2 * np.pi * qs)

@dataclass
class SynchrotronConfiguration:
    length: float = LENGTH
    qx: float = QX
    qy: float = QY
    betx: float = BETX
    bety: float = BETY
    alfx: float = ALFX
    alfy: float = ALFY
    dx: float = DX
    dpx: float = DPX
    dy: float = DY
    dpy: float = DPY
    longitudinal_mode: str = LONGITUDINAL_MODE
    qs: float = QS
    bets: float = None
    slip_factor: float = None
    momentum_compaction_factor: float = MOMENTUM_COMPACTION_FACTOR
    voltage_rf: float = VOLTAGE_RF
    dqx: float = DQX
    dqy: float = DQY
    nemitt: float = NEMITT
    intensity: int = INTENSITY

BANDWIDTH: float = 3e6
FC: float = 38.5e6

@dataclass
class DetectorConfiguration:
    bandwidth: float = BANDWIDTH
    fc: float = FC
    fl: float = field(init=False)
    fh: float = field(init=False)

    def __post_init__(self):
        self.fl = self.fc - self.bandwidth / 2
        self.fh = self.fc + self.bandwidth / 2

SIDE_BAND_WIDTH: float = 300e3
F_REV_INCREASE_RATE: float = 10e6
MIN_TRACK_TURNS: int = 10
SNR: float = -20
MIN_FREQ_RES: float = 5e3
SIMULATION_TIME: float = 1e-3
MAX_LEN: int = 10
DECAY_FACTOR: float = 0.8
LINE_SHAPE: str = "cos"
F_REV_MODE: float = 7.5e6
ALPHA: float = 0.4
EXCLUDE_COHERENT: bool = False
SCHOTTSKY_HARMONIC: float = 5
INTERPOLATE_METHOD: str = "cubic"
INTERPOLATE_COEF: int = 4
SMOOTHING_METHOD: str = "gaussian"

@dataclass
class AlgorithmConfiguration:
    sideband_width: float = SIDE_BAND_WIDTH
    f_rev_increase_rate: float = F_REV_INCREASE_RATE
    min_track_turns: int = MIN_TRACK_TURNS
    snr: float = SNR
    min_freq_res: float = MIN_FREQ_RES
    simulation_time: float = SIMULATION_TIME
    max_len: int = MAX_LEN
    decay_factor: float = DECAY_FACTOR
    line_shape: str = LINE_SHAPE
    f_rev_mode: float = F_REV_MODE
    alpha: float = ALPHA
    exclude_coherent: bool = EXCLUDE_COHERENT
    schottky_harmonic: float = SCHOTTSKY_HARMONIC
    interpolate_method: str = INTERPOLATE_METHOD
    interpolate_coef: int = INTERPOLATE_COEF
    smoothing_method: str = SMOOTHING_METHOD
