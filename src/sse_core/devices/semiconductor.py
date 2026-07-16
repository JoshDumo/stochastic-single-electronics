# src/sse_core/devices/semiconductor.py
import numpy as np
from numba import njit

from sse_core.devices.base import TwoTerminalDevice


# =============================================================================
# 1. Shockley Diode rate kernel
# =============================================================================
@njit(cache=True)
def diode_rates(
    v_active: float, i0: float, n: float, v_th: float
) -> tuple[float, float]:
    """
    JIT-compiled rate calculation for a classical Shockley diode.
    Uses a singularity guard at exactly 0V bias.
    """
    qe = 1.0
    v_reg = v_active if abs(v_active) > 1e-12 else 1e-12

    # Forward/reverse transition rates mapped to Shockley current densities
    # Rate = I / qe
    factor = i0 / qe
    lf = factor * (np.exp(v_reg / (n * v_th)) - 1.0) / (1.0 - np.exp(-v_reg / v_th))
    lr = factor * (np.exp(v_reg / (n * v_th)) - 1.0) / (-1.0 + np.exp(v_reg / v_th))

    # Guarantee rates never drop below zero due to floating point inaccuracies
    return max(0.0, lf), max(0.0, lr)


# =============================================================================
# 2. Subthreshold MOSFET rate kernel
# =============================================================================
@njit(cache=True)
def mosfet_rates(
    v_ds: float, v_gs: float, i0: float, vt: float, n: float, v_th: float, is_pmos: bool
) -> tuple[float, float]:
    """
    JIT-compiled rate calculation for subthreshold MOSFETs.
    Supports both n-channel and p-channel polarity.
    """
    qe = 1.0
    v_ds_reg = v_ds if abs(v_ds) > 1e-12 else 1e-12

    # Adjust polarity depending on channel carrier type
    v_gs_eff = -v_gs if is_pmos else v_gs
    vt_eff = -vt if is_pmos else vt
    v_ds_eff = -v_ds_reg if is_pmos else v_ds_reg

    # Subthreshold current scaling factor (thermionic emission over barrier)
    i_sub = i0 * np.exp((v_gs_eff - vt_eff) / (n * v_th))

    # Map to single-electron transitions satisfying local detailed balance
    lf = (
        (i_sub / qe)
        * (1.0 - np.exp(-v_ds_eff / v_th))
        / (1.0 - np.exp(-v_ds_eff / v_th))
    )
    lr = (
        (i_sub / qe)
        * (1.0 - np.exp(-v_ds_eff / v_th))
        / (-1.0 + np.exp(v_ds_eff / v_th))
    )

    # Fix rate limits to handle absolute drain-source directionality limits
    # Rates must be physically positive
    return max(0.0, lf), max(0.0, lr)


# =============================================================================
# 3. Object-Oriented Device Classes
# =============================================================================
class Diode(TwoTerminalDevice):
    """
    Shockley barrier diode modeled as a stochastic single-electron tunneling element.
    """

    def __init__(self, name: str, v_th: float, i0: float, n: float):
        super().__init__(name, v_th)
        self.i0: float = i0
        self.n: float = n

    def forward_rate(self, v_active: float, v_control: float = 0.0) -> float:
        lf, _ = diode_rates(v_active, self.i0, self.n, self.v_th)
        return lf

    def reverse_rate(self, v_active: float, v_control: float = 0.0) -> float:
        _, lr = diode_rates(v_active, self.i0, self.n, self.v_th)
        return lr


class MOSFET(TwoTerminalDevice):
    """
    A subthreshold MOSFET where current is regulated by gate-induced
    barrier lowering, functioning as a 3/4-terminal stochastic switch.
    """

    def __init__(
        self,
        name: str,
        v_th: float,
        i0: float,
        vt: float,
        n: float,
        is_pmos: bool = False,
    ):
        super().__init__(name, v_th)
        self.i0: float = i0
        self.vt: float = vt
        self.n: float = n
        self.is_pmos: bool = is_pmos

    def forward_rate(self, v_active: float, v_control: float = 0.0) -> float:
        """v_active -> v_ds, v_control -> v_gs"""
        lf, _ = mosfet_rates(
            v_active, v_control, self.i0, self.vt, self.n, self.v_th, self.is_pmos
        )
        return lf

    def reverse_rate(self, v_active: float, v_control: float = 0.0) -> float:
        """v_active -> v_ds, v_control -> v_gs"""
        _, lr = mosfet_rates(
            v_active, v_control, self.i0, self.vt, self.n, self.v_th, self.is_pmos
        )
        return lr
