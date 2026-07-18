# src/sse_core/devices/semiconductor.py
import numpy as np
from numba import njit

from sse_core.compiler.units import E_CHARGE
from sse_core.devices.base import TwoTerminalDevice


# =============================================================================
# 1. Shockley Diode rate kernel
# =============================================================================
@njit(cache=True)
def diode_rates(v_active, i0, n, v_th) -> tuple[float, float]:
    qe = E_CHARGE
    # 1. Calculate the subthreshold current density
    # We use the standard exponential model for tunneling
    i_sub = i0 * np.exp(v_active / (n * v_th))

    # 2. Detailed balance:
    # lf / lr = exp(v / v_th)
    # This keeps the math stable even as V -> 0
    lf = (i_sub / qe) / (1.0 + np.exp(-v_active / v_th))
    lr = lf * np.exp(-v_active / v_th)

    return max(0.0, lf), max(0.0, lr)


# =============================================================================
# 2. Subthreshold MOSFET rate kernel
# =============================================================================
@njit(cache=True)
def mosfet_rates(
    v_ds: float, v_gs: float, i0: float, vt: float, n: float, v_th: float, is_pmos: bool
):
    qe = 1.602176634e-19  # Elementary charge

    if is_pmos:
        # PMOS physical control voltages S->G and S->D
        v_sg = -v_gs
        v_sd = -v_ds

        # PMOS dominant electron flow is Drain -> Source (Conventional S->D)
        rate_D_to_S = (i0 / qe) * np.exp((v_sg - vt) / (n * v_th))
        rate_S_to_D = rate_D_to_S * np.exp(-v_sd / v_th)

        # Return (Source->Drain, Drain->Source)
        return max(0.0, rate_S_to_D), max(0.0, rate_D_to_S)
    else:
        # NMOS physical control voltages are exactly G->S and D->S
        # NMOS dominant electron flow is Source -> Drain (Conventional D->S)
        rate_S_to_D = (i0 / qe) * np.exp((v_gs - vt) / (n * v_th))
        rate_D_to_S = rate_S_to_D * np.exp(-v_ds / v_th)

        # Return (Source->Drain, Drain->Source)
        return max(0.0, rate_S_to_D), max(0.0, rate_D_to_S)


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
