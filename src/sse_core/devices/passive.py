# src/sse_core/devices/passive.py
import numpy as np
from numba import njit

from sse_core.compiler.units import E_CHARGE
from sse_core.devices.base import TwoTerminalDevice


@njit(cache=True)
def tunnel_junction_rates(
    v_active: float, r_tunnel: float, v_th: float
) -> tuple[float, float]:
    """
    JIT-compiled rate calculation for a single tunnel junction.
    Regulates the singularity at 0V using a minor numerical offset.
    """
    qe = E_CHARGE
    # Guard against 0V singularity by shifting slightly if v_active is exactly 0
    v_reg = v_active if abs(v_active) > 1e-12 else 1e-12

    numerator = v_reg / (qe * r_tunnel)
    lf = numerator / (1.0 - np.exp(-v_reg / v_th))
    lr = numerator / (-1.0 + np.exp(v_reg / v_th))

    return lf, lr


class TunnelJunction(TwoTerminalDevice):
    """
    A passive single-electron tunnel junction with a discrete tunneling barrier.
    """

    def __init__(self, name: str, v_th: float, resistance: float):
        super().__init__(name, v_th)
        self.resistance: float = resistance

    def forward_rate(self, v_active: float, v_control: float = 0.0) -> float:
        lf, _ = tunnel_junction_rates(v_active, self.resistance, self.v_th)
        return lf

    def reverse_rate(self, v_active: float, v_control: float = 0.0) -> float:
        _, lr = tunnel_junction_rates(v_active, self.resistance, self.v_th)
        return lr
