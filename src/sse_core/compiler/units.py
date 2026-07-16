# src/sse_core/compiler/units.py
from dataclasses import dataclass

# Physical constants in SI
E_CHARGE = 1.602176634e-19  # [Q] Elementary charge in Coulombs
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant in J/K


@dataclass
class UnitSystem:
    """
    Manages dimensionless scaling factors.
    Uses room temperature (300 K) as the default reference thermal voltage scale.
    """

    T: float = 300.0  # Reference temperature in Kelvin
    tau_scale: float = 1e-12  # Time scale [t] set to 1 ps (fast tunneling regime)

    @property
    def v_scale(self) -> float:
        """Voltage scale [V] = k_B * T / e (~25.85 mV at 300 K)"""
        return (K_BOLTZMANN * self.T) / E_CHARGE

    @property
    def c_scale(self) -> float:
        """Capacitance scale [C] = [Q] / [V]"""
        return E_CHARGE / self.v_scale

    @property
    def r_scale(self) -> float:
        """Resistance scale [R] = [V] * [t] / [Q]"""
        return (self.v_scale * self.tau_scale) / E_CHARGE

    @property
    def energy_scale(self) -> float:
        """Energy scale [E] = [Q] * [V]"""
        return E_CHARGE * self.v_scale


# Global default unit system
SI_UNITS = UnitSystem()
