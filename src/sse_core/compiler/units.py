# src/sse_core/compiler/units.py
from dataclasses import dataclass

# Absolute physical constants
E_CHARGE = 1.602176634e-19  # Elementary charge [C]
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant [J/K]


@dataclass
class UnitSystem:
    T: float = 300.0  # Reference temperature [Kelvin]
    tau_scale: float = 1e-12  # Reference time scale [seconds] (1 ps)

    @property
    def v_scale(self) -> float:
        """Voltage Scale [V] = k_B * T / e (~25.85 mV at 300 K)"""
        return (K_BOLTZMANN * self.T) / E_CHARGE

    @property
    def c_scale(self) -> float:
        """Capacitance Scale [C] = [Q] / [V]"""
        return E_CHARGE / self.v_scale

    @property
    def r_scale(self) -> float:
        """Resistance Scale [R] = [V] * [t] / [Q]"""
        return (self.v_scale * self.tau_scale) / E_CHARGE

    @property
    def current_scale(self) -> float:
        """Current Scale [I] = [Q] / [t]"""
        return E_CHARGE / self.tau_scale

    @property
    def energy_scale(self) -> float:
        """Energy Scale [E] = [Q] * [V] = k_B * T"""
        return E_CHARGE * self.v_scale


# Global package-wide unit registry
SI_UNITS = UnitSystem()
