# src/sse_core/devices/base.py
from abc import ABC, abstractmethod


class TwoTerminalDevice(ABC):
    """
    Abstract Base Class establishing the programmatic contract for all
    stochastic charge-transfer elements in the simulator.
    """

    def __init__(self, name: str, v_th: float):
        self.name: str = name
        self.v_th: float = v_th  # Thermal voltage V_th = kT/e

    @abstractmethod
    def forward_rate(self, v_active: float, v_control: float = 0.0) -> float:
        """
        Calculate and return the forward transition rate (lambda_f) in Hz.

        Parameters:
            v_active: Voltage difference across the primary active terminals.
            v_control: Secondary control potential (e.g., Gate-Source voltage).
        """
        pass

    @abstractmethod
    def reverse_rate(self, v_active: float, v_control: float = 0.0) -> float:
        """
        Calculate and return the reverse transition rate (lambda_r) in Hz.
        """
        pass
