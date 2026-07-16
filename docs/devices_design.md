# Stochastic Single Electronics (SSE) 
## Device Layer Specification
Document Version: 1.0.0

Target Audience: Circuit Designers, Simulation Engineers, and Library Developers
## 1. Introduction: The Device Layer Role
In the Stochastic Single Electronics (SSE) simulator, the Device Layer is responsible for computing the forward ($\lambda_f$) and reverse ($\lambda_r$) Poisson transition rates for discrete charge carrier jumps. Unlike traditional circuit simulation engines that evaluate continuous current-voltage ($I\text{-}V$) characteristics, the SSE backend tracks the probability-per-unit-time of single-electron transitions under fluctuating electrostatic potentials.The library enforces three strict design principles at this layer:
* Physical Consistency (Thermodynamic Gating): Every active device must satisfy the Local Detailed Balance (LDB) relation (or its physically scaled equivalent) to prevent unphysical state trajectories and ensure thermodynamic consistency.
* Terminal Encapsulation: High-level terminal assignments defined in the YAML netlist (such as $V_{gate}$, $V_{source}$, $V_{drain}$) are internally mapped to the mathematical variables required by physical rate kernels. The numerical solver is entirely decoupled from transistor polarity.
* Low-Overhead Execution: Mathematical kernels are implemented as pure functions compiled to native machine instructions using Numba's Just-In-Time (@njit) compiler.

## 2. Mathematical Foundations & Physical Rate Kernels
Active components allow electrons to jump across potential barriers. The rates of these jumps are governed by the local voltage drop across the active terminals ($V$) and, where applicable, the potentials of external control terminals ($V_{ctrl}$).
### A. Tunnel Junction
For a standard tunnel junction with tunnel resistance $R$:
#### Forward Rate:
$$\lambda_f(V) = \frac{V / (q_e R)}{1 - e^{-V/V_{th}}}$$
#### Reverse Rate:
$$\lambda_r(V) = \frac{V / (q_e R)}{e^{V/V_{th}} - 1}$$
To prevent division-by-zero singularities when $V \to 0$, a small numerical regulator ($10^{-9}\text{ V}$) is added during evaluation.
### B. MOSFET (Subthreshold Regime)
Under subthreshold conditions, the rate of charge transfer through the channel is controlled exponentially by the gate-source voltage:
#### Forward Rate:
$$\lambda_f(V_{act}, V_{ctrl}) = \lambda_0 \cdot e^{\frac{V_{ctrl}}{n V_{th}}}$$
#### Reverse Rate:
$$\lambda_r(V_{act}, V_{ctrl}) = \lambda_0 \cdot e^{\frac{V_{ctrl}}{n V_{th}}} \cdot e^{-\frac{V_{act}}{V_{th}}}$$
Where the pre-exponential scaling factor $\lambda_0$ is defined as:$$\lambda_0 = \frac{I_0}{q_e} e^{-\frac{V_T}{n V_{th}}}$$
### C. Shockley Diode
The classical diode equations are mapped to discrete carrier injection rates:
#### Forward Rate:
$$\lambda_f(V) = \frac{I_0}{q_e} e^{\frac{V}{n V_{th}}}$$
#### Reverse Rate:
$$\lambda_r(V) = \frac{I_0}{q_e}$$
## 3. Python Class Specifications
### Base Interface Definition (base.py)
All two-terminal devices must inherit from the TwoTerminalDevice abstract base class. This ensures the solver can access transition rates uniformly regardless of the underlying device physics.
```python
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
```
### Passive Devices Module (passive.py)
```python
# src/sse_core/devices/passive.py
import numpy as np
from numba import njit
from .base import TwoTerminalDevice

@njit(cache=True)
def tunnel_junction_rates(v_active: float, r_tunnel: float, v_th: float) -> tuple[float, float]:
    """
    JIT-compiled rate calculation for a single tunnel junction.
    Regulates the singularity at 0V using a minor numerical offset.
    """
    qe = 1.0
    v_reg = v_active + 1e-9  # Stabilize division near 0
    
    numerator = v_reg / (qe * r_tunnel)
    lf = numerator / (1.0 - np.exp(-v_reg / v_th))
    lr = numerator / (-1.0 + np.exp(v_reg / v_th))
    
    return lf, lr

class TunnelJunction(TwoTerminalDevice):
    def __init__(self, name: str, v_th: float, resistance: float):
        super().__init__(name, v_th)
        self.resistance: float = resistance

    def forward_rate(self, v_active: float, v_control: float = 0.0) -> float:
        lf, _ = tunnel_junction_rates(v_active, self.resistance, self.v_th)
        return lf

    def reverse_rate(self, v_active: float, v_control: float = 0.0) -> float:
        _, lr = tunnel_junction_rates(v_active, self.resistance, self.v_th)
        return lr
```
### Semiconductor Devices Module (semiconductor.py)
```python
# src/sse_core/devices/semiconductor.py
import numpy as np
from numba import njit
from .base import TwoTerminalDevice

@njit(cache=True)
def mosfet_rate_kernel(lfactor: float, v_ctrl: float, v_act: float, n: float, v_th: float, is_reverse: bool) -> float:
    """
    Shared compiled computational kernel for subthreshold MOSFET stochastic operations.
    """
    if not is_reverse:
        return lfactor * np.exp(v_ctrl / (n * v_th))
    else:
        return lfactor * np.exp(v_ctrl / (n * v_th)) * np.exp(-v_act / v_th)

@njit(cache=True)
def diode_rate_kernel(lfactor: float, v_act: float, n: float, v_th: float, is_reverse: bool) -> float:
    """
    Shared compiled computational kernel for Shockley diode stochastic operations.
    """
    if not is_reverse:
        return lfactor * np.exp(v_act / (n * v_th))
    else:
        return lfactor

# =============================================================================
# MOSFET Devices
# =============================================================================
class MOSFET(TwoTerminalDevice):
    """
    Base class containing common physics-level MOSFET calculations.
    """
    def __init__(self, name: str, v_th: float, I0: float, VT: float, n: float):
        super().__init__(name, v_th)
        self.I0: float = I0
        self.VT: float = VT
        self.n: float = n
        self.qe: float = 1.0
        # Pre-calculated constant factor to minimize run-time exponent computations
        self.lfactor: float = (self.I0 / self.qe) * np.exp(-self.VT / (self.n * self.v_th))

class NChannelMOSFET(MOSFET):
    """
    nMOS implementation enforcing active and control voltage polarity mapping:
    - V_active  = V_drain - V_source (V_ds)
    - V_control = V_gate - V_source  (V_gs)
    """
    def forward_rate(self, v_active: float, v_control: float = 0.0) -> float:
        return mosfet_rate_kernel(self.lfactor, v_control, v_active, self.n, self.v_th, False)

    def reverse_rate(self, v_active: float, v_control: float = 0.0) -> float:
        return mosfet_rate_kernel(self.lfactor, v_control, v_active, self.n, self.v_th, True)

class PChannelMOSFET(MOSFET):
    """
    pMOS implementation mapping inverted terminal potentials:
    - V_active  = V_source - V_drain (V_sd)
    - V_control = V_source - V_gate  (V_sg)
    """
    def forward_rate(self, v_active: float, v_control: float = 0.0) -> float:
        # Sign mapping occurs at device boundary; solver sees identical kernel
        return mosfet_rate_kernel(self.lfactor, v_control, v_active, self.n, self.v_th, False)

    def reverse_rate(self, v_active: float, v_control: float = 0.0) -> float:
        return mosfet_rate_kernel(self.lfactor, v_control, v_active, self.n, self.v_th, True)

# =============================================================================
# Diode Devices
# =============================================================================
class ShockleyDiode(TwoTerminalDevice):
    """
    Shockley Diode component wrapping diode_rate_kernel execution.
    """
    def __init__(self, name: str, v_th: float, I0: float, n: float):
        super().__init__(name, v_th)
        self.I0: float = I0
        self.n: float = n
        self.qe: float = 1.0
        self.lfactor: float = self.I0 / self.qe

    def forward_rate(self, v_active: float, v_control: float = 0.0) -> float:
        return diode_rate_kernel(self.lfactor, v_active, self.n, self.v_th, False)

    def reverse_rate(self, v_active: float, v_control: float = 0.0) -> float:
        return diode_rate_kernel(self.lfactor, v_active, self.n, self.v_th, True)
```
## 4. Verification Test Cases: Physics-Informed Unit Tests
To guarantee the physical validity of the simulation backend, we enforce verification of the Local Detailed Balance (LDB) relation. These checks run automatically inside our test suite across a sweep of temperatures and voltages.
```python
# tests/test_devices/test_physics.py
import pytest
import numpy as np
from sse_core.devices.passive import TunnelJunction
from sse_core.devices.semiconductor import NChannelMOSFET, ShockleyDiode

@pytest.mark.parametrize("v_th", [0.0259, 0.0350])  # Test at different temperatures
@pytest.mark.parametrize("v_act", [-0.15, -0.05, 0.05, 0.15])
def test_tunnel_junction_ldb(v_th, v_act):
    """
    Verify that the Tunnel Junction rate calculations satisfy standard LDB:
    ln( lf(V) / lr(-V) ) = V / V_th
    """
    tj = TunnelJunction(name="TJ_Test", v_th=v_th, resistance=1.0e5)
    
    lf = tj.forward_rate(v_active=v_act)
    lr = tj.reverse_rate(v_active=-v_act)
    
    calculated_ratio = np.log(lf / lr)
    expected_ratio = v_act / v_th
    
    assert calculated_ratio == pytest.approx(expected_ratio, rel=1e-6)

@pytest.mark.parametrize("v_th", [0.0259])
@pytest.mark.parametrize("v_act", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("v_ctrl", [0.1, 0.3, 0.5])
def test_mosfet_ldb(v_th, v_act, v_ctrl):
    """
    Verify that the MOSFET rates satisfy local detailed balance:
    ln( lf(V_act, V_ctrl) / lr(-V_act, V_ctrl) ) = V_act / V_th
    """
    mos = NChannelMOSFET(name="nMOS_Test", v_th=v_th, I0=1e-6, VT=0.4, n=1.1)
    
    lf = mos.forward_rate(v_active=v_act, v_control=v_ctrl)
    lr = mos.reverse_rate(v_active=-v_act, v_control=v_ctrl)
    
    calculated_ratio = np.log(lf / lr)
    expected_ratio = v_act / v_th
    
    assert calculated_ratio == pytest.approx(expected_ratio, rel=1e-6)

@pytest.mark.parametrize("v_th", [0.0259])
@pytest.mark.parametrize("v_act", [0.02, 0.06, 0.1])
def test_diode_scaled_ldb(v_th, v_act):
    """
    Verify that the Shockley Diode rates satisfy scaled LDB incorporating the ideality factor 'n':
    ln( lf(V) / lr(-V) ) = V / (n * V_th)
    """
    n_factor = 1.3
    diode = ShockleyDiode(name="Diode_Test", v_th=v_th, I0=1e-6, n=n_factor)
    
    lf = diode.forward_rate(v_active=v_act)
    lr = diode.reverse_rate(v_active=-v_act)
    
    calculated_ratio = np.log(lf / lr)
    expected_ratio = v_act / (n_factor * v_th)
    
    assert calculated_ratio == pytest.approx(expected_ratio, rel=1e-6)
```

