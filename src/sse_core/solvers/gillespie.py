# src/sse_core/solvers/gillespie.py
from typing import Any

import numpy as np
from numba import njit
from sse_core.compiler.builder import CompiledAssembly
from sse_core.compiler.parser import CircuitNetlist
from sse_core.compiler.units import E_CHARGE
from sse_core.devices.mapping import extract_device_voltages
from sse_core.devices.passive import TunnelJunction
from sse_core.devices.semiconductor import MOSFET, Diode


@njit(cache=True)
def select_gillespie_event(
    rates_f: np.ndarray, rates_r: np.ndarray, total_rate: float, r: float
) -> tuple[int, bool]:
    """
    JIT-compiled event selection for the Gillespie algorithm.
    """
    cumulative = 0.0
    for idx in range(len(rates_f)):
        cumulative += rates_f[idx]
        if cumulative >= r:
            return idx, False
        cumulative += rates_r[idx]
        if cumulative >= r:
            return idx, True
    return len(rates_f) - 1, True


class GillespieSolver:
    """
    Stochastic single-electronics time-evolution simulator operating
    natively in the SI domain (Volts, Joules, Amperes, Seconds).
    """

    def __init__(self, netlist: CircuitNetlist, assembly: CompiledAssembly):
        self.netlist = netlist
        self.assembly = assembly
        self.devices = self._instantiate_device_objects()
        self.t_finish = netlist.simulation.t_finish
        self.rng = np.random.default_rng(netlist.simulation.seed)
        self.v_th = netlist.simulation.v_th

    def _instantiate_device_objects(self):
        devices = []
        active_comps = [
            comp
            for comp in self.netlist.components
            if comp.type
            in ["tunnel_junction", "n_channel_mosfet", "p_channel_mosfet", "diode"]
        ]
        for comp in active_comps:
            v_th = self.netlist.simulation.v_th
            if comp.type == "tunnel_junction":
                dev = TunnelJunction(comp.name, v_th, comp.specs["resistance"])
            elif comp.type == "diode":
                dev = Diode(comp.name, v_th, comp.specs["I0"], comp.specs["n"])
            elif comp.type in ["n_channel_mosfet", "p_channel_mosfet"]:
                is_pmos = comp.type == "p_channel_mosfet"
                # is_pmos = comp.type == "n_channel_mosfet"
                dev = MOSFET(
                    comp.name,
                    v_th,
                    comp.specs["I0"],
                    comp.specs["VT"],
                    comp.specs["n"],
                    is_pmos,
                )
            devices.append(dev)
        return devices

    def compute_node_potentials(
        self, q: np.ndarray, vr: np.ndarray
    ) -> dict[str, float]:
        if self.assembly.free_names:
            # CRITICAL FIX: Electrons have NEGATIVE physical charge
            q_si = q.astype(np.float64) * (-E_CHARGE)
            v_free = self.assembly.C_inv @ (q_si - self.assembly.Cx @ vr)
        else:
            v_free = np.zeros(0)

        potentials = {
            name: float(v_free[i]) for i, name in enumerate(self.assembly.free_names)
        }
        potentials.update(
            {name: float(vr[i]) for i, name in enumerate(self.assembly.regulated_names)}
        )
        return potentials

    def compute_electrostatic_energy(self, q: np.ndarray, vr: np.ndarray) -> float:
        # CRITICAL FIX: Electrons have NEGATIVE physical charge
        q_si = q.astype(np.float64) * (-E_CHARGE)

        term1 = 0.5 * float(q_si.T @ self.assembly.C_inv @ q_si)
        term2 = -float(q_si.T @ self.assembly.C_inv @ self.assembly.Cx @ vr)

        cx_t_c_inv_cx = self.assembly.Cx.T @ self.assembly.C_inv @ self.assembly.Cx
        schur_cap = self.assembly.Cr - cx_t_c_inv_cx
        term3 = 0.5 * float(vr.T @ schur_cap @ vr)

        return term1 + term2 + term3

    def compute_all_rates(
        self, potentials: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        Nd = len(self.devices)
        rates_f, rates_r = np.zeros(Nd), np.zeros(Nd)
        total_rate = 0.0

        for idx, dev in enumerate(self.devices):
            comp_config = [c for c in self.netlist.components if c.name == dev.name][0]
            v_act, v_ctrl = extract_device_voltages(comp_config, potentials)
            lf = dev.forward_rate(v_act, v_ctrl)
            lr = dev.reverse_rate(v_act, v_ctrl)
            rates_f[idx], rates_r[idx] = lf, lr
            total_rate += lf + lr

        return rates_f, rates_r, total_rate

    def execute_step(self, q: np.ndarray, vr: np.ndarray) -> tuple[np.ndarray, float]:
        potentials = self.compute_node_potentials(q, vr)
        rates_f, rates_r, total_rate = self.compute_all_rates(potentials)

        # If rates are functionally zero, advance to end of simulation
        if total_rate <= 1e-20:
            return q, self.t_finish

        dt = -np.log(self.rng.uniform(1e-15, 1.0)) / total_rate
        r = self.rng.uniform(0.0, total_rate)

        idx, is_rev = select_gillespie_event(rates_f, rates_r, total_rate, r)

        delta = self.assembly.free_Delta[:, idx]

        # Ensure delta is integer-rounded to maintain strict charge quantization
        new_q = q.copy() + ((-1 if is_rev else 1) * np.round(delta).astype(np.int64))
        return new_q, dt

    def simulate(
        self, q_init: np.ndarray, vr: np.ndarray, max_steps: int = 200000
    ) -> dict[str, Any]:
        q = q_init.astype(np.int64)
        t = 0.0

        history_q = [q.copy()]
        history_t = [t]
        history_e = [self.compute_electrostatic_energy(q, vr)]
        history_v = {
            name: [val] for name, val in self.compute_node_potentials(q, vr).items()
        }

        for _ in range(max_steps):
            if t >= self.t_finish:
                break

            q, dt = self.execute_step(q, vr)
            t += dt

            history_t.append(min(t, self.t_finish))
            history_q.append(q.copy())
            history_e.append(self.compute_electrostatic_energy(q, vr))
            for name, val in self.compute_node_potentials(q, vr).items():
                history_v[name].append(val)

        return {
            "time": np.array(history_t),
            "charge": np.array(history_q),
            "energy": np.array(history_e),
            "potentials": {n: np.array(v) for n, v in history_v.items()},
        }
