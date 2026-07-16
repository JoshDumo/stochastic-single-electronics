# src/sse_core/solvers/gillespie.py
from typing import Any

import numpy as np
from numba import njit
from sse_core.compiler.builder import CompiledAssembly
from sse_core.compiler.parser import CircuitNetlist
from sse_core.devices.passive import TunnelJunction
from sse_core.devices.semiconductor import MOSFET, Diode


@njit(cache=True)
def select_gillespie_event(
    rates_f: np.ndarray, rates_r: np.ndarray, total_rate: float, r2: float
) -> tuple[int, bool]:
    """
    Highly optimized JIT-compiled event selection wheel algorithm.
    Slices the cumulative rate space and returns the index and type of transition.
    """
    cumulative_sum = 0.0
    n_devices = len(rates_f)

    for idx in range(n_devices):
        # Accumulate forward transition rate
        cumulative_sum += rates_f[idx]
        if cumulative_sum >= r2:
            return idx, False

        # Accumulate reverse transition rate
        cumulative_sum += rates_r[idx]
        if cumulative_sum >= r2:
            return idx, True

    # Floating point fallback safeguard
    return n_devices - 1, True


class GillespieSolver:
    """
    Stochastic single-electronics time-evolution simulator utilizing
    the Gillespie direct method for event-driven physical transitions.
    """

    def __init__(self, netlist: CircuitNetlist, assembly: CompiledAssembly):
        self.netlist = netlist
        self.assembly = assembly
        self.devices = self._instantiate_device_objects()
        self.t_finish = netlist.simulation.t_finish
        self.v_th = netlist.simulation.v_th
        self.rng = np.random.default_rng(netlist.simulation.seed)

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
                dev = MOSFET(
                    comp.name,
                    v_th,
                    comp.specs["I0"],
                    comp.specs["VT"],
                    comp.specs["n"],
                    is_pmos,
                )
            else:
                raise TypeError(f"Unhandled device class type: '{comp.type}'")
            devices.append(dev)
        return devices

    def compute_node_potentials(
        self, q: np.ndarray, vr: np.ndarray
    ) -> dict[str, float]:
        if self.assembly.free_names:
            v_free = self.assembly.C_inv @ (q - self.assembly.Cx @ vr)
        else:
            v_free = np.zeros(0)

        potentials = {}
        for idx, name in enumerate(self.assembly.free_names):
            potentials[name] = float(v_free[idx])
        for idx, name in enumerate(self.assembly.regulated_names):
            potentials[name] = float(vr[idx])

        return potentials

    def compute_all_rates(
        self, potentials: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        from sse_core.devices.mapping import extract_device_voltages

        Nd = len(self.devices)
        rates_f = np.zeros(Nd)
        rates_r = np.zeros(Nd)
        total_rate = 0.0

        for idx, dev in enumerate(self.devices):
            comp_config = [
                comp for comp in self.netlist.components if comp.name == dev.name
            ][0]
            v_act, v_ctrl = extract_device_voltages(comp_config, potentials)
            lf = dev.forward_rate(v_act, v_ctrl)
            lr = dev.reverse_rate(v_act, v_ctrl)
            rates_f[idx] = lf
            rates_r[idx] = lr
            total_rate += lf + lr

        return rates_f, rates_r, total_rate

    def execute_step(self, q: np.ndarray, vr: np.ndarray) -> tuple[np.ndarray, float]:
        potentials = self.compute_node_potentials(q, vr)
        rates_f, rates_r, total_rate = self.compute_all_rates(potentials)

        if total_rate <= 1e-15:
            return q, self.t_finish

        r1 = self.rng.uniform(1e-15, 1.0)
        tau = -np.log(r1) / total_rate

        # Draw a point on our cumulative rates wheel
        r2 = self.rng.uniform(0.0, total_rate)

        # Invoke our high-performance compiled Numba selection engine
        selected_device_idx, is_reverse_transition = select_gillespie_event(
            rates_f, rates_r, total_rate, r2
        )

        updated_q = q.copy()
        if selected_device_idx != -1:
            delta_column = self.assembly.free_Delta[:, selected_device_idx]
            if is_reverse_transition:
                updated_q -= np.round(delta_column).astype(np.int64)
            else:
                updated_q += np.round(delta_column).astype(np.int64)

        return updated_q, tau

    def simulate(
        self, initial_charge_vector: np.ndarray, vr_potentials: np.ndarray
    ) -> dict[str, Any]:
        t = 0.0
        q = initial_charge_vector.copy().astype(np.int64)
        history_t = [t]
        history_q = [q.copy()]
        current_potentials = self.compute_node_potentials(q, vr_potentials)
        history_v: dict[str, list[float]] = {
            name: [val] for name, val in current_potentials.items()
        }

        while t < self.t_finish:
            q_next, dt = self.execute_step(q, vr_potentials)
            if dt >= self.t_finish:
                t = self.t_finish
                history_t.append(t)
                history_q.append(q.copy())
                for name, val in current_potentials.items():
                    history_v[name].append(val)
                break
            t += dt
            q = q_next

            history_t.append(t)
            history_q.append(q.copy())
            current_potentials = self.compute_node_potentials(q, vr_potentials)
            for name, val in current_potentials.items():
                history_v[name].append(val)

        return {
            "time": np.array(history_t),
            "charge": np.array(history_q),
            "potentials": {name: np.array(vals) for name, vals in history_v.items()},
        }
