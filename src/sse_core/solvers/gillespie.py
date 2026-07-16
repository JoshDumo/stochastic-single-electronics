# src/sse_core/solvers/gillespie.py
from typing import Any

import numpy as np
from sse_core.compiler.builder import CompiledAssembly
from sse_core.compiler.parser import CircuitNetlist
from sse_core.devices.passive import TunnelJunction
from sse_core.devices.semiconductor import MOSFET, Diode


class GillespieSolver:
    """
    Stochastic single-electronics time-evolution simulator utilizing
    the Gillespie direct method for event-driven physical transitions.
    """

    def __init__(self, netlist: CircuitNetlist, assembly: CompiledAssembly):
        self.netlist = netlist
        self.assembly = assembly

        # Instantiate runtime-equivalent physical device classes
        self.devices = self._instantiate_device_objects()

        # Simulation state
        self.t_finish = netlist.simulation.t_finish
        self.v_th = netlist.simulation.v_th
        self.rng = np.random.default_rng(netlist.simulation.seed)

    def _instantiate_device_objects(self):
        """Maps declarative component models to concrete rate-computing objects."""
        devices = []
        # Filter active components (exclude capacitors)
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
        """
        Maps the charge vector q (Nf,) and regulated potential vector vr (Nr,)
        to the absolute physical voltage at each node.

        Formula: V = C_inv @ (q - Cx @ vr)
        """
        if self.assembly.free_names:
            # V_free = C_inv * (q - Cx * Vr)
            v_free = self.assembly.C_inv @ (q - self.assembly.Cx @ vr)
        else:
            v_free = np.zeros(0)

        # Build comprehensive mapping dictionary
        potentials = {}
        for idx, name in enumerate(self.assembly.free_names):
            potentials[name] = float(v_free[idx])
        for idx, name in enumerate(self.assembly.regulated_names):
            potentials[name] = float(vr[idx])

        return potentials

    def compute_all_rates(
        self, potentials: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Extracts terminal voltages and computes the forward/reverse rate arrays
        for all active devices in the current system state.

        Returns:
            tuple (rates_f, rates_r, total_rate)
        """
        from sse_core.devices.mapping import extract_device_voltages

        Nd = len(self.devices)
        rates_f = np.zeros(Nd)
        rates_r = np.zeros(Nd)
        total_rate = 0.0

        for idx, dev in enumerate(self.devices):
            # Locate corresponding config to map terminals
            comp_config = [
                comp for comp in self.netlist.components if comp.name == dev.name
            ][0]

            # Extract actual voltages mapped by terminal names
            v_act, v_ctrl = extract_device_voltages(comp_config, potentials)

            lf = dev.forward_rate(v_act, v_ctrl)
            lr = dev.reverse_rate(v_act, v_ctrl)

            rates_f[idx] = lf
            rates_r[idx] = lr
            total_rate += lf + lr

        return rates_f, rates_r, total_rate

    def execute_step(self, q: np.ndarray, vr: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Executes a single Gillespie step. Calculates rates, draws transition time,
        selects the firing channel, and applies the discrete state jump.

        Returns:
            tuple (updated_q, delta_t)
        """
        # 1. Compute potentials and rate landscape
        potentials = self.compute_node_potentials(q, vr)
        rates_f, rates_r, total_rate = self.compute_all_rates(potentials)

        # 2. Check for thermodynamic freeze-out (all rates are zero)
        if total_rate <= 1e-15:
            # Advance simulation to finish since no tunnelings are active
            return q, self.t_finish

        # 3. Draw time step (exponential distribution)
        # r1 is pulled from our seeded RNG
        r1 = self.rng.uniform(1e-15, 1.0)
        tau = -np.log(r1) / total_rate

        # 4. Choose which transition channel fires
        # We partition the rates interval: [0, total_rate]
        # and see where our uniform scale point r2 lands.
        r2 = self.rng.uniform(0.0, total_rate)

        cumulative_sum = 0.0
        selected_device_idx = -1
        is_reverse_transition = False

        for idx in range(len(self.devices)):
            # Check forward transition boundary
            cumulative_sum += rates_f[idx]
            if cumulative_sum >= r2:
                selected_device_idx = idx
                is_reverse_transition = False
                break

            # Check reverse transition boundary
            cumulative_sum += rates_r[idx]
            if cumulative_sum >= r2:
                selected_device_idx = idx
                is_reverse_transition = True
                break

        # 5. Apply the selected state transition vector using our compiled incidence matrix
        updated_q = q.copy()
        if selected_device_idx != -1:
            delta_column = self.assembly.free_Delta[:, selected_device_idx]
            if is_reverse_transition:
                # Reverse transition subtracts charge from target node A, adds to source B
                updated_q -= np.round(delta_column).astype(np.int64)
            else:
                updated_q += np.round(delta_column).astype(np.int64)

        return updated_q, tau

    def simulate(
        self, initial_charge_vector: np.ndarray, vr_potentials: np.ndarray
    ) -> dict[str, Any]:
        """
        Executes the full stochastic time-evolution loop from t = 0 to t_finish.

        Parameters:
            initial_charge_vector: Starting excess charge array (Nf,)
            vr_potentials: Potentials on regulated nodes (Nr,)

        Returns:
            A dictionary containing recorded arrays:
                - "time": (M,) Array of transition timestamps.
                - "charge": (M, Nf) Array of free node excess charge histories.
                - "potentials": (M, N) Dict mapping node names to potential histories.
        """
        # 1. Initialize time and state buffers
        t = 0.0
        q = initial_charge_vector.copy().astype(np.int64)

        # Pre-allocate history structures using dynamic lists
        history_t = [t]
        history_q = [q.copy()]

        current_potentials = self.compute_node_potentials(q, vr_potentials)
        history_v: dict[str, list[float]] = {
            name: [val] for name, val in current_potentials.items()
        }

        # 2. Main Simulation Loop
        while t < self.t_finish:
            # Execute step to calculate transition rates and pull next event
            q_next, dt = self.execute_step(q, vr_potentials)

            # If a physical freeze-out occurred, break early
            if dt >= self.t_finish:
                t = self.t_finish
                history_t.append(t)
                history_q.append(q.copy())
                for name, val in current_potentials.items():
                    history_v[name].append(val)
                break

            t += dt
            q = q_next

            # Record transition snapshots
            history_t.append(t)
            history_q.append(q.copy())

            current_potentials = self.compute_node_potentials(q, vr_potentials)
            for name, val in current_potentials.items():
                history_v[name].append(val)

        # 3. Package and cast records to high-performance NumPy arrays
        return {
            "time": np.array(history_t),
            "charge": np.array(history_q),
            "potentials": {name: np.array(vals) for name, vals in history_v.items()},
        }
