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

    def __init__(
        self,
        netlist: CircuitNetlist,
        assembly: CompiledAssembly,
    ):
        self.netlist = netlist
        self.assembly = assembly

        # Keep one canonical ordered list of active components.
        #
        # This order must match the order used by SSEMatrixBuilder when it
        # constructs free_Delta and device_terminals.
        self.active_components = [
            component
            for component in self.netlist.components
            if component.type
            in [
                "tunnel_junction",
                "n_channel_mosfet",
                "p_channel_mosfet",
                "diode",
            ]
        ]

        self.devices = self._instantiate_device_objects()

        self.t_finish = netlist.simulation.t_finish
        self.rng = np.random.default_rng(netlist.simulation.seed)
        self.v_th = netlist.simulation.v_th

        if len(self.active_components) != len(self.devices):
            raise RuntimeError(
                "Active-component and instantiated-device counts differ."
            )

        if len(self.active_components) != self.assembly.free_Delta.shape[1]:
            raise RuntimeError(
                "The solver active-component order does not match the "
                "compiled incidence matrix."
            )

        if len(self.active_components) != len(self.assembly.device_terminals):
            raise RuntimeError(
                "The solver active-component order does not match the "
                "compiled device-terminal list."
            )

    def _instantiate_device_objects(self):
        """
        Instantiate device models in exactly the same order as the compiled
        incidence and terminal arrays.
        """

        devices = []

        for component in self.active_components:
            v_th = self.netlist.simulation.v_th

            if component.type == "tunnel_junction":
                device = TunnelJunction(
                    component.name,
                    v_th,
                    component.specs["resistance"],
                )

            elif component.type == "diode":
                device = Diode(
                    component.name,
                    v_th,
                    component.specs["I0"],
                    component.specs["n"],
                )

            elif component.type in [
                "n_channel_mosfet",
                "p_channel_mosfet",
            ]:
                is_pmos = component.type == "p_channel_mosfet"

                device = MOSFET(
                    component.name,
                    v_th,
                    component.specs["I0"],
                    component.specs["VT"],
                    component.specs["n"],
                    is_pmos,
                )

            else:
                raise ValueError(
                    f"Unsupported active component type '{component.type}'."
                )

            devices.append(device)

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

    def compute_electrostatic_energy(
        self,
        q: np.ndarray,
        vr: np.ndarray,
    ) -> float:
        """
        Return the electrostatic field energy E(q, Vr) in joules.

        The simulator state q stores integer excess-electron counts.
        Physical free-node charge is therefore

            q_physical = -e * q.

        For the partitioned Maxwell capacitance matrix

            M = [[C,  Cx],
                [Cx.T, Cr]],

        the electrostatic energy at fixed regulated-node voltages is

            E = 1/2 q_physical.T C^-1 q_physical
                + 1/2 Vr.T
                    (Cr - Cx.T C^-1 Cx)
                Vr.

        There is no free-charge/regulator-voltage cross term in E.
        """

        q_count = np.asarray(q, dtype=np.float64)
        vr = np.asarray(vr, dtype=np.float64)

        if q_count.shape != (len(self.assembly.free_names),):
            raise ValueError(
                "Charge vector has shape "
                f"{q_count.shape}; expected "
                f"{(len(self.assembly.free_names),)}."
            )

        if vr.shape != (len(self.assembly.regulated_names),):
            raise ValueError(
                "Regulated-voltage vector has shape "
                f"{vr.shape}; expected "
                f"{(len(self.assembly.regulated_names),)}."
            )

        q_physical = -E_CHARGE * q_count

        free_energy = 0.5 * float(q_physical.T @ self.assembly.C_inv @ q_physical)

        schur_capacitance = (
            self.assembly.Cr
            - self.assembly.Cx.T @ self.assembly.C_inv @ self.assembly.Cx
        )

        regulated_energy = 0.5 * float(vr.T @ schur_capacitance @ vr)

        return free_energy + regulated_energy

    def compute_open_circuit_potential(
        self,
        q: np.ndarray,
        vr: np.ndarray,
    ) -> float:
        """
        Return the open-circuit potential Phi(q, Vr) in joules.

        Phi differs from the electrostatic field energy E because the
        regulated voltage sources perform work while maintaining their
        node voltages:

            Phi = E
                - Vr.T Cx.T C^-1 q_physical.

        Its gradient with respect to physical free-node charge gives the
        free-node voltage:

            grad_q Phi = C^-1 (q_physical - Cx Vr).

        Phi will later be used for transition energetics and local
        detailed-balance checks. It must not replace E in the stored
        electrostatic-energy trajectory.
        """

        q_count = np.asarray(q, dtype=np.float64)
        vr = np.asarray(vr, dtype=np.float64)

        q_physical = -E_CHARGE * q_count

        electrostatic_energy = self.compute_electrostatic_energy(
            q_count,
            vr,
        )

        source_coupling = float(
            vr.T @ self.assembly.Cx.T @ self.assembly.C_inv @ q_physical
        )

        return electrostatic_energy - source_coupling

    def compute_all_rates(
        self,
        potentials: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Evaluate all forward and reverse transition rates for one state.

        The returned arrays use the canonical active-component ordering
        stored in self.active_components.
        """

        number_of_devices = len(self.devices)

        rates_forward = np.zeros(
            number_of_devices,
            dtype=np.float64,
        )

        rates_reverse = np.zeros(
            number_of_devices,
            dtype=np.float64,
        )

        total_rate = 0.0

        for device_index, (device, component) in enumerate(
            zip(self.devices, self.active_components)
        ):
            v_active, v_control = extract_device_voltages(
                component,
                potentials,
            )

            forward_rate = float(device.forward_rate(v_active, v_control))

            reverse_rate = float(device.reverse_rate(v_active, v_control))

            if not np.isfinite(forward_rate) or forward_rate < 0.0:
                raise RuntimeError(
                    f"Device '{component.name}' produced invalid "
                    f"forward rate {forward_rate}."
                )

            if not np.isfinite(reverse_rate) or reverse_rate < 0.0:
                raise RuntimeError(
                    f"Device '{component.name}' produced invalid "
                    f"reverse rate {reverse_rate}."
                )

            rates_forward[device_index] = forward_rate
            rates_reverse[device_index] = reverse_rate

            total_rate += forward_rate + reverse_rate

        if not np.isfinite(total_rate):
            raise RuntimeError(f"Total Gillespie rate is not finite: {total_rate}.")

        return rates_forward, rates_reverse, total_rate

    def compute_embedded_rates(
        self,
        q: np.ndarray,
        vr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Evaluate thermodynamically consistent state-dependent device rates.

        Each direction has its own candidate final state:

            forward: q -> q + Delta
            reverse: q -> q - Delta

        The fixed-voltage device rate is evaluated using the average of the
        relevant terminal voltages before and after that particular
        transition.

        For MOS devices, both the active drain-source voltage and the
        gate-source control voltage are midpoint evaluated.
        """

        q = np.asarray(q, dtype=np.int64)
        vr = np.asarray(vr, dtype=np.float64)

        expected_q_shape = (len(self.assembly.free_names),)

        expected_vr_shape = (len(self.assembly.regulated_names),)

        if q.shape != expected_q_shape:
            raise ValueError(
                f"Charge state has shape {q.shape}; expected {expected_q_shape}."
            )

        if vr.shape != expected_vr_shape:
            raise ValueError(
                f"Regulated-voltage vector has shape {vr.shape}; "
                f"expected {expected_vr_shape}."
            )

        number_of_devices = len(self.devices)

        rates_forward = np.zeros(
            number_of_devices,
            dtype=np.float64,
        )

        rates_reverse = np.zeros(
            number_of_devices,
            dtype=np.float64,
        )

        potentials_before = self.compute_node_potentials(
            q,
            vr,
        )

        total_rate = 0.0

        for device_index, (device, component) in enumerate(
            zip(
                self.devices,
                self.active_components,
            )
        ):
            free_delta = np.rint(
                self.assembly.free_Delta[
                    :,
                    device_index,
                ]
            ).astype(np.int64)

            q_after_forward = q + free_delta
            q_after_reverse = q - free_delta

            potentials_after_forward = self.compute_node_potentials(
                q_after_forward,
                vr,
            )

            potentials_after_reverse = self.compute_node_potentials(
                q_after_reverse,
                vr,
            )

            (
                v_active_before,
                v_control_before,
            ) = extract_device_voltages(
                component,
                potentials_before,
            )

            (
                v_active_after_forward,
                v_control_after_forward,
            ) = extract_device_voltages(
                component,
                potentials_after_forward,
            )

            (
                v_active_after_reverse,
                v_control_after_reverse,
            ) = extract_device_voltages(
                component,
                potentials_after_reverse,
            )

            v_active_forward_midpoint = 0.5 * (v_active_before + v_active_after_forward)

            v_control_forward_midpoint = 0.5 * (
                v_control_before + v_control_after_forward
            )

            v_active_reverse_midpoint = 0.5 * (v_active_before + v_active_after_reverse)

            v_control_reverse_midpoint = 0.5 * (
                v_control_before + v_control_after_reverse
            )

            forward_rate = float(
                device.forward_rate(
                    v_active_forward_midpoint,
                    v_control_forward_midpoint,
                )
            )

            reverse_rate = float(
                device.reverse_rate(
                    v_active_reverse_midpoint,
                    v_control_reverse_midpoint,
                )
            )

            if not np.isfinite(forward_rate) or forward_rate < 0.0:
                raise RuntimeError(
                    f"Device '{component.name}' produced invalid "
                    f"embedded forward rate {forward_rate}."
                )

            if not np.isfinite(reverse_rate) or reverse_rate < 0.0:
                raise RuntimeError(
                    f"Device '{component.name}' produced invalid "
                    f"embedded reverse rate {reverse_rate}."
                )

            rates_forward[device_index] = forward_rate

            rates_reverse[device_index] = reverse_rate

            total_rate += forward_rate + reverse_rate

        if not np.isfinite(total_rate):
            raise RuntimeError(
                f"Total embedded Gillespie rate is not finite: {total_rate}."
            )

        return (
            rates_forward,
            rates_reverse,
            total_rate,
        )

    def _sample_event(
        self,
        q: np.ndarray,
        vr: np.ndarray,
    ) -> tuple[np.ndarray, float, dict[str, Any]] | None:
        """
        Sample one Gillespie event and return its complete physical ledger
        record.

        This method does not decide whether the sampled event lies beyond
        t_finish. That decision is made by simulate().

        The compiled native forward direction is defined by free_Delta:

            +1 at the component drain/first terminal
            -1 at the component source/second terminal.

        Reverse events use the opposite incidence vector.
        """

        q_before = np.asarray(q, dtype=np.int64).copy()

        potentials_before = self.compute_node_potentials(
            q_before,
            vr,
        )

        (
            rates_forward,
            rates_reverse,
            total_rate,
        ) = self.compute_embedded_rates(
            q_before,
            vr,
        )

        if total_rate <= 1.0e-20:
            return None

        waiting_time = -np.log(self.rng.uniform(1.0e-15, 1.0)) / total_rate

        rate_selector = self.rng.uniform(
            0.0,
            total_rate,
        )

        device_index, is_reverse = select_gillespie_event(
            rates_forward,
            rates_reverse,
            total_rate,
            rate_selector,
        )

        # Forward uses the compiled incidence vector.
        # Reverse uses its negative.
        direction = -1 if is_reverse else 1

        compiled_free_delta = direction * np.round(
            self.assembly.free_Delta[:, device_index]
        ).astype(np.int64)

        drain_index, source_index = self.assembly.device_terminals[device_index]

        full_delta = np.zeros(
            len(self.assembly.free_names) + len(self.assembly.regulated_names),
            dtype=np.int64,
        )

        full_delta[drain_index] += direction
        full_delta[source_index] -= direction

        free_delta = full_delta[: len(self.assembly.free_names)]

        regulated_delta = full_delta[len(self.assembly.free_names) :]

        # The independently reconstructed terminal incidence must agree
        # exactly with the compiled free-node incidence matrix.
        if not np.array_equal(
            free_delta,
            compiled_free_delta,
        ):
            component = self.active_components[device_index]

            raise RuntimeError(
                f"Compiled incidence mismatch for device "
                f"'{component.name}': terminal reconstruction "
                f"{free_delta} differs from compiled delta "
                f"{compiled_free_delta}."
            )

        q_after = q_before + free_delta

        component = self.active_components[device_index]

        v_active_before, v_control_before = extract_device_voltages(
            component,
            potentials_before,
        )

        selected_rate = (
            rates_reverse[device_index] if is_reverse else rates_forward[device_index]
        )

        event = {
            "device_index": int(device_index),
            "is_reverse": bool(is_reverse),
            "direction": int(direction),
            "waiting_time": float(waiting_time),
            "total_rate": float(total_rate),
            "forward_rate": float(rates_forward[device_index]),
            "reverse_rate": float(rates_reverse[device_index]),
            "selected_rate": float(selected_rate),
            "v_active_before": float(v_active_before),
            "v_control_before": float(v_control_before),
            "q_before": q_before.copy(),
            "q_after": q_after.copy(),
            "free_delta_count": free_delta.copy(),
            "regulated_delta_count": regulated_delta.copy(),
        }

        return q_after, float(waiting_time), event

    def execute_step(
        self,
        q: np.ndarray,
        vr: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Sample and execute one Gillespie event.

        This low-level method does not enforce t_finish. Full trajectory
        time-limit handling is performed by simulate().
        """

        sampled = self._sample_event(q, vr)

        if sampled is None:
            return np.asarray(q, dtype=np.int64).copy(), self.t_finish

        q_after, waiting_time, _event = sampled

        return q_after, waiting_time

    def simulate(
        self,
        q_init: np.ndarray,
        vr: np.ndarray,
        max_steps: int = 200000,
    ) -> dict[str, Any]:
        """
        Simulate a stochastic trajectory and record every executed event.

        An event is executed only when its sampled occurrence time is at or
        before t_finish. If the next event lies beyond t_finish, the final
        residence interval is recorded without applying that event.
        """

        q = np.asarray(q_init, dtype=np.int64).copy()
        vr = np.asarray(vr, dtype=np.float64)

        t = 0.0

        history_q = [q.copy()]
        history_t = [t]

        history_e = [self.compute_electrostatic_energy(q, vr)]

        history_v = {
            name: [value]
            for name, value in self.compute_node_potentials(
                q,
                vr,
            ).items()
        }

        event_time = []
        event_waiting_time = []
        event_device_index = []
        event_is_reverse = []
        event_direction = []

        event_total_rate = []
        event_forward_rate = []
        event_reverse_rate = []
        event_selected_rate = []

        event_v_active_before = []
        event_v_control_before = []

        event_q_before = []
        event_q_after = []
        event_free_delta = []
        event_regulated_delta = []

        termination_reason = "max_steps"

        def append_state(
            state_time: float,
            state_charge: np.ndarray,
        ) -> None:
            history_t.append(float(state_time))
            history_q.append(state_charge.copy())

            history_e.append(
                self.compute_electrostatic_energy(
                    state_charge,
                    vr,
                )
            )

            potentials = self.compute_node_potentials(
                state_charge,
                vr,
            )

            for node_name, node_voltage in potentials.items():
                history_v[node_name].append(node_voltage)

        for _ in range(max_steps):
            if t >= self.t_finish:
                termination_reason = "t_finish"
                break

            sampled = self._sample_event(q, vr)

            if sampled is None:
                # No event is available. Record the final residence interval
                # without changing the state.
                if t < self.t_finish:
                    t = self.t_finish
                    append_state(t, q)

                termination_reason = "no_available_event"
                break

            q_after, waiting_time, event = sampled

            proposed_event_time = t + waiting_time

            if proposed_event_time > self.t_finish:
                # The state survives unchanged until t_finish. Do not execute
                # or record the sampled event.
                t = self.t_finish
                append_state(t, q)

                termination_reason = "t_finish"
                break

            # The event occurs within the requested simulation interval.
            t = proposed_event_time
            q = q_after

            append_state(t, q)

            event_time.append(t)
            event_waiting_time.append(event["waiting_time"])

            event_device_index.append(event["device_index"])

            event_is_reverse.append(event["is_reverse"])

            event_direction.append(event["direction"])

            event_total_rate.append(event["total_rate"])

            event_forward_rate.append(event["forward_rate"])

            event_reverse_rate.append(event["reverse_rate"])

            event_selected_rate.append(event["selected_rate"])

            event_v_active_before.append(event["v_active_before"])

            event_v_control_before.append(event["v_control_before"])

            event_q_before.append(event["q_before"])

            event_q_after.append(event["q_after"])

            event_free_delta.append(event["free_delta_count"])

            event_regulated_delta.append(event["regulated_delta_count"])

        if t >= self.t_finish:
            termination_reason = "t_finish"

        number_of_events = len(event_time)
        number_of_free_nodes = len(self.assembly.free_names)
        number_of_regulated_nodes = len(self.assembly.regulated_names)

        def stack_integer_vectors(
            values: list[np.ndarray],
            width: int,
        ) -> np.ndarray:
            if values:
                return np.stack(values).astype(
                    np.int64,
                    copy=False,
                )

            return np.empty(
                (0, width),
                dtype=np.int64,
            )

        events = {
            "time": np.asarray(
                event_time,
                dtype=np.float64,
            ),
            "waiting_time": np.asarray(
                event_waiting_time,
                dtype=np.float64,
            ),
            "device_index": np.asarray(
                event_device_index,
                dtype=np.int64,
            ),
            "is_reverse": np.asarray(
                event_is_reverse,
                dtype=np.bool_,
            ),
            "direction": np.asarray(
                event_direction,
                dtype=np.int8,
            ),
            "total_rate": np.asarray(
                event_total_rate,
                dtype=np.float64,
            ),
            "forward_rate": np.asarray(
                event_forward_rate,
                dtype=np.float64,
            ),
            "reverse_rate": np.asarray(
                event_reverse_rate,
                dtype=np.float64,
            ),
            "selected_rate": np.asarray(
                event_selected_rate,
                dtype=np.float64,
            ),
            "v_active_before": np.asarray(
                event_v_active_before,
                dtype=np.float64,
            ),
            "v_control_before": np.asarray(
                event_v_control_before,
                dtype=np.float64,
            ),
            "q_before": stack_integer_vectors(
                event_q_before,
                number_of_free_nodes,
            ),
            "q_after": stack_integer_vectors(
                event_q_after,
                number_of_free_nodes,
            ),
            "free_delta_count": stack_integer_vectors(
                event_free_delta,
                number_of_free_nodes,
            ),
            "regulated_delta_count": stack_integer_vectors(
                event_regulated_delta,
                number_of_regulated_nodes,
            ),
        }

        assert number_of_events == len(events["device_index"])

        return {
            "time": np.asarray(
                history_t,
                dtype=np.float64,
            ),
            "charge": np.asarray(
                history_q,
                dtype=np.int64,
            ),
            "energy": np.asarray(
                history_e,
                dtype=np.float64,
            ),
            "potentials": {
                name: np.asarray(
                    values,
                    dtype=np.float64,
                )
                for name, values in history_v.items()
            },
            "events": events,
            "devices": {
                "name": [component.name for component in self.active_components],
                "type": [component.type for component in self.active_components],
            },
            "termination_reason": termination_reason,
            "completed": bool(t >= self.t_finish),
        }
