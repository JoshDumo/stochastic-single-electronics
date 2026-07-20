from typing import Any

import numpy as np
from scipy.linalg import expm
from sse_core.compiler.builder import CompiledAssembly
from sse_core.compiler.units import E_CHARGE
from sse_core.devices.mapping import extract_device_voltages


def _physical_free_charge(q_count: np.ndarray) -> np.ndarray:
    """
    Convert native excess-electron counts to physical charge in coulombs.

    The simulator convention is

        Q_physical = -e * n_electron.
    """

    return -E_CHARGE * np.asarray(q_count, dtype=np.float64)


def _electrostatic_energy(
    q_count: np.ndarray,
    assembly: CompiledAssembly,
    vr: np.ndarray,
) -> float:
    """
    Independently evaluate the electrostatic field energy E in joules.

    For

        M = [[C,    Cx],
             [Cx.T, Cr]],

    the energy at fixed regulated-node voltages is

        E = 1/2 Q.T C^-1 Q
            + 1/2 Vr.T
              (Cr - Cx.T C^-1 Cx)
              Vr.
    """

    q_physical = _physical_free_charge(q_count)
    vr = np.asarray(vr, dtype=np.float64)

    free_energy = 0.5 * float(q_physical.T @ assembly.C_inv @ q_physical)

    schur_capacitance = assembly.Cr - assembly.Cx.T @ assembly.C_inv @ assembly.Cx

    regulated_energy = 0.5 * float(vr.T @ schur_capacitance @ vr)

    return free_energy + regulated_energy


def _open_circuit_potential(
    q_count: np.ndarray,
    assembly: CompiledAssembly,
    vr: np.ndarray,
) -> float:
    """
    Evaluate the open-circuit potential Phi in joules:

        Phi = E - Vr.T Cx.T C^-1 Q.
    """

    q_physical = _physical_free_charge(q_count)
    vr = np.asarray(vr, dtype=np.float64)

    source_coupling = float(vr.T @ assembly.Cx.T @ assembly.C_inv @ q_physical)

    return (
        _electrostatic_energy(
            q_count,
            assembly,
            vr,
        )
        - source_coupling
    )


def _source_work_for_event(
    free_delta_count: np.ndarray,
    regulated_delta_count: np.ndarray,
    assembly: CompiledAssembly,
    vr: np.ndarray,
) -> float:
    """
    Return the work performed by the regulated voltage sources during
    one event.

    There are two contributions:

    1. Direct carrier transfer to or from regulated terminals.
    2. Charge induced on regulated conductors by the change of free-node
       charge.

    In physical-charge notation,

        W_sources =
            -Vr.T delta_Qr_direct
            + Vr.T Cx.T C^-1 delta_Qfree.
    """

    vr = np.asarray(vr, dtype=np.float64)

    delta_q_free = _physical_free_charge(free_delta_count)

    delta_q_regulated_direct = _physical_free_charge(regulated_delta_count)

    direct_work = -float(vr.T @ delta_q_regulated_direct)

    induced_regulated_charge = assembly.Cx.T @ assembly.C_inv @ delta_q_free

    induced_work = float(vr.T @ induced_regulated_charge)

    return direct_work + induced_work


def audit_first_law(
    history: dict[str, Any],
    assembly: CompiledAssembly,
    vr: np.ndarray,
) -> dict[str, Any]:
    """
    Audit event-by-event energy bookkeeping.

    Sign conventions:

        delta_E:
            Change in electrostatic field energy.

        W_sources:
            Work performed on the circuit by regulated voltage sources.

        Q_system:
            Heat absorbed by the circuit from the device environment.

        Q_environment:
            Heat delivered to the device environment.

    Therefore,

        delta_E = W_sources + Q_system

    and

        Q_environment = -Q_system.

    This audit checks accounting consistency. It is not, by itself, a
    proof that the transition rates satisfy local detailed balance.
    """

    vr = np.asarray(vr, dtype=np.float64)

    if "events" not in history:
        raise ValueError("History does not contain an event ledger.")

    charges = np.asarray(
        history["charge"],
        dtype=np.int64,
    )

    stored_energies = np.asarray(
        history["energy"],
        dtype=np.float64,
    )

    events = history["events"]

    q_before = np.asarray(
        events["q_before"],
        dtype=np.int64,
    )

    q_after = np.asarray(
        events["q_after"],
        dtype=np.int64,
    )

    free_delta_count = np.asarray(
        events["free_delta_count"],
        dtype=np.int64,
    )

    regulated_delta_count = np.asarray(
        events["regulated_delta_count"],
        dtype=np.int64,
    )

    number_of_events = len(q_before)

    if len(q_after) != number_of_events:
        raise ValueError("Event ledger q_before and q_after lengths differ.")

    if len(free_delta_count) != number_of_events:
        raise ValueError("Event ledger free_delta_count length differs.")

    if len(regulated_delta_count) != number_of_events:
        raise ValueError("Event ledger regulated_delta_count length differs.")

    if len(stored_energies) != len(charges):
        raise ValueError("State charge and energy history lengths differ.")

    # Independently recompute every stored state energy.
    recomputed_energies = np.asarray(
        [
            _electrostatic_energy(
                state,
                assembly,
                vr,
            )
            for state in charges
        ],
        dtype=np.float64,
    )

    stored_energy_error = stored_energies - recomputed_energies

    event_delta_e = np.zeros(
        number_of_events,
        dtype=np.float64,
    )

    event_work_sources = np.zeros(
        number_of_events,
        dtype=np.float64,
    )

    event_heat_system = np.zeros(
        number_of_events,
        dtype=np.float64,
    )

    event_heat_environment = np.zeros(
        number_of_events,
        dtype=np.float64,
    )

    event_heat_identity_error = np.zeros(
        number_of_events,
        dtype=np.float64,
    )

    for event_index in range(number_of_events):
        before = q_before[event_index]
        after = q_after[event_index]

        expected_after = before + free_delta_count[event_index]

        if not np.array_equal(
            after,
            expected_after,
        ):
            raise ValueError(
                f"Event {event_index} has inconsistent "
                "q_before, q_after and free_delta_count."
            )

        energy_before = _electrostatic_energy(
            before,
            assembly,
            vr,
        )

        energy_after = _electrostatic_energy(
            after,
            assembly,
            vr,
        )

        delta_e = energy_after - energy_before

        work_sources = _source_work_for_event(
            free_delta_count[event_index],
            regulated_delta_count[event_index],
            assembly,
            vr,
        )

        heat_system = delta_e - work_sources
        heat_environment = -heat_system

        # Equivalent expression from
        #
        #   delta_Q_system
        #       = delta_Phi
        #         + Vr.T delta_Qr_direct.
        #
        phi_before = _open_circuit_potential(
            before,
            assembly,
            vr,
        )

        phi_after = _open_circuit_potential(
            after,
            assembly,
            vr,
        )

        delta_q_regulated_direct = _physical_free_charge(
            regulated_delta_count[event_index]
        )

        heat_system_from_phi = (
            phi_after - phi_before + float(vr.T @ delta_q_regulated_direct)
        )

        event_delta_e[event_index] = delta_e
        event_work_sources[event_index] = work_sources
        event_heat_system[event_index] = heat_system
        event_heat_environment[event_index] = heat_environment

        event_heat_identity_error[event_index] = heat_system - heat_system_from_phi

    total_delta_e = float(np.sum(event_delta_e))

    total_work_sources = float(np.sum(event_work_sources))

    total_heat_system = float(np.sum(event_heat_system))

    total_heat_environment = float(np.sum(event_heat_environment))

    trajectory_delta_e = float(recomputed_energies[-1] - recomputed_energies[0])

    trajectory_energy_error = total_delta_e - trajectory_delta_e

    first_law_residual = total_delta_e - total_work_sources - total_heat_system

    max_stored_energy_error = (
        float(np.max(np.abs(stored_energy_error))) if len(stored_energy_error) else 0.0
    )

    max_heat_identity_error = (
        float(np.max(np.abs(event_heat_identity_error))) if number_of_events else 0.0
    )

    return {
        "event_delta_E": event_delta_e,
        "event_W_sources": event_work_sources,
        "event_Q_system": event_heat_system,
        "event_Q_environment": (event_heat_environment),
        "event_heat_identity_error": (event_heat_identity_error),
        "recomputed_energy": recomputed_energies,
        "stored_energy_error": stored_energy_error,
        "delta_E": total_delta_e,
        "trajectory_delta_E": trajectory_delta_e,
        "W_sources": total_work_sources,
        "Q_system": total_heat_system,
        "Q_environment": total_heat_environment,
        "first_law_residual": first_law_residual,
        "trajectory_energy_error": (trajectory_energy_error),
        "max_stored_energy_error": (max_stored_energy_error),
        "max_heat_identity_error": (max_heat_identity_error),
    }


def verify_first_law(
    history: dict[str, Any],
    assembly: CompiledAssembly,
    vr: np.ndarray,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-28,
) -> bool:
    """
    Return True when the ledger, stored energies and energy bookkeeping
    are mutually consistent.
    """

    audit = audit_first_law(
        history,
        assembly,
        vr,
    )

    energy_scale = max(
        abs(audit["delta_E"]),
        abs(audit["W_sources"]),
        abs(audit["Q_system"]),
        1.0e-30,
    )

    tolerance = atol + rtol * energy_scale

    return bool(
        abs(audit["first_law_residual"]) <= tolerance
        and abs(audit["trajectory_energy_error"]) <= tolerance
        and audit["max_stored_energy_error"] <= tolerance
        and audit["max_heat_identity_error"] <= tolerance
    )


def verify_charge_flux_conservation(
    history: dict[str, Any],
    assembly: CompiledAssembly,
) -> bool:
    """
    Verify the event ledger and the net free-node state displacement.

    Each two-terminal event must conserve carrier count across its free
    and regulated terminals.
    """

    if "events" not in history:
        return False

    charges = np.asarray(
        history["charge"],
        dtype=np.int64,
    )

    events = history["events"]

    q_before = np.asarray(
        events["q_before"],
        dtype=np.int64,
    )

    q_after = np.asarray(
        events["q_after"],
        dtype=np.int64,
    )

    free_delta = np.asarray(
        events["free_delta_count"],
        dtype=np.int64,
    )

    regulated_delta = np.asarray(
        events["regulated_delta_count"],
        dtype=np.int64,
    )

    number_of_events = len(q_before)

    if q_after.shape != q_before.shape:
        return False

    if free_delta.shape != q_before.shape:
        return False

    if regulated_delta.shape != (
        number_of_events,
        len(assembly.regulated_names),
    ):
        return False

    if not np.array_equal(
        q_after - q_before,
        free_delta,
    ):
        return False

    total_carrier_delta = np.sum(free_delta, axis=1) + np.sum(regulated_delta, axis=1)

    if not np.array_equal(
        total_carrier_delta,
        np.zeros(
            number_of_events,
            dtype=np.int64,
        ),
    ):
        return False

    reconstructed_net_delta = np.sum(
        free_delta,
        axis=0,
    )

    actual_net_delta = charges[-1] - charges[0]

    if not np.array_equal(
        reconstructed_net_delta,
        actual_net_delta,
    ):
        return False

    # Events occupy the first number_of_events state transitions.
    # A possible final duplicate state at t_finish is not an event.
    if number_of_events:
        if not np.array_equal(
            q_before,
            charges[:number_of_events],
        ):
            return False

        if not np.array_equal(
            q_after,
            charges[1 : number_of_events + 1],
        ):
            return False

    return True


def audit_local_detailed_balance(
    history: dict[str, Any],
    solver: Any,
    vr: np.ndarray,
) -> dict[str, Any]:
    """
    Audit local detailed balance for every recorded event.

    For an event taking the system from q_before to q_after,

        log(
            lambda_event(q_before)
            / lambda_conjugate(q_after)
        )
        =
        -Q_system / (e * V_T).

    Here:

    - lambda_event is the rate of the actually selected direction at
      q_before.
    - lambda_conjugate is the opposite-direction rate of the same
      device at q_after.
    - Q_system is the heat absorbed by the circuit during the event.
    - e * V_T = k_B * T.

    This function also checks the fixed-voltage device relation

        log(lambda_forward / lambda_reverse)
        = V_active / V_T

    at the initial and final states. That separates errors in the
    fixed-voltage device kernel from errors in embedding the device in
    a charge-dependent circuit.
    """

    if "events" not in history:
        raise ValueError("History does not contain an event ledger.")

    vr = np.asarray(vr, dtype=np.float64)
    events = history["events"]

    q_before = np.asarray(
        events["q_before"],
        dtype=np.int64,
    )

    q_after = np.asarray(
        events["q_after"],
        dtype=np.int64,
    )

    free_delta_count = np.asarray(
        events["free_delta_count"],
        dtype=np.int64,
    )

    regulated_delta_count = np.asarray(
        events["regulated_delta_count"],
        dtype=np.int64,
    )

    device_index = np.asarray(
        events["device_index"],
        dtype=np.int64,
    )

    direction = np.asarray(
        events["direction"],
        dtype=np.int8,
    )

    number_of_events = len(device_index)

    expected_free_shape = (
        number_of_events,
        len(solver.assembly.free_names),
    )

    expected_regulated_shape = (
        number_of_events,
        len(solver.assembly.regulated_names),
    )

    if q_before.shape != expected_free_shape:
        raise ValueError(
            f"q_before has shape {q_before.shape}; expected {expected_free_shape}."
        )

    if q_after.shape != expected_free_shape:
        raise ValueError(
            f"q_after has shape {q_after.shape}; expected {expected_free_shape}."
        )

    if free_delta_count.shape != expected_free_shape:
        raise ValueError("free_delta_count has an unexpected shape.")

    if regulated_delta_count.shape != expected_regulated_shape:
        raise ValueError("regulated_delta_count has an unexpected shape.")

    if not np.array_equal(
        q_after - q_before,
        free_delta_count,
    ):
        raise ValueError("The event ledger contains inconsistent state changes.")

    if np.any((direction != 1) & (direction != -1)):
        raise ValueError("Event directions must be either +1 or -1.")

    event_rate = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    conjugate_rate = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    log_rate_ratio = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    heat_system = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    expected_log_rate_ratio = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    ldb_residual = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    v_active_before = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    v_active_after = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    v_active_midpoint = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    v_control_before = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    v_control_after = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    v_control_midpoint = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    fixed_voltage_residual_before = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    fixed_voltage_residual_after = np.full(
        number_of_events,
        np.nan,
        dtype=np.float64,
    )

    valid_rate_pair = np.zeros(
        number_of_events,
        dtype=np.bool_,
    )

    device_names: list[str] = []
    device_types: list[str] = []

    thermal_energy = E_CHARGE * float(solver.v_th)

    if thermal_energy <= 0.0:
        raise ValueError("The thermal voltage must be positive.")

    for event_number in range(number_of_events):
        index = int(device_index[event_number])

        if index < 0 or index >= len(solver.active_components):
            raise ValueError(
                f"Event {event_number} references invalid device index {index}."
            )

        component = solver.active_components[index]

        device_names.append(component.name)
        device_types.append(component.type)

        before = q_before[event_number]
        after = q_after[event_number]

        potentials_before = solver.compute_node_potentials(
            before,
            vr,
        )

        potentials_after = solver.compute_node_potentials(
            after,
            vr,
        )

        # Circuit-embedded midpoint rates.
        (
            embedded_forward_before,
            embedded_reverse_before,
            _,
        ) = solver.compute_embedded_rates(
            before,
            vr,
        )

        (
            embedded_forward_after,
            embedded_reverse_after,
            _,
        ) = solver.compute_embedded_rates(
            after,
            vr,
        )

        # Fixed-voltage rates used only to audit the isolated device kernel.
        (
            fixed_forward_before,
            fixed_reverse_before,
            _,
        ) = solver.compute_all_rates(potentials_before)

        (
            fixed_forward_after,
            fixed_reverse_after,
            _,
        ) = solver.compute_all_rates(potentials_after)

        (
            active_before,
            control_before,
        ) = extract_device_voltages(
            component,
            potentials_before,
        )

        (
            active_after,
            control_after,
        ) = extract_device_voltages(
            component,
            potentials_after,
        )

        v_active_before[event_number] = active_before

        v_active_after[event_number] = active_after

        v_active_midpoint[event_number] = 0.5 * (active_before + active_after)

        v_control_before[event_number] = control_before

        v_control_after[event_number] = control_after

        v_control_midpoint[event_number] = 0.5 * (control_before + control_after)

        if direction[event_number] == 1:
            selected_event_rate = embedded_forward_before[index]

            selected_conjugate_rate = embedded_reverse_after[index]

        else:
            selected_event_rate = embedded_reverse_before[index]

            selected_conjugate_rate = embedded_forward_after[index]

        event_rate[event_number] = selected_event_rate

        conjugate_rate[event_number] = selected_conjugate_rate

        rates_are_valid = bool(
            np.isfinite(selected_event_rate)
            and np.isfinite(selected_conjugate_rate)
            and selected_event_rate > 0.0
            and selected_conjugate_rate > 0.0
        )

        valid_rate_pair[event_number] = rates_are_valid

        energy_before = _electrostatic_energy(
            before,
            solver.assembly,
            vr,
        )

        energy_after = _electrostatic_energy(
            after,
            solver.assembly,
            vr,
        )

        delta_energy = energy_after - energy_before

        source_work = _source_work_for_event(
            free_delta_count[event_number],
            regulated_delta_count[event_number],
            solver.assembly,
            vr,
        )

        event_heat_system = delta_energy - source_work

        heat_system[event_number] = event_heat_system

        expected_log_rate_ratio[event_number] = -event_heat_system / thermal_energy

        if rates_are_valid:
            actual_log_ratio = np.log(selected_event_rate) - np.log(
                selected_conjugate_rate
            )

            log_rate_ratio[event_number] = actual_log_ratio

            ldb_residual[event_number] = (
                actual_log_ratio - expected_log_rate_ratio[event_number]
            )

        forward_before = fixed_forward_before[index]
        reverse_before = fixed_reverse_before[index]

        if (
            np.isfinite(forward_before)
            and np.isfinite(reverse_before)
            and forward_before > 0.0
            and reverse_before > 0.0
        ):
            fixed_voltage_residual_before[event_number] = (
                np.log(forward_before)
                - np.log(reverse_before)
                - active_before / solver.v_th
            )

        forward_after = fixed_forward_after[index]
        reverse_after = fixed_reverse_after[index]

        if (
            np.isfinite(forward_after)
            and np.isfinite(reverse_after)
            and forward_after > 0.0
            and reverse_after > 0.0
        ):
            fixed_voltage_residual_after[event_number] = (
                np.log(forward_after)
                - np.log(reverse_after)
                - active_after / solver.v_th
            )

    valid_residuals = ldb_residual[valid_rate_pair]

    max_abs_residual = (
        float(np.max(np.abs(valid_residuals))) if len(valid_residuals) else np.inf
    )

    return {
        "device_name": device_names,
        "device_type": device_types,
        "device_index": device_index.copy(),
        "direction": direction.copy(),
        "event_rate": event_rate,
        "conjugate_rate": conjugate_rate,
        "valid_rate_pair": valid_rate_pair,
        "log_rate_ratio": log_rate_ratio,
        "heat_system": heat_system,
        "expected_log_rate_ratio": (expected_log_rate_ratio),
        "ldb_residual": ldb_residual,
        "max_abs_residual": (max_abs_residual),
        "v_active_before": (v_active_before),
        "v_active_after": (v_active_after),
        "v_active_midpoint": (v_active_midpoint),
        "v_control_before": (v_control_before),
        "v_control_after": (v_control_after),
        "v_control_midpoint": (v_control_midpoint),
        "fixed_voltage_residual_before": (fixed_voltage_residual_before),
        "fixed_voltage_residual_after": (fixed_voltage_residual_after),
    }


def verify_local_detailed_balance(
    history: dict[str, Any],
    solver: Any,
    vr: np.ndarray,
    *,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-9,
) -> bool:
    """
    Return True when every recorded event satisfies local detailed
    balance within a dimensionless log-rate tolerance.
    """

    audit = audit_local_detailed_balance(
        history,
        solver,
        vr,
    )

    valid = audit["valid_rate_pair"]

    if len(valid) == 0:
        return True

    if not np.all(valid):
        return False

    expected = audit["expected_log_rate_ratio"]

    residual = audit["ldb_residual"]

    tolerance = atol + rtol * np.maximum(
        1.0,
        np.abs(expected),
    )

    return bool(np.all(np.abs(residual) <= tolerance))


def build_finite_state_model(
    solver: Any,
    vr: np.ndarray,
    states: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build a finite-state continuous-time Markov model using the
    simulator's midpoint-corrected embedded-device rates.

    Parameters
    ----------
    solver:
        Configured GillespieSolver.

    vr:
        Regulated-node voltage vector.

    states:
        Integer array with shape

            (number_of_states, number_of_free_nodes).

        Transitions leaving this finite state set are omitted, giving
        reflecting truncation boundaries.

    Returns
    -------
    dict containing:

        states:
            The validated integer state array.

        generator:
            Total row-oriented Markov generator K. A row probability
            vector evolves according to

                dp/dt = p @ K.

        channel_rates:
            Device-resolved off-diagonal rates with shape

                (number_of_devices, number_of_states, number_of_states).

            Keeping channels separate is necessary when several devices
            connect the same pair of states but carry different
            thermodynamic affinities.
    """

    vr = np.asarray(vr, dtype=np.float64)
    states = np.asarray(states, dtype=np.int64)

    if states.ndim != 2:
        raise ValueError("states must be a two-dimensional integer array.")

    number_of_states, number_of_free_nodes = states.shape

    if number_of_states == 0:
        raise ValueError("The finite state space cannot be empty.")

    if number_of_free_nodes != len(solver.assembly.free_names):
        raise ValueError(
            f"State width is {number_of_free_nodes}; expected "
            f"{len(solver.assembly.free_names)}."
        )

    if vr.shape != (len(solver.assembly.regulated_names),):
        raise ValueError(
            f"Regulated-voltage vector has shape {vr.shape}; "
            f"expected "
            f"{(len(solver.assembly.regulated_names),)}."
        )

    state_keys = [tuple(int(value) for value in state) for state in states]

    if len(set(state_keys)) != number_of_states:
        raise ValueError("The finite state array contains duplicate states.")

    state_index = {state_key: index for index, state_key in enumerate(state_keys)}

    number_of_devices = len(solver.active_components)

    channel_rates = np.zeros(
        (
            number_of_devices,
            number_of_states,
            number_of_states,
        ),
        dtype=np.float64,
    )

    for origin_index, state in enumerate(states):
        (
            rates_forward,
            rates_reverse,
            _,
        ) = solver.compute_embedded_rates(
            state,
            vr,
        )

        for device_index in range(number_of_devices):
            free_delta = np.rint(
                solver.assembly.free_Delta[
                    :,
                    device_index,
                ]
            ).astype(np.int64)

            if not np.any(free_delta):
                component = solver.active_components[device_index]

                raise ValueError(
                    "The finite-state second-law helper does not yet "
                    "support active components with no free-node state "
                    f"change: '{component.name}'."
                )

            forward_state = state + free_delta
            reverse_state = state - free_delta

            forward_key = tuple(int(value) for value in forward_state)

            reverse_key = tuple(int(value) for value in reverse_state)

            forward_destination = state_index.get(forward_key)

            reverse_destination = state_index.get(reverse_key)

            # A missing destination means that the transition leaves
            # the selected finite truncation and is reflected.
            if forward_destination is not None:
                channel_rates[
                    device_index,
                    origin_index,
                    forward_destination,
                ] += rates_forward[device_index]

            if reverse_destination is not None:
                channel_rates[
                    device_index,
                    origin_index,
                    reverse_destination,
                ] += rates_reverse[device_index]

    generator = np.sum(
        channel_rates,
        axis=0,
    )

    # Device transitions must be state-changing in this helper.
    np.fill_diagonal(
        generator,
        0.0,
    )

    outgoing_rate = np.sum(
        generator,
        axis=1,
    )

    np.fill_diagonal(
        generator,
        -outgoing_rate,
    )

    if not np.all(np.isfinite(generator)):
        raise RuntimeError("The finite-state generator contains non-finite values.")

    if np.any(channel_rates < 0.0):
        raise RuntimeError("The finite-state model contains negative transition rates.")

    np.testing.assert_allclose(
        np.sum(generator, axis=1),
        np.zeros(number_of_states),
        rtol=0.0,
        atol=1.0e-10
        * max(
            1.0,
            float(np.max(np.abs(generator))),
        ),
    )

    return {
        "states": states.copy(),
        "generator": generator,
        "channel_rates": channel_rates,
    }


def evolve_probability(
    probability: np.ndarray,
    generator: np.ndarray,
    duration: float,
) -> np.ndarray:
    """
    Evolve a row probability vector exactly over a finite time interval:

        p(t + dt) = p(t) exp(K dt).
    """

    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    generator = np.asarray(
        generator,
        dtype=np.float64,
    )

    if duration < 0.0:
        raise ValueError("Evolution duration must be non-negative.")

    number_of_states = len(probability)

    if generator.shape != (
        number_of_states,
        number_of_states,
    ):
        raise ValueError("Generator shape does not match the probability vector.")

    if np.any(probability < 0.0):
        raise ValueError("Probability entries cannot be negative.")

    if not np.isclose(
        np.sum(probability),
        1.0,
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise ValueError("Probability vector must sum to one.")

    transition_matrix = expm(generator * duration)

    evolved = probability @ transition_matrix

    # Remove only matrix-exponential roundoff.
    if np.min(evolved) < -1.0e-12:
        raise RuntimeError(
            "Probability evolution produced a materially negative probability."
        )

    evolved = np.clip(
        evolved,
        0.0,
        None,
    )

    normalization = float(np.sum(evolved))

    if normalization <= 0.0:
        raise RuntimeError("Evolved probability has zero normalization.")

    return evolved / normalization


def relative_entropy(
    probability: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Return the dimensionless Kullback-Leibler divergence

        D(p || r) = sum_i p_i log(p_i / r_i).
    """

    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    if probability.shape != reference.shape:
        raise ValueError("Probability and reference shapes differ.")

    if np.any(probability < 0.0):
        raise ValueError("Probability entries cannot be negative.")

    if np.any(reference <= 0.0):
        raise ValueError("Reference probabilities must be strictly positive.")

    positive = probability > 0.0

    return float(
        np.sum(
            probability[positive]
            * (np.log(probability[positive]) - np.log(reference[positive]))
        )
    )


def gibbs_distribution(
    solver: Any,
    vr: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """
    Construct the normalized Gibbs distribution over a finite state set:

        p_eq(q) proportional to exp[-Phi(q) / (e V_T)].

    This is the equilibrium distribution when the circuit topology and
    regulated sources admit an equilibrium potential.
    """

    vr = np.asarray(
        vr,
        dtype=np.float64,
    )

    states = np.asarray(
        states,
        dtype=np.int64,
    )

    thermal_energy = E_CHARGE * float(solver.v_th)

    if thermal_energy <= 0.0:
        raise ValueError("Thermal voltage must be positive.")

    potentials = np.asarray(
        [
            _open_circuit_potential(
                state,
                solver.assembly,
                vr,
            )
            for state in states
        ],
        dtype=np.float64,
    )

    # Shift before exponentiation for numerical stability.
    shifted = potentials - np.min(potentials)

    weights = np.exp(-shifted / thermal_energy)

    return weights / np.sum(weights)


def audit_second_law(
    probability: np.ndarray,
    generator: np.ndarray,
    channel_rates: np.ndarray,
) -> dict[str, float]:
    """
    Compute ensemble entropy-production rates for a finite-state Markov
    process.

    All returned entropy rates are divided by Boltzmann's constant and
    therefore have units of inverse seconds.

    The decomposition is

        sigma_total / k_B
            = dS_system/dt / k_B
            + sigma_environment / k_B.

    For each device-resolved state pair i <-> j,

        J_ij = p_i W_ij - p_j W_ji,

        sigma_ij / k_B
            = J_ij log[
                (p_i W_ij)
                / (p_j W_ji)
              ] >= 0.

    Device channels remain separate so that parallel transitions with
    different affinities are not incorrectly merged.
    """

    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    generator = np.asarray(
        generator,
        dtype=np.float64,
    )

    channel_rates = np.asarray(
        channel_rates,
        dtype=np.float64,
    )

    if probability.ndim != 1:
        raise ValueError("probability must be one-dimensional.")

    number_of_states = len(probability)

    if generator.shape != (
        number_of_states,
        number_of_states,
    ):
        raise ValueError("Generator shape does not match probability shape.")

    if channel_rates.ndim != 3 or channel_rates.shape[1:] != generator.shape:
        raise ValueError(
            "channel_rates must have shape "
            "(number_of_devices, number_of_states, number_of_states)."
        )

    if np.any(probability <= 0.0):
        raise ValueError(
            "The instantaneous second-law audit requires a strictly "
            "positive probability distribution."
        )

    if not np.isclose(
        np.sum(probability),
        1.0,
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise ValueError("Probability vector must sum to one.")

    if np.any(channel_rates < 0.0):
        raise ValueError("Transition rates cannot be negative.")

    # Verify that the supplied total generator really corresponds to
    # the device-resolved channels.
    reconstructed_generator = np.sum(
        channel_rates,
        axis=0,
    )

    np.fill_diagonal(
        reconstructed_generator,
        0.0,
    )

    np.fill_diagonal(
        reconstructed_generator,
        -np.sum(
            reconstructed_generator,
            axis=1,
        ),
    )

    generator_scale = max(
        1.0,
        float(np.max(np.abs(generator))),
    )

    if not np.allclose(
        generator,
        reconstructed_generator,
        rtol=1.0e-12,
        atol=1.0e-12 * generator_scale,
    ):
        raise ValueError("Generator is inconsistent with channel_rates.")

    probability_rate = probability @ generator

    system_entropy_rate = -float(probability_rate @ np.log(probability))

    environment_entropy_rate = 0.0
    pairwise_total_entropy_rate = 0.0
    dynamical_activity = 0.0

    minimum_pair_entropy_rate = np.inf
    number_of_pairs = 0

    number_of_devices = channel_rates.shape[0]

    for device_index in range(number_of_devices):
        rates = channel_rates[device_index]

        for state_i in range(number_of_states):
            for state_j in range(
                state_i + 1,
                number_of_states,
            ):
                rate_ij = float(rates[state_i, state_j])

                rate_ji = float(rates[state_j, state_i])

                if rate_ij == 0.0 and rate_ji == 0.0:
                    continue

                if rate_ij <= 0.0 or rate_ji <= 0.0:
                    raise ValueError(
                        "A transition channel has no positive conjugate reverse rate."
                    )

                flux_ij = probability[state_i] * rate_ij

                flux_ji = probability[state_j] * rate_ji

                current = flux_ij - flux_ji

                environment_pair = current * (np.log(rate_ij) - np.log(rate_ji))

                total_pair = current * (np.log(flux_ij) - np.log(flux_ji))

                environment_entropy_rate += environment_pair

                pairwise_total_entropy_rate += total_pair

                dynamical_activity += flux_ij + flux_ji

                minimum_pair_entropy_rate = min(
                    minimum_pair_entropy_rate,
                    total_pair,
                )

                number_of_pairs += 1

    if number_of_pairs == 0:
        minimum_pair_entropy_rate = 0.0

    decomposed_total_entropy_rate = system_entropy_rate + environment_entropy_rate

    decomposition_residual = decomposed_total_entropy_rate - pairwise_total_entropy_rate

    return {
        "system_entropy_rate_over_kb": (system_entropy_rate),
        "environment_entropy_rate_over_kb": (environment_entropy_rate),
        "total_entropy_rate_over_kb": (pairwise_total_entropy_rate),
        "decomposed_total_entropy_rate_over_kb": (decomposed_total_entropy_rate),
        "decomposition_residual": (decomposition_residual),
        "minimum_pair_entropy_rate_over_kb": (float(minimum_pair_entropy_rate)),
        "dynamical_activity": (float(dynamical_activity)),
    }


def verify_second_law(
    probability: np.ndarray,
    generator: np.ndarray,
    channel_rates: np.ndarray,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> bool:
    """
    Verify the ensemble second law for a finite-state model.

    This checks:

    1. Every channel-resolved pair contribution is non-negative.
    2. The total entropy-production rate is non-negative.
    3. The system-plus-environment decomposition agrees with the
       explicitly positive pairwise expression.
    """

    audit = audit_second_law(
        probability,
        generator,
        channel_rates,
    )

    scale = max(
        1.0,
        audit["dynamical_activity"],
    )

    tolerance = atol + rtol * scale

    return bool(
        audit["total_entropy_rate_over_kb"] >= -tolerance
        and audit["minimum_pair_entropy_rate_over_kb"] >= -tolerance
        and abs(audit["decomposition_residual"]) <= tolerance
    )
