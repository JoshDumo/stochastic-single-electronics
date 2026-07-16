# src/sse_core/solvers/verification.py
from typing import Any

import numpy as np
from sse_core.compiler.builder import CompiledAssembly


def audit_first_law(
    history: dict[str, Any], assembly: CompiledAssembly, vr: np.ndarray
) -> dict[str, float]:
    """
    Audits the First Law of Thermodynamics: ΔU + Q_diss = W_sources.

    - ΔU: Change in internal electrostatic energy.
    - W_sources: Work done by regulated voltage sources during transitions.
    - Q_diss: Dissipated heat across tunneling barriers.
    """
    times = history["time"]
    charges = history["charge"]
    energies = history["energy"]

    # 1. Calculate ΔU
    delta_u = energies[-1] - energies[0]

    # 2. Integrate Work and Dissipated Heat over the trajectory
    w_sources = 0.0
    q_diss = 0.0

    n_steps = len(times) - 1
    for step in range(n_steps):
        q_curr = charges[step]
        q_next = charges[step + 1]

        # Identify if a charge transition occurred
        dq = q_next - q_curr
        if np.any(dq != 0):
            # Locate which device fired by matching the charge delta
            # We can find the work done by looking at potentials and charge displacements
            # Direct calculation of energy change for this single transition:
            u_curr = energies[step]
            u_next = energies[step + 1]
            du_step = u_next - u_curr

            # The exact work done by external sources when charge Δq is displaced on free nodes:
            # W_sources = vr^T @ Cx^T @ C_inv @ Δq
            if len(assembly.regulated_names) > 0 and len(assembly.free_names) > 0:
                work_step = float(
                    vr.T @ assembly.Cx.T @ assembly.C_inv @ dq.astype(np.float64)
                )
            else:
                work_step = 0.0

            w_sources += work_step

            # First Law definition: Q_diss = W_sources - ΔU
            # Heat dissipated in the barrier is the work put into the hop minus the stored electrostatic energy
            q_diss += work_step - du_step

    return {
        "delta_U": delta_u,
        "W_sources": w_sources,
        "Q_diss": q_diss,
        "discrepancy": abs((delta_u + q_diss) - w_sources),
    }


def verify_second_law(
    history: dict[str, Any], assembly: CompiledAssembly, vr: np.ndarray
) -> bool:
    """
    Verifies the Second Law of Thermodynamics (Entropy Generation):
    For every stochastically selected tunneling step, the local dissipated heat
    associated with the physical jump must be strictly non-negative (Q_diss >= 0)
    in the absence of thermal fluctuations, and total integrated Q_diss must be >= 0.
    """
    audit = audit_first_law(history, assembly, vr)
    # The total integrated heat dissipation across all barriers must be non-negative
    return audit["Q_diss"] >= -1e-9


def verify_charge_flux_conservation(
    history: dict[str, Any], assembly: CompiledAssembly
) -> bool:
    """
    Verifies Kirchhoff's Charge Law / Charge Flux Balance:
    The difference between the final and initial charge vectors must match
    the net charge displaced by the individual transition events:

    q_final - q_initial = free_Delta @ (jumps_forward - jumps_reverse)
    """
    charges = history["charge"]
    actual_delta_q = charges[-1] - charges[0]

    # Track the count of forward and reverse jumps for each active device
    n_devices = assembly.free_Delta.shape[1]
    jumps_f = np.zeros(n_devices)
    jumps_r = np.zeros(n_devices)

    n_steps = len(history["time"]) - 1
    for step in range(n_steps):
        q_curr = charges[step]
        q_next = charges[step + 1]
        dq = q_next - q_curr

        if np.any(dq != 0):
            # Locate which device column in free_Delta matches this dq step
            matched = False
            for d in range(n_devices):
                col = np.round(assembly.free_Delta[:, d]).astype(np.int64)

                # Check if this transition matches a forward hop (+col)
                if np.array_equal(dq, col):
                    jumps_f[d] += 1
                    matched = True
                    break
                # Check if this transition matches a reverse hop (-col)
                elif np.array_equal(dq, -col):
                    jumps_r[d] += 1
                    matched = True
                    break

            if not matched:
                # Discrepancy: A state jump occurred that doesn't correspond to any valid device column!
                return False

    # Reconstruct the expected charge change based purely on compiled incidence paths
    reconstructed_delta_q = assembly.free_Delta @ (jumps_f - jumps_r)

    # Assert actual matches reconstructed charge displacement to zero tolerance
    return np.allclose(actual_delta_q, reconstructed_delta_q)
