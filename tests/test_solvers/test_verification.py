# tests/test_solvers/test_verifications.py
import numpy as np
import pytest
from sse_core.compiler.builder import SSECompiler
from sse_core.compiler.parser import SSEParser
from sse_core.solvers.gillespie import GillespieSolver
from sse_core.solvers.verification import (
    audit_first_law,
    audit_local_detailed_balance,
    audit_second_law,
    build_finite_state_model,
    evolve_probability,
    gibbs_distribution,
    relative_entropy,
    verify_charge_flux_conservation,
    verify_first_law,
    verify_local_detailed_balance,
    verify_second_law,
)


def test_thermodynamic_energy_bookkeeping():
    """
    Verify energy and charge accounting for a stochastic single-island
    trajectory.

    This test does not yet assert local detailed balance or the second
    law.
    """

    circuit_yaml = """
    schema_version: "1.0.0"

    simulation:
      solver: "gillespie"
      t_finish: 1.0e-6
      v_th: 0.0259
      seed: 42

    nodes:
      free:
        - name: "island"

      regulated:
        - name: "vg"
          type: "constant"
          value: 0.012

        - name: "reservoir"
          type: "constant"
          value: 0.003

    components:
      - type: "capacitor"
        name: "Cg"
        terminals: ["island", "vg"]
        specs:
          capacitance: 2.0e-18

      - type: "capacitor"
        name: "C0"
        terminals: ["island", "reservoir"]
        specs:
          capacitance: 3.0e-18

      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["island", "reservoir"]
        specs:
          resistance: 1.0e5
    """

    parsed_netlist = SSEParser.parse_string(circuit_yaml)

    assembly = SSECompiler.compile_string(circuit_yaml)

    solver = GillespieSolver(
        parsed_netlist,
        assembly,
    )

    vr = np.array(
        [0.012, 0.003],
        dtype=np.float64,
    )

    history = solver.simulate(
        q_init=np.array([0]),
        vr=vr,
        max_steps=50,
    )

    assert len(history["events"]["time"]) > 0

    audit = audit_first_law(
        history,
        assembly,
        vr,
    )

    assert verify_first_law(
        history,
        assembly,
        vr,
    )

    assert verify_charge_flux_conservation(
        history,
        assembly,
    )

    assert audit["first_law_residual"] == pytest.approx(
        0.0,
        abs=1.0e-28,
    )

    assert audit["trajectory_energy_error"] == pytest.approx(
        0.0,
        abs=1.0e-28,
    )

    assert audit["max_stored_energy_error"] == pytest.approx(
        0.0,
        abs=1.0e-28,
    )

    assert audit["max_heat_identity_error"] == pytest.approx(
        0.0,
        abs=1.0e-28,
    )


def test_tunnel_junction_embedded_rates_satisfy_ldb():
    """
    Midpoint-evaluated embedded tunnel-junction rates must satisfy
    local detailed balance between a transition and its conjugate
    reverse transition.
    """

    capacitance = 1.0e-18

    circuit_yaml = f"""
    schema_version: "1.0.0"

    simulation:
      solver: "gillespie"
      t_finish: 1.0e-9
      v_th: 0.0259
      seed: 42

    nodes:
      free:
        - name: "island"

      regulated:
        - name: "gnd"
          type: "constant"
          value: 0.0

    components:
      - type: "capacitor"
        name: "C0"
        terminals: ["island", "gnd"]
        specs:
          capacitance: {capacitance}

      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["island", "gnd"]
        specs:
          resistance: 1.0e5
    """

    parsed_netlist = SSEParser.parse_string(circuit_yaml)

    assembly = SSECompiler.compile_string(circuit_yaml)

    solver = GillespieSolver(
        parsed_netlist,
        assembly,
    )

    vr = np.array(
        [0.0],
        dtype=np.float64,
    )

    history = {
        "events": {
            "q_before": np.array(
                [[1]],
                dtype=np.int64,
            ),
            "q_after": np.array(
                [[2]],
                dtype=np.int64,
            ),
            "free_delta_count": np.array(
                [[1]],
                dtype=np.int64,
            ),
            "regulated_delta_count": np.array(
                [[-1]],
                dtype=np.int64,
            ),
            "device_index": np.array(
                [0],
                dtype=np.int64,
            ),
            "direction": np.array(
                [1],
                dtype=np.int8,
            ),
        }
    }

    audit = audit_local_detailed_balance(
        history,
        solver,
        vr,
    )

    assert audit["valid_rate_pair"][0]

    # The isolated fixed-voltage device kernel obeys its required rate
    # ratio at both endpoint voltages.
    assert audit["fixed_voltage_residual_before"][0] == pytest.approx(
        0.0,
        abs=1.0e-10,
    )

    assert audit["fixed_voltage_residual_after"][0] == pytest.approx(
        0.0,
        abs=1.0e-10,
    )

    # The midpoint-embedded event and its conjugate reverse now obey
    # circuit-level local detailed balance.
    assert audit["ldb_residual"][0] == pytest.approx(
        0.0,
        abs=1.0e-10,
    )

    assert verify_local_detailed_balance(
        history,
        solver,
        vr,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_simulated_tunnel_junction_trajectory_satisfies_ldb():
    """
    Every event generated by the midpoint-corrected solver must satisfy
    local detailed balance with its conjugate reverse transition.
    """

    circuit_yaml = """
    schema_version: "1.0.0"

    simulation:
      solver: "gillespie"
      t_finish: 1.0e-7
      v_th: 0.0259
      seed: 42

    nodes:
      free:
        - name: "island"

      regulated:
        - name: "gate"
          type: "constant"
          value: 0.01

        - name: "reservoir"
          type: "constant"
          value: 0.0

    components:
      - type: "capacitor"
        name: "Cg"
        terminals: ["island", "gate"]
        specs:
          capacitance: 2.0e-18

      - type: "capacitor"
        name: "C0"
        terminals: ["island", "reservoir"]
        specs:
          capacitance: 3.0e-18

      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["island", "reservoir"]
        specs:
          resistance: 1.0e5
    """

    parsed_netlist = SSEParser.parse_string(circuit_yaml)

    assembly = SSECompiler.compile_string(circuit_yaml)

    solver = GillespieSolver(
        parsed_netlist,
        assembly,
    )

    vr = np.array(
        [0.01, 0.0],
        dtype=np.float64,
    )

    history = solver.simulate(
        q_init=np.array([0]),
        vr=vr,
        max_steps=100,
    )

    assert len(history["events"]["time"]) > 0

    audit = audit_local_detailed_balance(
        history,
        solver,
        vr,
    )

    assert np.all(audit["valid_rate_pair"])

    np.testing.assert_allclose(
        audit["ldb_residual"],
        np.zeros_like(audit["ldb_residual"]),
        rtol=1.0e-9,
        atol=1.0e-9,
    )

    assert verify_local_detailed_balance(
        history,
        solver,
        vr,
    )


def test_finite_state_ensemble_satisfies_second_law():
    """
    Verify the ensemble second law for the relaxation of a single
    electron island.

    The test checks:

    1. The Gibbs distribution is stationary.
    2. Entropy production vanishes at equilibrium.
    3. Entropy production is positive away from equilibrium.
    4. Relative entropy to equilibrium decreases under exact
       master-equation evolution.
    """

    circuit_yaml = """
    schema_version: "1.0.0"

    simulation:
      solver: "gillespie"
      t_finish: 1.0e-7
      v_th: 0.0259
      seed: 42

    nodes:
      free:
        - name: "island"

      regulated:
        - name: "gnd"
          type: "constant"
          value: 0.0

    components:
      - type: "capacitor"
        name: "C0"
        terminals: ["island", "gnd"]
        specs:
          capacitance: 2.0e-18

      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["island", "gnd"]
        specs:
          resistance: 1.0e5
    """

    parsed_netlist = SSEParser.parse_string(circuit_yaml)

    assembly = SSECompiler.compile_string(circuit_yaml)

    solver = GillespieSolver(
        parsed_netlist,
        assembly,
    )

    vr = np.array(
        [0.0],
        dtype=np.float64,
    )

    # The boundary states have negligible Gibbs probability for this
    # capacitance and temperature.
    states = np.arange(
        -6,
        7,
        dtype=np.int64,
    ).reshape(-1, 1)

    model = build_finite_state_model(
        solver,
        vr,
        states,
    )

    generator = model["generator"]
    channel_rates = model["channel_rates"]

    # A Markov generator has zero row sums. The entries here can be very
    # large physical rates, so test the residual relative to the generator
    # scale rather than with a fixed absolute tolerance.
    generator_rate_scale = max(
        1.0,
        float(np.max(np.abs(generator))),
    )

    row_sum_residual = np.sum(
        generator,
        axis=1,
    )

    assert (
        np.max(np.abs(row_sum_residual)) / generator_rate_scale
        < 100.0 * np.finfo(np.float64).eps
    )

    rate_scale = max(
        1.0,
        float(np.max(-np.diag(generator))),
    )

    equilibrium = gibbs_distribution(
        solver,
        vr,
        states,
    )

    # LDB implies that the Gibbs distribution is stationary.
    stationary_residual = equilibrium @ generator

    assert np.max(np.abs(stationary_residual)) / rate_scale < 1.0e-11

    equilibrium_audit = audit_second_law(
        equilibrium,
        generator,
        channel_rates,
    )

    equilibrium_activity = max(
        1.0,
        equilibrium_audit["dynamical_activity"],
    )

    # At equilibrium all probability currents and entropy production
    # vanish, up to floating-point error.
    assert (
        abs(equilibrium_audit["total_entropy_rate_over_kb"]) / equilibrium_activity
        < 1.0e-11
    )

    assert verify_second_law(
        equilibrium,
        generator,
        channel_rates,
    )

    # Construct a strictly positive distribution displaced from
    # equilibrium.
    electron_number = states[:, 0].astype(np.float64)

    displaced_profile = np.exp(-0.5 * ((electron_number - 2.0) / 0.65) ** 2)

    displaced_profile /= np.sum(displaced_profile)

    probability_initial = 0.02 * equilibrium + 0.98 * displaced_profile

    probability_initial /= np.sum(probability_initial)

    nonequilibrium_audit = audit_second_law(
        probability_initial,
        generator,
        channel_rates,
    )

    assert nonequilibrium_audit["total_entropy_rate_over_kb"] > 0.0

    assert nonequilibrium_audit["minimum_pair_entropy_rate_over_kb"] >= -1.0e-10 * max(
        1.0,
        nonequilibrium_audit["dynamical_activity"],
    )

    assert verify_second_law(
        probability_initial,
        generator,
        channel_rates,
    )

    relative_entropy_before = relative_entropy(
        probability_initial,
        equilibrium,
    )

    # Choose a short but dynamically meaningful exact evolution time.
    duration = 0.05 / rate_scale

    probability_after = evolve_probability(
        probability_initial,
        generator,
        duration,
    )

    relative_entropy_after = relative_entropy(
        probability_after,
        equilibrium,
    )

    # For equilibrium relaxation, the nonequilibrium free-energy
    # distance D(p || p_eq) cannot increase.
    assert relative_entropy_after < relative_entropy_before
