import numpy as np
import pytest
from sse_core.compiler.builder import SSECompiler
from sse_core.compiler.parser import SSEParser
from sse_core.compiler.units import E_CHARGE
from sse_core.devices.semiconductor import (
    grounded_body_mosfet_rates,
)
from sse_core.solvers.gillespie import GillespieSolver


def test_gillespie_single_step_execution():
    """
    Verify that a single execution step updates the time and changes
    node charges in accordance with the compiled incidence matrix.
    """
    yaml_circuit = """
    schema_version: "1.0.0"
    simulation:
      solver: "gillespie"
      t_finish: 1.0e-6
      seed: 42
    nodes:
      free: [{"name": "out"}]
      regulated: [{"name": "gnd", "type": "constant", "value": 0.0}]
    components:
      - type: "capacitor"
        name: "C1"
        terminals: ["out", "gnd"]
        specs: {capacitance: 1.0e-15}
      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["out", "gnd"]
        specs: {resistance: 1.0e5}
    """
    assembly = SSECompiler.compile_string(yaml_circuit)

    # Instantiate solver
    from sse_core.compiler.parser import SSEParser

    parsed_netlist = SSEParser.parse_string(yaml_circuit)
    solver = GillespieSolver(parsed_netlist, assembly)

    # State: island starts with 1 excess electron (q = -1 qe)
    # Regulated node is ground (0.0 V)
    q_init = np.array([-1])
    vr = np.array([0.0])

    q_next, dt = solver.execute_step(q_init, vr)

    # Verify transition happened
    assert dt > 0.0
    # Because TJ1 goes from 'out' to 'gnd', the forward tunneling transfers +1 e to 'out'.
    # If the excess electron tunnels off, the excess charge on 'out' goes from -1 to 0.
    assert q_next[0] == 0 or q_next[0] == -2


def test_gillespie_gate_bias_drives_island_toward_induced_charge():
    """
    A gate-biased island with Cg = 2 fF and Vg = 20 mV is not in the
    Coulomb-blockade regime.

    The gate induces an equilibrium excess-electron number of approximately

        n_g = Cg * Vg / e.

    The stochastic trajectory should move from n=0 toward that induced
    charge rather than remain near zero.
    """

    gate_capacitance = 2.0e-15
    gate_voltage = 0.02

    circuit_yaml = f"""
    schema_version: "1.0.0"

    simulation:
      solver: "gillespie"
      t_finish: 1.0e-9
      v_th: 0.001
      seed: 100

    nodes:
      free:
        - name: "out"

      regulated:
        - name: "vg"
          type: "constant"
          value: {gate_voltage}

        - name: "gnd"
          type: "constant"
          value: 0.0

    components:
      - type: "capacitor"
        name: "Cg"
        terminals: ["vg", "out"]
        specs:
          capacitance: {gate_capacitance}

      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["out", "gnd"]
        specs:
          resistance: 1.0e6
    """

    parsed_netlist = SSEParser.parse_string(circuit_yaml)
    assembly = SSECompiler.compile_string(circuit_yaml)
    solver = GillespieSolver(parsed_netlist, assembly)

    q_init = np.array([0])
    vr = np.array([gate_voltage, 0.0])

    history = solver.simulate(q_init, vr)

    charge_states = history["charge"][:, 0]
    induced_electron_number = gate_capacitance * gate_voltage / E_CHARGE

    initial_distance = abs(charge_states[0] - induced_electron_number)
    final_distance = abs(charge_states[-1] - induced_electron_number)

    assert len(charge_states) > 1

    # A single junction event changes the integer state by one.
    assert np.all(np.abs(np.diff(charge_states)) <= 1)

    # The trajectory must move toward the gate-induced charge.
    assert final_distance < initial_distance
    assert charge_states[-1] > charge_states[0]


def test_electrostatic_energy_and_open_circuit_potential():
    """
    Verify the analytical electrostatic energy E and open-circuit
    potential Phi for a single free island coupled to two regulated
    nodes.

    All circuit quantities are expressed directly in SI units.
    """

    c_gate = 2.0e-18
    c_ground = 3.0e-18
    c_total = c_gate + c_ground

    v_gate = 0.012
    v_ground = 0.0

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
        - name: "vg"
          type: "constant"
          value: {v_gate}

        - name: "gnd"
          type: "constant"
          value: {v_ground}

    components:
      - type: "capacitor"
        name: "Cg"
        terminals: ["island", "vg"]
        specs:
          capacitance: {c_gate}

      - type: "capacitor"
        name: "C0"
        terminals: ["island", "gnd"]
        specs:
          capacitance: {c_ground}
    """

    parsed_netlist = SSEParser.parse_string(circuit_yaml)
    assembly = SSECompiler.compile_string(circuit_yaml)
    solver = GillespieSolver(parsed_netlist, assembly)

    # Native simulator state. q = -2 corresponds to physical charge +2e
    # under the current q_physical = -e*q convention.
    q = np.array([-2], dtype=np.int64)
    vr = np.array([v_gate, v_ground], dtype=np.float64)

    q_physical = -E_CHARGE * float(q[0])

    # Effective capacitance between the two regulated terminals.
    c_effective = c_gate * c_ground / c_total

    expected_electrostatic_energy = (
        0.5 * q_physical**2 / c_total + 0.5 * c_effective * (v_gate - v_ground) ** 2
    )

    expected_source_coupling = (
        -v_gate * c_gate / c_total * q_physical
        - v_ground * c_ground / c_total * q_physical
    )

    expected_phi = expected_electrostatic_energy - expected_source_coupling

    actual_electrostatic_energy = solver.compute_electrostatic_energy(q, vr)

    actual_phi = solver.compute_open_circuit_potential(
        q,
        vr,
    )

    assert actual_electrostatic_energy == pytest.approx(
        expected_electrostatic_energy,
        rel=1e-12,
        abs=1e-30,
    )

    assert actual_phi == pytest.approx(
        expected_phi,
        rel=1e-12,
        abs=1e-30,
    )


def test_gillespie_event_ledger_is_self_consistent():
    """
    Every executed event must explicitly account for the observed
    free-node state change and the corresponding direct regulated-node
    carrier transfer.
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
          value: 0.01

        - name: "gnd"
          type: "constant"
          value: 0.0

    components:
      - type: "capacitor"
        name: "Cg"
        terminals: ["island", "vg"]
        specs:
          capacitance: 1.0e-17

      - type: "capacitor"
        name: "C0"
        terminals: ["island", "gnd"]
        specs:
          capacitance: 1.0e-17

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

    history = solver.simulate(
        q_init=np.array([0]),
        vr=np.array([0.01, 0.0]),
        max_steps=25,
    )

    events = history["events"]
    number_of_events = len(events["time"])

    assert number_of_events > 0

    assert events["q_before"].shape == (
        number_of_events,
        1,
    )

    assert events["q_after"].shape == (
        number_of_events,
        1,
    )

    assert events["free_delta_count"].shape == (
        number_of_events,
        1,
    )

    assert events["regulated_delta_count"].shape == (
        number_of_events,
        2,
    )

    # Every ledger state change must equal its recorded free-node
    # incidence vector.
    np.testing.assert_array_equal(
        events["q_after"] - events["q_before"],
        events["free_delta_count"],
    )

    # Each event transfers one carrier between terminals; it cannot
    # create or destroy carrier count.
    total_delta_per_event = np.sum(
        events["free_delta_count"],
        axis=1,
    ) + np.sum(
        events["regulated_delta_count"],
        axis=1,
    )

    np.testing.assert_array_equal(
        total_delta_per_event,
        np.zeros(
            number_of_events,
            dtype=np.int64,
        ),
    )

    # Summed event increments must reproduce the full trajectory's
    # net state change.
    np.testing.assert_array_equal(
        np.sum(
            events["free_delta_count"],
            axis=0,
        ),
        history["charge"][-1] - history["charge"][0],
    )

    expected_selected_rate = np.where(
        events["is_reverse"],
        events["reverse_rate"],
        events["forward_rate"],
    )

    np.testing.assert_allclose(
        events["selected_rate"],
        expected_selected_rate,
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_array_equal(
        events["direction"],
        np.where(
            events["is_reverse"],
            -1,
            1,
        ),
    )

    assert np.all(np.diff(events["time"]) > 0.0)
    assert np.all(events["time"] <= solver.t_finish)

    # Every executed event time must also occur in the state history.
    assert np.all(
        np.isin(
            events["time"],
            history["time"],
        )
    )


def test_gillespie_does_not_execute_event_after_t_finish():
    """
    When the next sampled event lies after t_finish, the island remains
    in its current state and the event ledger stays empty.
    """

    circuit_yaml = """
    schema_version: "1.0.0"

    simulation:
      solver: "gillespie"
      t_finish: 1.0e-12
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
          capacitance: 1.0e-15

      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["island", "gnd"]
        specs:
          resistance: 1.0e30
    """

    parsed_netlist = SSEParser.parse_string(circuit_yaml)

    assembly = SSECompiler.compile_string(circuit_yaml)

    solver = GillespieSolver(
        parsed_netlist,
        assembly,
    )

    history = solver.simulate(
        q_init=np.array([0]),
        vr=np.array([0.0]),
    )

    np.testing.assert_allclose(
        history["time"],
        np.array([0.0, solver.t_finish]),
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_array_equal(
        history["charge"],
        np.array([[0], [0]]),
    )

    assert len(history["events"]["time"]) == 0
    assert history["completed"]
    assert history["termination_reason"] == "t_finish"


def test_gillespie_max_steps_guard():
    """Verify that the simulator respects the max_steps dynamic limit."""
    yaml_circuit = """
    schema_version: "1.0.0"
    simulation: {solver: "gillespie", t_finish: 1.0e-3, seed: 42}
    nodes:
      free: [{"name": "out"}]
      regulated: [{"name": "gnd", "type": "constant", "value": 0.0}]
    components:
      - type: "capacitor"
        name: "C1"
        terminals: ["out", "gnd"]
        specs: {capacitance: 1.0e-15}
      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["out", "gnd"]
        specs: {resistance: 1.0e3} # Low resistance = highly active tunneling
    """
    assembly = SSECompiler.compile_string(yaml_circuit)
    parsed_netlist = SSEParser.parse_string(yaml_circuit)
    solver = GillespieSolver(parsed_netlist, assembly)

    q_init = np.array([0])
    vr = np.array([0.0])

    # Force a max steps limit of 5
    history = solver.simulate(q_init, vr, max_steps=5)

    # Assert we did not run into an infinite loop and capped transitions exactly
    # 1 initial state + 5 transition steps = 6 recorded points
    assert len(history["time"]) <= 6


def test_solver_uses_grounded_body_mosfet_rates():
    """
    A MOSFET with bulk != source must use the grounded-body kernel,
    including direction-specific midpoint voltages for embedded rates.
    """

    v_thermal = 0.026
    v_threshold = 4.0 * v_thermal
    i0 = 1.0e-12

    circuit_yaml = f"""
    schema_version: "1.0.0"

    simulation:
      solver: "gillespie"
      t_finish: 1.0e-9
      v_th: {v_thermal}
      seed: 42

    nodes:
      free:
        - name: "out"

      regulated:
        - name: "gate"
          type: "constant"
          value: {1.4 * v_thermal}

        - name: "source"
          type: "constant"
          value: {-0.2 * v_thermal}

        - name: "bulk"
          type: "constant"
          value: 0.0

    components:
      - type: "capacitor"
        name: "Cout"
        terminals: ["out", "bulk"]
        specs:
          capacitance: 1.0e-16

      - type: "n_channel_mosfet"
        name: "bias_nmos"
        terminals:
          drain: "out"
          gate: "gate"
          source: "source"
          bulk: "bulk"
        specs:
          I0: {i0}
          VT: {v_threshold}
          n: 1.0
    """

    netlist = SSEParser.parse_string(circuit_yaml)
    assembly = SSECompiler.compile_string(circuit_yaml)
    solver = GillespieSolver(netlist, assembly)

    regulated_values = {node.name: node.value for node in netlist.nodes.regulated}

    vr = np.asarray(
        [regulated_values[name] for name in assembly.regulated_names],
        dtype=np.float64,
    )

    q = np.array([0], dtype=np.int64)

    component = solver.active_components[0]
    device = solver.devices[0]

    assert component.terminals.bulk != component.terminals.source
    assert solver._uses_grounded_body_model(component)

    # Fixed-state kernel selection.
    potentials = solver.compute_node_potentials(q, vr)

    actual_forward, actual_reverse, _ = solver.compute_all_rates(potentials)

    expected_forward, expected_reverse = grounded_body_mosfet_rates(
        potentials[component.terminals.drain],
        potentials[component.terminals.gate],
        potentials[component.terminals.source],
        potentials[component.terminals.bulk],
        device.i0,
        device.vt,
        device.n,
        device.v_th,
        device.is_pmos,
    )

    np.testing.assert_allclose(
        actual_forward[0],
        expected_forward,
        rtol=1.0e-12,
        atol=0.0,
    )

    np.testing.assert_allclose(
        actual_reverse[0],
        expected_reverse,
        rtol=1.0e-12,
        atol=0.0,
    )

    # Embedded midpoint evaluation.
    free_delta = np.rint(assembly.free_Delta[:, 0]).astype(np.int64)

    potentials_forward = solver.compute_node_potentials(
        q + free_delta,
        vr,
    )

    potentials_reverse = solver.compute_node_potentials(
        q - free_delta,
        vr,
    )

    def midpoint(after, terminal):
        return 0.5 * (potentials[terminal] + after[terminal])

    expected_embedded_forward = grounded_body_mosfet_rates(
        midpoint(
            potentials_forward,
            component.terminals.drain,
        ),
        midpoint(
            potentials_forward,
            component.terminals.gate,
        ),
        midpoint(
            potentials_forward,
            component.terminals.source,
        ),
        midpoint(
            potentials_forward,
            component.terminals.bulk,
        ),
        device.i0,
        device.vt,
        device.n,
        device.v_th,
        device.is_pmos,
    )[0]

    expected_embedded_reverse = grounded_body_mosfet_rates(
        midpoint(
            potentials_reverse,
            component.terminals.drain,
        ),
        midpoint(
            potentials_reverse,
            component.terminals.gate,
        ),
        midpoint(
            potentials_reverse,
            component.terminals.source,
        ),
        midpoint(
            potentials_reverse,
            component.terminals.bulk,
        ),
        device.i0,
        device.vt,
        device.n,
        device.v_th,
        device.is_pmos,
    )[1]

    (
        embedded_forward,
        embedded_reverse,
        _,
    ) = solver.compute_embedded_rates(q, vr)

    np.testing.assert_allclose(
        embedded_forward[0],
        expected_embedded_forward,
        rtol=1.0e-12,
        atol=0.0,
    )

    np.testing.assert_allclose(
        embedded_reverse[0],
        expected_embedded_reverse,
        rtol=1.0e-12,
        atol=0.0,
    )
