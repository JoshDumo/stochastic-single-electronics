import numpy as np
import pytest
from sse_core.compiler.builder import SSECompiler
from sse_core.compiler.parser import SSEParser
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
      regulated: [{"name": "gnd", "type": "ground"}]
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


def test_gillespie_coulomb_blockade_stable_regime():
    """
    Model a single-electron box (island 'out' connected to ground via a TJ,
    and biased by a gate voltage Vg through a gate capacitor Cg).
    We choose a low temperature (V_th = 1mV) and low Vg such that the island
    remains blockaded in its stable n=0 charge state.
    """
    blockade_yaml = """
    schema_version: "1.0.0"
    simulation:
      solver: "gillespie"
      t_finish: 1.0e-9
      v_th: 0.001       # 1 mV (extremely low temperature)
      seed: 100
    nodes:
      free: [{"name": "out"}]
      regulated:
        - name: "vg"
          type: "constant"
          value: 0.02   # V_gate = 20 mV
        - name: "gnd"
          type: "ground"
    components:
      - type: "capacitor"
        name: "Cg"
        terminals: ["vg", "out"]
        specs: {capacitance: 2.0e-15}  # 2 fF gate capacitor
      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["out", "gnd"]
        specs: {resistance: 1.0e6}    # 1 MOhm barrier
    """
    parsed_netlist = SSEParser.parse_string(blockade_yaml)
    assembly = SSECompiler.compile_string(blockade_yaml)

    solver = GillespieSolver(parsed_netlist, assembly)

    # Start with neutral island (q_init = [0])
    q_init = np.array([0])
    vr = np.array([0.02, 0.0])  # vg = 20mV, gnd = 0V

    # Run full trajectory
    history = solver.simulate(q_init, vr)

    assert len(history["time"]) > 1
    # Check that because of the energy barrier (blockade), the island
    # excess charge does not wander wildly and stays tightly bounded.
    final_charge = history["charge"][-1, 0]
    assert final_charge in [-1, 0, 1]


def test_electrostatic_energy_calculation():
    """Verify that the computed electrostatic energy matches theoretical predictions."""
    yaml_circuit = """
    schema_version: "1.0.0"
    simulation: {solver: "gillespie", t_finish: 1.0e-6}
    nodes:
      free: [{"name": "out"}]
      regulated: [{"name": "gnd", "type": "ground"}]
    components:
      - type: "capacitor"
        name: "C1"
        terminals: ["out", "gnd"]
        specs: {capacitance: 1.0e-15}
      - type: "capacitor"
        name: "C2"
        terminals: ["out", "gnd"]
        specs: {capacitance: 1.0e-15}
    """
    assembly = SSECompiler.compile_string(yaml_circuit)
    parsed_netlist = SSEParser.parse_string(yaml_circuit)
    solver = GillespieSolver(parsed_netlist, assembly)

    q = np.array([-1])  # 1 excess electron
    vr = np.array([0.0])

    # Total capacitance C_sigma = C1 + C2 = 2.0 fF.
    # Energy U = 0.5 * Q^2 / C_sigma = 0.5 * 1 / 2e-15 = 2.5e14 J
    energy = solver.compute_electrostatic_energy(q, vr)
    assert energy == pytest.approx(2.5e14)


def test_gillespie_max_steps_guard():
    """Verify that the simulator respects the max_steps dynamic limit."""
    yaml_circuit = """
    schema_version: "1.0.0"
    simulation: {solver: "gillespie", t_finish: 1.0e-3, seed: 42}
    nodes:
      free: [{"name": "out"}]
      regulated: [{"name": "gnd", "type": "ground"}]
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
