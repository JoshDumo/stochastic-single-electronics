import matplotlib.pyplot as plt
import numpy as np
from sse_core.compiler.builder import SSECompiler

# Adjust these imports based on your actual library structure
from sse_core.compiler.parser import SSEParser
from sse_core.solvers.gillespie import GillespieSolver

PBIT_YAML = """
schema_version: "1.0.0"
simulation:
  solver: "gillespie"
  t_finish: 1.0e-5
  v_th: 0.026
  seed: 42
nodes:
  free:
    - {name: "node3"}
    - {name: "node4"}
  regulated:
    - name: "vdd"
      type: "constant"
      value: 0.0286
    - name: "vss"
      type: "constant"
      value: -0.0286
components:
  # NODE 3 CAPACITORS (Gate of Inv1 + Output of Inv2)
  - type: "capacitor"
    name: "C_gate_1_p"
    terminals: ["vdd", "node3"]
    specs: {capacitance: 50.0e-18}
  - type: "capacitor"
    name: "C_gate_1_n"
    terminals: ["vss", "node3"]
    specs: {capacitance: 50.0e-18}
  - type: "capacitor"
    name: "C_out_2_p"
    terminals: ["vdd", "node3"]
    specs: {capacitance: 0.5e-18}
  - type: "capacitor"
    name: "C_out_2_n"
    terminals: ["vss", "node3"]
    specs: {capacitance: 0.5e-18}

  # NODE 4 CAPACITORS (Output of Inv1 + Gate of Inv2)
  - type: "capacitor"
    name: "C_gate_2_p"
    terminals: ["vdd", "node4"]
    specs: {capacitance: 50.0e-18}
  - type: "capacitor"
    name: "C_gate_2_n"
    terminals: ["vss", "node4"]
    specs: {capacitance: 50.0e-18}
  - type: "capacitor"
    name: "C_out_1_p"
    terminals: ["vdd", "node4"]
    specs: {capacitance: 0.5e-18}
  - type: "capacitor"
    name: "C_out_1_n"
    terminals: ["vss", "node4"]
    specs: {capacitance: 0.5e-18}

  # STAGE 1 INVERTER (Input: node3 -> Output: node4)
  - type: "p_channel_mosfet"
    name: "M_inv1_p"
    terminals: {drain: "node4", gate: "node3", source: "vdd", bulk: "vdd"}
    specs: {I0: 1.6e-7, VT: 0.0, n: 1.0}
  - type: "n_channel_mosfet"
    name: "M_inv1_n"
    terminals: {drain: "node4", gate: "node3", source: "vss", bulk: "vss"}
    specs: {I0: 1.6e-7, VT: 0.0, n: 1.0}

  # STAGE 2 INVERTER (Input: node4 -> Output: node3)
  - type: "p_channel_mosfet"
    name: "M_inv2_p"
    terminals: {drain: "node3", gate: "node4", source: "vdd", bulk: "vdd"}
    specs: {I0: 1.6e-7, VT: 0.0, n: 1.0}
  - type: "n_channel_mosfet"
    name: "M_inv2_n"
    terminals: {drain: "node3", gate: "node4", source: "vss", bulk: "vss"}
    specs: {I0: 1.6e-7, VT: 0.0, n: 1.0}
"""


def test_pbit_bimodal_distribution_regression():
    # 1. Parse and Compile the Blueprint
    parsed_netlist = SSEParser.parse_string(PBIT_YAML)
    assembly = SSECompiler.compile_string(PBIT_YAML)
    solver = GillespieSolver(parsed_netlist, assembly)

    # 2. Setup initial conditions
    # Regulated nodes: [vdd, vss] based on YAML order
    vr = np.array([0.029, -0.0282])

    # Free nodes: [node3, node4] start with 0 excess electrons
    q_init = np.array([0, 0])

    # 3. Run the Gillespie Simulation
    # Running for a healthy amount of steps to capture multiple spontaneous flips
    history = solver.simulate(q_init, vr, max_steps=200000)

    # Extract the potential history for Node 3 (the output we are observing)
    # Assuming the history object returns potentials directly or can be calculated
    v_out = history["potentials"]["node3"]

    # 4. Plotting the Results
    plt.figure(figsize=(8, 5))
    plt.hist(v_out, bins=70, color="mediumpurple", edgecolor="black", alpha=0.8)
    plt.axvline(0, color="black", linestyle="dashed", linewidth=1, label="0V Midpoint")
    plt.xlabel("$V_\\mathrm{out}$ (Node 3 Potential) [V]")
    plt.ylabel("N (Events)")
    plt.title("P-Bit Bimodal Distribution (Spontaneous Flipping)")
    plt.legend()
    plt.savefig("pbit_distribution_debug.png")

    # 5. The Bimodal Pass Criteria
    mean_v = np.mean(v_out)

    # Calculate the probability mass near the rails (e.g., > 10mV and < -10mV)
    prob_high = np.mean(v_out > 0.010)
    prob_low = np.mean(v_out < -0.010)

    # The mean should be centered near 0V
    assert abs(mean_v) < 0.005, (
        f"Distribution mean {mean_v:.4f}V deviated too far from 0V. The system is heavily biased."
    )

    # It must spend a significant amount of time in BOTH states to be a true p-bit
    # We allow a generous margin (e.g., 30% to 70% in each state) to account for stochastic variance
    assert 0.30 < prob_high < 0.70, (
        f"System failed to dwell in the High State. Probability was {prob_high:.2%}"
    )

    assert 0.30 < prob_low < 0.70, (
        f"System failed to dwell in the Low State. Probability was {prob_low:.2%}"
    )
