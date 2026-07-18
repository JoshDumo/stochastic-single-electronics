import matplotlib.pyplot as plt
import numpy as np
import pytest
from sse_core.compiler.builder import SSECompiler
from sse_core.compiler.parser import SSEParser
from sse_core.solvers.gillespie import GillespieSolver


def test_example_inverter_gaussian_distribution_regression():
    inverter_yaml = """
    schema_version: "1.0.0"
    simulation:
      solver: "gillespie"
      t_finish: 5.0e-7
      v_th: 0.023
      seed: 999
    nodes:
      free: [{"name": "out"}]
      regulated:
        - name: "vdd"
          type: "constant"
          value: 0.115
        - name: "vin"
          type: "constant"
          value: 0.005
        - name: "vss"
          type: "constant"
          value: -0.115
        - name: "gnd"
          type: "ground"
    components:
      # These represent the load of the NEXT inverter stage
      - type: "capacitor"
        name: "C_gate_p"
        terminals: ["vin", "out"]  # Changed from vdd to vin
        specs: {capacitance: 5.0e-17}
      - type: "capacitor"
        name: "C_gate_n"
        terminals: ["vin", "out"]  # Changed from vss to vin
        specs: {capacitance: 5.0e-17}
      # Parasitic output capacitances
      - type: "capacitor"
        name: "C_out_p"
        terminals: ["vdd", "out"] 
        specs: {capacitance: 5.0e-17}
      - type: "capacitor"
        name: "C_out_n"
        terminals: ["out", "vss"] # Changed to match the ordering pattern of the P-type
        specs: {capacitance: 5.0e-17}
      - type: "p_channel_mosfet"
        name: "M_pullup"
        terminals: {drain: "out", gate: "vin", source: "vdd", bulk: "vdd"}
        specs: {I0: 1.6e-7, VT: -0.0, n: 1.0}
      - type: "n_channel_mosfet"
        name: "M_pulldown"
        terminals: {drain: "out", gate: "vin", source: "vss", bulk: "vss"}
        specs: {I0: 1.6e-7, VT: 0.0, n: 1.0}
    """
    assembly = SSECompiler.compile_string(inverter_yaml)
    parsed_netlist = SSEParser.parse_string(inverter_yaml)
    solver = GillespieSolver(parsed_netlist, assembly)

    vr = np.array([0.115, 0.0005, -0.115])
    q_init = np.array([0])

    history = solver.simulate(q_init, vr, max_steps=200000)
    charges = history["charge"][:, 0]

    plt.figure(figsize=(8, 5))
    plt.hist(charges, bins=50, color="skyblue", edgecolor="black")
    plt.axvline(
        np.mean(charges),
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {np.mean(charges):.2f}",
    )
    plt.title("Distribution of Excess Electrons on Island")
    plt.xlabel("Electron Count")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("distribution_debug.png")

    # The simulator counts EXCESS ELECTRONS.
    # The notebook tracked charge in +e, so -45e means 45 electrons.
    Expected_Electrons = 45.0
    empirical_mean = np.mean(charges)

    assert empirical_mean == pytest.approx(Expected_Electrons, abs=10.0), (
        f"Distribution mean {empirical_mean} deviated too far from {Expected_Electrons}"
    )
