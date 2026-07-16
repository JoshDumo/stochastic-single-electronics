# tests/test_solvers/test_verifications.py
import numpy as np
import pytest
from sse_core.compiler.builder import SSECompiler
from sse_core.compiler.parser import SSEParser
from sse_core.solvers.gillespie import GillespieSolver
from sse_core.solvers.verification import (
    audit_first_law,
    verify_charge_flux_conservation,  # <--- Added!
    verify_second_law,
)

# In tests/test_solvers/test_verifications.py


def test_thermodynamic_laws_conservation():
    """
    Simulate a Single-Electron Box and perform a full thermodynamic audit
    to verify that the First and Second Laws are satisfied to machine precision.
    """
    seb_yaml = """
    schema_version: "1.0.0"
    simulation:
      solver: "gillespie"
      t_finish: 1.0e-3
      v_th: 0.0259      # Room temperature
      seed: 42
    nodes:
      free: [{"name": "island"}]
      regulated:
        - name: "vg"
          type: "constant"
          value: 0.2
        - name: "gnd"
          type: "ground"
    components:
      - type: "capacitor"
        name: "Cg"
        terminals: ["vg", "island"]
        specs: {capacitance: 1.0}  # <-- Normalized to 1.0 Farad-equivalent
      - type: "capacitor"
        name: "C1"
        terminals: ["island", "gnd"]
        specs: {capacitance: 1.0}  # <-- Normalized to 1.0 Farad-equivalent
      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["island", "gnd"]
        specs: {resistance: 1.0e4}
    """
    parsed_netlist = SSEParser.parse_string(seb_yaml)
    assembly = SSECompiler.compile_string(seb_yaml)
    solver = GillespieSolver(parsed_netlist, assembly)

    # Force a starting non-equilibrium state to induce active tunneling events
    q_init = np.array([-2])
    vr = np.array([0.2, 0.0])  # vg = 0.2V, gnd = 0V

    # Run the simulation
    history = solver.simulate(q_init, vr, max_steps=50)

    # Perform First Law Audit
    audit = audit_first_law(history, assembly, vr)

    # Assert First Law: ΔU + Q_diss == W_sources
    assert audit["discrepancy"] == pytest.approx(0.0, abs=1e-5)

    # Assert Second Law: Total dissipation is non-negative
    assert verify_second_law(history, assembly, vr) is True

    # Assert Charge Flux Balance matches transition history!
    assert verify_charge_flux_conservation(history, assembly) is True
