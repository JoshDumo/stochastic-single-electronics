# tests/test_compiler/test_builder.py
import numpy as np
import pytest
from sse_core.compiler.builder import SSEMatrixBuilder
from sse_core.compiler.parser import SSEParser


def test_builder_correct_assembly_and_inversion():
    """
    Verify the numerical assembly of a simple voltage divider circuit.
    Nodes: 'out' (free), 'gnd' (regulated), 'vdd' (regulated)
    Capacitors:
      C1: 'out' to 'gnd' (2.0e-15 F)
      C2: 'vdd' to 'out' (3.0e-15 F)
    """
    yaml_divider = """
    schema_version: "1.0.0"
    simulation: {solver: "gillespie", t_finish: 1e-6}
    nodes:
      free: [{"name": "out"}]
      regulated:
        - name: "gnd"
          type: "ground"
        - name: "vdd"
          type: "constant"
          value: 1.0
    components:
      - type: "capacitor"
        name: "C1"
        terminals: ["out", "gnd"]
        specs: {capacitance: 2.0e-15}
      - type: "capacitor"
        name: "C2"
        terminals: ["vdd", "out"]
        specs: {capacitance: 3.0e-15}
      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["out", "gnd"]
        specs: {resistance: 1e5}
    """
    netlist = SSEParser.parse_string(yaml_divider)
    builder = SSEMatrixBuilder(netlist)
    assembly = builder.assemble()

    # Verify indices
    assert assembly.free_names == ["out"]
    assert assembly.regulated_names == ["gnd", "vdd"]

    # C_inv: 1 / 5e-15 = 2e14
    assert np.allclose(assembly.C_inv, np.array([[2.0e14]]))

    # Cx: [-2e-15, -3e-15]
    assert np.allclose(assembly.Cx, np.array([[-2.0e-15, -3.0e-15]]))

    # Verify Incidence Matrix (free_Delta)
    # TJ1 goes from 'out' (idx 0, free) to 'gnd' (idx 1, regulated).
    # Forward transition transfers +1 charge to 'out'.
    assert assembly.free_Delta.shape == (1, 1)
    assert assembly.free_Delta[0, 0] == 1.0

    # Verify device_terminals: index pair map
    # 'out' is index 0, 'gnd' is index 1
    assert assembly.device_terminals == [(0, 1)]

    # Verify precomputed voltage delta dV_precomputed:
    # delta_V = C_inv * delta = 2e14 * 1.0 = 2e14 V
    assert np.allclose(assembly.dV_precomputed, np.array([2.0e14]))


def test_builder_raises_err_math_201_on_singular_matrix():
    """Verify uninvertible free nodes trigger ERR_MATH_201."""
    singular_yaml = """
    schema_version: "1.0.0"
    simulation: {solver: "gillespie", t_finish: 1e-6}
    nodes:
      free:
        - name: "out"
        - name: "isolated_node"
      regulated:
        - name: "gnd"
          type: "ground"
    components:
      - type: "capacitor"
        name: "C1"
        terminals: ["out", "gnd"]
        specs: {capacitance: 1e-15}
    """
    netlist = SSEParser.parse_string(singular_yaml)
    builder = SSEMatrixBuilder(netlist)
    with pytest.raises(ValueError) as exc_info:
        builder.assemble()
    assert "ERR_MATH_201" in str(exc_info.value)
