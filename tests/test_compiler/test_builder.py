# tests/test_compiler/test_builder.py
import numpy as np
import pytest
from sse_core.compiler.builder import SSECompiler, SSEMatrixBuilder
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
          type: "constant"
          value: 0.0
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

    # The total capacitance C_total = 2e-15 + 3e-15 = 5e-15 F
    # C_inv = 1 / C_total = 2e14
    expected_C_inv = np.array([[2.0e14]])
    assert np.allclose(assembly.C_inv, expected_C_inv, rtol=1e-4)

    # Cx is the mutual capacitance matrix (C_out_gnd, C_out_vdd)
    # The builder stamp logic is correct; verify against these actual SI values
    expected_Cx = np.array([[-2.0e-15, -3.0e-15]])
    assert np.allclose(assembly.Cx, expected_Cx, rtol=1e-4)

    # dV = C_inv * Delta (where Delta is 1.0 for the tunneling event)
    # dV = 2e14 * 1.0 = 2e14
    assert np.allclose(assembly.dV_precomputed, np.array([2.0e14]), rtol=1e-4)


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
          type: "constant"
          value: 0.0
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


def test_compiler_end_to_end_orchestration():
    """Verify that SSECompiler correctly orchestrates parsing, linting, and building."""
    valid_yaml = """
    schema_version: "1.0.0"
    simulation: {solver: "gillespie", t_finish: 1e-6}
    nodes:
      free: [{"name": "out"}]
      regulated: [{"name": "gnd", "type": "constant", "value": 0.0}]
    components:
      - type: "capacitor"
        name: "C1"
        terminals: ["out", "gnd"]
        specs: {capacitance: 1e-15}
      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["out", "gnd"]
        specs: {resistance: 1e5}
    """
    # Compile in a single call!
    assembly = SSECompiler.compile_string(valid_yaml)
    assert assembly.free_names == ["out"]

    # 1. Update the C_inv assertion
    # 1 / 1e-15 = 1e15
    assert np.allclose(assembly.C_inv, np.array([[1.0e15]]), rtol=1e-4)
