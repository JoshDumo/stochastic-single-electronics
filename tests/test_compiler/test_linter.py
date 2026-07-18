# tests/test_compiler/test_linter.py
from sse_core.compiler.linter import SSETopologyLinter
from sse_core.compiler.parser import SSEParser


def test_linter_passes_valid_circuit():
    """Verify that a fully connected circuit with ground passes linting."""
    valid_yaml = """
    schema_version: "1.0.0"
    simulation:
      solver: "gillespie"
      t_finish: 1e-6
    nodes:
      free:
        - name: "out"
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
        specs: {capacitance: 1e-15}
      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["vdd", "out"]
        specs: {resistance: 1e5}
    """
    netlist = SSEParser.parse_string(valid_yaml)
    linter = SSETopologyLinter(netlist)
    errors = linter.lint()
    assert len(errors) == 0


def test_linter_catches_missing_reference_err_net_104():
    """Verify linter catches absence of ANY regulated reference node."""
    no_ref_yaml = """
    schema_version: "1.0.0"
    simulation: {solver: "gillespie", t_finish: 1e-6}
    nodes:
      free:
        - name: "n1"
        - name: "n2"
      regulated: []  # No reference potential defined
    components:
      - type: "capacitor"
        name: "C1"
        terminals: ["n1", "n2"]
        specs: {capacitance: 1e-15}
    """
    netlist = SSEParser.parse_string(no_ref_yaml)
    linter = SSETopologyLinter(netlist)
    errors = linter.lint()

    # We look for the same ERR_NET_104, just with updated messaging
    # to reflect that any reference node is acceptable
    assert any("ERR_NET_104" in err for err in errors)
    assert "regulated node" in "".join(errors).lower()


def test_linter_catches_dangling_terminal_err_net_101():
    """Verify linter catches components connected to undeclared nodes."""
    dangling_yaml = """
    schema_version: "1.0.0"
    simulation: {solver: "gillespie", t_finish: 1e-6}
    nodes:
      free:
        - name: "out"
      regulated:
        - name: "gnd"
          type: "constant"
          value: 0.0
    components:
      - type: "capacitor"
        name: "C1"
        terminals: ["out", "typo_node"]  # <--- 'typo_node' is not declared
        specs: {capacitance: 1e-15}
    """
    netlist = SSEParser.parse_string(dangling_yaml)
    linter = SSETopologyLinter(netlist)
    errors = linter.lint()
    assert any("ERR_NET_101" in err for err in errors)
    assert "typo_node" in "".join(errors)
