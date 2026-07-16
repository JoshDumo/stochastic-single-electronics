# tests/test_compiler/test_parser.py
import pytest
from pydantic import ValidationError
from sse_core.compiler.parser import SSEParser


def test_parser_valid_minimal_netlist():
    """Verify that a standard valid YAML string parses without issues."""
    valid_yaml = """
    schema_version: "1.0.0"
    simulation:
      solver: "gillespie"
      t_finish: 1.0e-6
      v_th: 0.0259
    nodes:
      free:
        - name: "out"
          initial_charge: 0
      regulated:
        - name: "gnd"
          type: "ground"
        - name: "vdd"
          type: "constant"
          value: 1.0
    components:
      - type: "capacitor"
        name: "C_load"
        terminals: ["out", "gnd"]
        specs:
          capacitance: 1.0e-15
    """
    netlist = SSEParser.parse_string(valid_yaml)
    assert netlist.schema_version == "1.0.0"
    assert netlist.simulation.solver == "gillespie"
    assert len(netlist.nodes.free) == 1
    assert len(netlist.nodes.regulated) == 2
    assert len(netlist.components) == 1
    assert netlist.components[0].specs["capacitance"] == 1.0e-15


def test_parser_invalid_mosfet_threshold_polarity():
    """Verify that an nMOS with a negative threshold voltage triggers a physical bound validation error."""
    invalid_yaml = """
    schema_version: "1.0.0"
    simulation:
      solver: "gillespie"
      t_finish: 1.0e-6
    nodes:
      free:
        - name: "out"
      regulated:
        - name: "gnd"
          type: "ground"
        - name: "v_in"
          type: "constant"
          value: 1.0
    components:
      - type: "n_channel_mosfet"
        name: "M1"
        terminals:
          drain: "out"
          gate: "v_in"
          source: "gnd"
          bulk: "gnd"
        specs:
          I0: 1.0e-6
          VT: -0.4    # <--- PHYSICALLY INVALID for nMOS in our schema rules!
          n: 1.1
    """
    with pytest.raises(ValidationError) as exc_info:
        SSEParser.parse_string(invalid_yaml)
    assert "unphysical negative threshold voltage" in str(exc_info.value)


def test_parser_node_name_collision():
    """Verify that duplicate names across free and regulated nodes raise a collision error."""
    collision_yaml = """
    schema_version: "1.0.0"
    simulation:
      solver: "gillespie"
      t_finish: 1.0e-6
    nodes:
      free:
        - name: "out"       # <--- Collision!
      regulated:
        - name: "out"       # <--- Collision!
          type: "ground"
    components: []
    """
    with pytest.raises(ValidationError) as exc_info:
        SSEParser.parse_string(collision_yaml)
    assert "Name collision detected" in str(exc_info.value)
