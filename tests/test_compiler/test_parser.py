# tests/test_compiler/test_parser.py
import pytest
from pydantic import ValidationError
from sse_core.compiler.parser import SSEParser


# =============================================================================
# HAPPY PATH TEST
# =============================================================================
# =============================================================================
# HAPPY PATH TEST
# =============================================================================
def test_parser_valid_minimal_netlist():
    """Verify that a standard valid YAML string parses using constant 0V references."""
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
        - name: "gnd"          # Still named 'gnd' for clarity
          type: "constant"     # No longer 'ground'
          value: 0.0
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
    assert netlist.nodes.regulated[0].type == "constant"
    assert netlist.nodes.regulated[0].value == 0.0


# =============================================================================
# ERR_CFG_001: Negative physical parameter checks
# =============================================================================
def test_parser_err_cfg_001_negative_parameter():
    """Verify that negative physical parameters trigger ERR_CFG_001."""
    invalid_yaml = """
    schema_version: "1.0.0"
    simulation:
      solver: "gillespie"
      t_finish: 1.0e-6
    nodes:
      free: [{"name": "out"}]
      regulated: [{"name": "gnd", "type": "constant", "value": 0.0}]
    components:
      - type: "capacitor"
        name: "C1"
        terminals: ["out", "gnd"]
        specs:
          capacitance: -1.0e-15  # <--- PHYSICALLY IMPOSSIBLE
    """
    with pytest.raises(ValidationError) as exc_info:
        SSEParser.parse_string(invalid_yaml)
    assert "ERR_CFG_001" in str(exc_info.value)


# =============================================================================
# ERR_CFG_002: Missing parameter fields
# =============================================================================
@pytest.mark.parametrize(
    "missing_param_yaml,expected_err",
    [
        (
            # Missing "value" for constant node
            """
            schema_version: "1.0.0"
            simulation: {solver: "gillespie", t_finish: 1e-6}
            nodes:
              free: [{"name": "out"}]
              regulated: [{"name": "vdd", "type": "constant"}] # <--- Missing 'value'
            components: []
            """,
            "ERR_CFG_002",
        ),
        (
            # Missing "time_step" when choosing tau_leaping
            """
            schema_version: "1.0.0"
            simulation:
              solver: "tau_leaping"  # <--- Needs time_step!
              t_finish: 1.0e-6
            nodes:
              free: [{"name": "out"}]
              regulated: [{"name": "gnd", "type": "constant", "value": 0.0}]
            components: []
            """,
            "ERR_CFG_002",
        ),
    ],
)
def test_parser_err_cfg_002_missing_parameter(missing_param_yaml, expected_err):
    """Verify missing structural parameters trigger ERR_CFG_002."""
    with pytest.raises(ValidationError) as exc_info:
        SSEParser.parse_string(missing_param_yaml)
    assert expected_err in str(exc_info.value)


# =============================================================================
# ERR_CFG_003: Illegal parameter configuration (Repurposed)
# =============================================================================
def test_parser_err_cfg_003_illegal_constant_specs():
    """Verify constant nodes with illegal specifications (like specs blocks) trigger ERR_CFG_003."""
    invalid_yaml = """
    schema_version: "1.0.0"
    simulation: {solver: "gillespie", t_finish: 1e-6}
    nodes:
      free: [{"name": "out"}]
      regulated:
        - name: "vdd"
          type: "constant"
          value: 1.0
          specs: {amplitude: 1.0} # <--- Illegal: constant sources cannot have specs
    components: []
    """
    with pytest.raises(ValidationError) as exc_info:
        SSEParser.parse_string(invalid_yaml)
    assert "ERR_CFG_003" in str(exc_info.value)


# =============================================================================
# ERR_CFG_004: Unphysical MOSFET threshold voltages
# =============================================================================
def test_parser_err_cfg_004_invalid_mosfet_threshold_polarity():
    """Verify that an nMOS with a negative threshold voltage triggers ERR_CFG_004."""
    invalid_yaml = """
    schema_version: "1.0.0"
    simulation: {solver: "gillespie", t_finish: 1.0e-6}
    nodes:
      free: [{"name": "out"}]
      regulated: [{"name": "gnd", "type": "constant", "value": 0.0}]
    components:
      - type: "n_channel_mosfet"
        name: "M1"
        terminals:
          drain: "out"
          gate: "gnd"
          source: "gnd"
          bulk: "gnd"
        specs:
          I0: 1.0e-6
          VT: -0.4    # <--- PHYSICALLY INVALID for nMOS!
          n: 1.1
    """
    with pytest.raises(ValidationError) as exc_info:
        SSEParser.parse_string(invalid_yaml)
    assert "ERR_CFG_004" in str(exc_info.value)


# =============================================================================
# ERR_NET_103: Node name collisions (Parser boundary check)
# =============================================================================
def test_parser_err_net_103_node_name_collision():
    """Verify duplicate names across free and regulated node domains trigger ERR_NET_103."""
    collision_yaml = """
    schema_version: "1.0.0"
    simulation: {solver: "gillespie", t_finish: 1.0e-6}
    nodes:
      free: [{"name": "out"}]
      regulated:
        - name: "out"       # <--- COLLISION!
          type: "constant"
          value: 0.0
    components: []
    """
    with pytest.raises(ValidationError) as exc_info:
        SSEParser.parse_string(collision_yaml)
    assert "ERR_NET_103" in str(exc_info.value)
