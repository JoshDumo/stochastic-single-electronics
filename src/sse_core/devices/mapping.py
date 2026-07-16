# src/sse_core/devices/mapping.py
from sse_core.compiler.models import ComponentConfig, MOSFETTerminals


def extract_device_voltages(
    comp: ComponentConfig, node_potentials: dict[str, float]
) -> tuple[float, float]:
    """
    Extracts the active and control voltages for a component from the current
    dictionary of node potentials, respecting physical terminal polarities.

    Returns:
        tuple[v_active, v_control]
    """
    if isinstance(comp.terminals, list):
        # Two-terminal devices (Tunnel Junction, Diode)
        node_a, node_b = comp.terminals
        v_active = node_potentials.get(node_a, 0.0) - node_potentials.get(node_b, 0.0)
        v_control = 0.0
    elif isinstance(comp.terminals, MOSFETTerminals):
        # Four-terminal devices (MOSFET)
        terms: MOSFETTerminals = comp.terminals
        v_drain = node_potentials.get(terms.drain, 0.0)
        v_gate = node_potentials.get(terms.gate, 0.0)
        v_source = node_potentials.get(terms.source, 0.0)

        v_active = v_drain - v_source
        v_control = v_gate - v_source
    else:
        raise TypeError(
            f"Unknown terminal configuration type for component '{comp.name}'."
        )

    return v_active, v_control
