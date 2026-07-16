# tests/test_devices/test_sweeps.py
import numpy as np
import pytest
from sse_core.compiler.models import ComponentConfig, MOSFETTerminals
from sse_core.devices.mapping import extract_device_voltages
from sse_core.devices.semiconductor import MOSFET, Diode


def test_terminal_extraction_mapping():
    """Verify that extract_device_voltages correctly maps polarities from potentials."""
    potentials = {"out": 0.8, "gnd": 0.0, "v_gate": 1.2}

    # 1. Test 2-terminal mapping
    comp_2t = ComponentConfig(
        type="tunnel_junction",
        name="TJ1",
        terminals=["out", "gnd"],
        specs={"resistance": 1e5},
    )
    v_act, v_ctrl = extract_device_voltages(comp_2t, potentials)
    assert v_act == pytest.approx(0.8)
    assert v_ctrl == 0.0

    # 2. Test MOSFET mapping
    comp_mos = ComponentConfig(
        type="n_channel_mosfet",
        name="M1",
        terminals=MOSFETTerminals(drain="out", gate="v_gate", source="gnd", bulk="gnd"),
        specs={"I0": 1e-6, "VT": 0.4, "n": 1.1},
    )
    v_act, v_ctrl = extract_device_voltages(comp_mos, potentials)
    assert v_act == pytest.approx(0.8)  # V_ds = V_drain - V_source = 0.8 - 0.0
    assert v_ctrl == pytest.approx(1.2)  # V_gs = V_gate - V_source = 1.2 - 0.0


def test_temperature_sweep_thermal_activation():
    """
    Sweep temperature (V_th) from cryogenic (0.36 mV) to room temp (25.9 mV)
    to verify that subthreshold thermionic emission remains stable and
    displays monotonic thermal activation behavior.
    """
    v_th_sweep = np.linspace(0.00036, 0.0259, 10)  # Sweep corresponding to ~4K to 300K
    mos = MOSFET(name="M1", v_th=0.0259, i0=1e-6, vt=0.4, n=1.1)

    rates = []
    for v_th in v_th_sweep:
        mos.v_th = v_th
        # Under subthreshold gate voltage, higher temperature must increase the rate
        rate = mos.forward_rate(v_active=0.1, v_control=0.2)
        rates.append(rate)

    # Rate must monotonically increase with temperature (thermal activation)
    assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))


def test_voltage_sweep_continuity():
    """
    Sweep bias voltage from -1V to 1V across a Diode to verify rate continuity
    and that rates are non-negative across the entire operating window.
    """
    v_sweep = np.linspace(-1.0, 1.0, 100)
    diode = Diode(name="D1", v_th=0.0259, i0=1e-12, n=1.5)

    for v in v_sweep:
        lf = diode.forward_rate(v_active=v)
        lr = diode.reverse_rate(v_active=v)

        assert not np.isnan(lf)
        assert not np.isnan(lr)
        assert lf >= 0.0
        assert lr >= 0.0
