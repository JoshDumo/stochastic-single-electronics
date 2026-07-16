# tests/test_devices/test_physics.py
import numpy as np
import pytest
from sse_core.devices.passive import TunnelJunction
from sse_core.devices.semiconductor import MOSFET, Diode


@pytest.mark.parametrize("v_th", [0.0259, 0.0350])
@pytest.mark.parametrize("v_act", [-0.15, -0.05, 0.05, 0.15])
def test_tunnel_junction_ldb(v_th, v_act):
    """
    Verify that the Tunnel Junction rate calculations satisfy standard LDB:
    ln( lf(V) / lr(V) ) = V / V_th
    """
    tj = TunnelJunction(name="TJ_Test", v_th=v_th, resistance=1.0e5)

    # CRITICAL: Evaluate both rates at the same bias point!
    lf = tj.forward_rate(v_active=v_act)
    lr = tj.reverse_rate(v_active=v_act)

    calculated_ratio = np.log(lf / lr)
    expected_ratio = v_act / v_th

    assert calculated_ratio == pytest.approx(expected_ratio, rel=1e-6)


def test_tunnel_junction_zero_bias_stability():
    """Verify the singularity guard prevents division-by-zero or NaN at exactly 0V."""
    v_th = 0.0259
    tj = TunnelJunction(name="TJ_Zero_Test", v_th=v_th, resistance=1.0e5)

    lf = tj.forward_rate(v_active=0.0)
    lr = tj.reverse_rate(v_active=0.0)

    assert not np.isnan(lf)
    assert not np.isnan(lr)
    assert lf > 0.0
    assert lr > 0.0
    # At 0V bias, forward and reverse transitions must be identical by symmetry
    assert lf == pytest.approx(lr, rel=1e-6)


# =============================================================================
# Diode Tests
# =============================================================================
def test_diode_zero_bias_symmetry():
    """Verify that diode rates are symmetric and stable at exactly 0V."""
    diode = Diode(name="D1", v_th=0.0259, i0=1e-12, n=1.5)
    lf = diode.forward_rate(v_active=0.0)
    lr = diode.reverse_rate(v_active=0.0)

    assert not np.isnan(lf)
    assert not np.isnan(lr)
    assert lf == pytest.approx(lr, rel=1e-6)


# =============================================================================
# MOSFET Tests
# =============================================================================
@pytest.mark.parametrize("is_pmos", [False, True])
def test_mosfet_gate_control_scaling(is_pmos):
    """
    Verify that increasing gate overdrive exponentially scales the transition rates
    according to subthreshold thermionic scaling.
    """
    vt = -0.4 if is_pmos else 0.4
    mos = MOSFET(name="M1", v_th=0.0259, i0=1e-6, vt=vt, n=1.1, is_pmos=is_pmos)

    # Define weak vs strong gate control voltages
    if not is_pmos:
        v_gs_weak = 0.0
        v_gs_strong = 0.2
    else:
        v_gs_weak = 0.0
        v_gs_strong = -0.2

    v_ds = -0.1 if is_pmos else 0.1

    rate_weak = mos.forward_rate(v_active=v_ds, v_control=v_gs_weak)
    rate_strong = mos.forward_rate(v_active=v_ds, v_control=v_gs_strong)

    # Assert strong gate overdrive results in a vastly higher tunneling rate
    assert rate_strong > rate_weak * 10.0
