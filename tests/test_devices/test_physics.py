# tests/test_devices/test_physics.py
import numpy as np
import pytest
from sse_core.devices.passive import TunnelJunction


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
