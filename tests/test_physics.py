import numpy as np

from rockphys.physics import gassmann_fwd, gassmann_inv, poisson


def test_gassmann_inverse_exposes_negative_dry_modulus_for_qc():
    kd_k0 = gassmann_inv(
        Ksat=np.array([20.0, 10.0]),
        K0=np.array([36.0, 36.0]),
        Kfl=2.8,
        phi=np.array([0.05, 0.25]),
    )

    assert kd_k0[0] < 0.0
    assert kd_k0[1] > 0.0


def test_forward_gassmann_and_poisson_outputs_remain_finite():
    kd_k0 = np.array([0.20, 0.35])
    k_sat = gassmann_fwd(
        Kd_K0=kd_k0,
        K0=np.array([36.0, 30.0]),
        Kfl_new=1.0,
        phi=np.array([0.20, 0.30]),
    )
    vp = np.array([2.6, 3.1])
    vs = np.array([1.3, 1.7])

    assert np.all(np.isfinite(k_sat))
    assert np.all(k_sat > 0.0)
    assert np.all(np.isfinite(poisson(vp, vs)))
