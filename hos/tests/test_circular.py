import numpy as np
import pytest
from hos.toolboxes import circular


def make_az_grid(Q=40):
    return np.linspace(0, 2 * np.pi, Q, endpoint=False)


# ── circHarm ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind,is_complex", [("real", False), ("exp", True)])
def test_circHarm_shape(kind, is_complex):
    Y = circular.circHarm(make_az_grid(20), N=3, kind=kind)
    assert Y.shape == (20, 7)


@pytest.mark.parametrize("kind,is_complex", [("real", False), ("exp", True)])
def test_circHarm_dtype(kind, is_complex):
    Y = circular.circHarm(make_az_grid(), N=2, kind=kind)
    assert np.iscomplexobj(Y) == is_complex


def test_circHarm_invalid_kind_raises():
    with pytest.raises(ValueError):
        circular.circHarm(make_az_grid(), N=2, kind="badkind")


def test_circHarm_non_uniform_azimuth():
    az = np.array([0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2])
    Y = circular.circHarm(az, N=2, kind="real")
    assert Y.shape == (5, 5)


@pytest.mark.parametrize("fn,is_complex", [
    (circular.circHarmReal, False),
    (circular.circHarmExp, True),
])
def test_circHarm_helpers_shape_dtype(fn, is_complex):
    Y = fn(make_az_grid(20), N=3)
    assert Y.shape == (20, 7)
    assert np.iscomplexobj(Y) == is_complex


# ── iCHT / CHT round-trip ─────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["real", "exp"])
def test_icht_cht_round_trip(kind):
    N = 3
    numCoeffs = 2 * N + 1
    T = 8
    az = make_az_grid(Q=60)
    rng = np.random.default_rng(20)
    if kind == "real":
        data_nm_truth = rng.standard_normal((numCoeffs, T))
    else:
        data_nm_truth = (rng.standard_normal((numCoeffs, T))
                         + 1j * rng.standard_normal((numCoeffs, T)))
    _, data = circular.CHT(data_nm_truth, az, N, kind=kind)
    _, _, data_nm_rec = circular.iCHT(data, az, N, kind=kind)
    np.testing.assert_allclose(data_nm_rec, data_nm_truth, atol=1e-8)


def test_icht_cht_round_trip_multichannel():
    N = 2
    numCoeffs = 2 * N + 1
    Q = 30
    T = 5
    chs = 2
    az = make_az_grid(Q=Q)
    rng = np.random.default_rng(22)
    data_nm_truth = rng.standard_normal((numCoeffs, chs, T))
    _, data = circular.CHT(data_nm_truth, az, N, kind="real")
    _, _, data_nm_rec = circular.iCHT(data, az, N, kind="real")
    np.testing.assert_allclose(data_nm_rec, data_nm_truth, atol=1e-8)


# ── dimension validation ───────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", [(40,), (40, 2, 3, 4)])
def test_icht_bad_dims_raises(shape):
    az = make_az_grid()
    with pytest.raises(ValueError):
        circular.iCHT(np.ones(shape), az, N=2)


@pytest.mark.parametrize("shape", [(5,), (5, 2, 3, 4)])
def test_cht_bad_dims_raises(shape):
    az = make_az_grid()
    with pytest.raises(ValueError):
        circular.CHT(np.ones(shape), az, N=2)
