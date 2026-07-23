import numpy as np
import pytest
from hos.toolboxes import spherical as sph


def make_grid(N=2):
    naz = max(4 * (N + 1), 12)
    nel = max(2 * (N + 1), 6)
    az = np.linspace(0, 2 * np.pi, naz, endpoint=False)
    el = np.linspace(-np.pi / 2 * 0.85, np.pi / 2 * 0.85, nel)
    az_g, el_g = np.meshgrid(az, el)
    r_g = np.ones_like(az_g)
    return np.stack([az_g.ravel(), el_g.ravel(), r_g.ravel()], axis=1)


# ── nmACN ─────────────────────────────────────────────────────────────────────

def test_nmACN_formula():
    n, m, acn = sph.nmACN(3)
    np.testing.assert_array_equal(acn, n**2 + n + m)


def test_nmACN_full_order_size():
    N = 4
    n, m, acn = sph.nmACN(N)
    assert len(n) == (N + 1) ** 2


def test_nmACN_maxM_zero_returns_m0_only():
    n, m, acn = sph.nmACN(4, maxM=0)
    np.testing.assert_array_equal(m, 0)
    assert len(n) == 5


def test_nmACN_maxM_one_limits_degrees():
    n, m, acn = sph.nmACN(4, maxM=1)
    assert np.all(np.abs(m) <= 1)


def test_nmACN_non_negative_and_monotonic():
    n, m, acn = sph.nmACN(3)
    assert np.all(acn >= 0)
    assert list(acn) == sorted(acn)


# ── Nkr / Nkr_f ──────────────────────────────────────────────────────────────

def test_Nkr_round_trip():
    N = 4
    r = 0.0875
    c = 343
    f = sph.Nkr(N, r=r, c=c)
    r_rec = sph.Nkr_f(N, f, c=c)
    assert r_rec == pytest.approx(r, rel=1e-6)


def test_Nkr_scales_with_order():
    assert sph.Nkr(4) > sph.Nkr(2)


# ── sphHarm ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["realsn3d", "realn3d", "complex"])
def test_sphHarm_shape(kind):
    pos = make_grid(N=2)
    Y = sph.sphHarm(pos, N=2, kind=kind)
    assert Y.shape == (pos.shape[0], 9)


@pytest.mark.parametrize("kind,is_complex", [
    ("complex", True),
    ("realn3d", False),
    ("realsn3d", False),
])
def test_sphHarm_dtype(kind, is_complex):
    pos = make_grid(N=1)
    Y = sph.sphHarm(pos, N=1, kind=kind)
    assert np.iscomplexobj(Y) == is_complex


def test_sphHarm_invalid_kind_raises():
    with pytest.raises(ValueError):
        sph.sphHarm(make_grid(N=1), N=1, kind="badkind")


def test_sphHarm_order_zero_is_constant():
    pos = make_grid(N=0)
    Y = sph.sphHarm(pos, N=0, kind="realsn3d")
    assert Y.shape == (pos.shape[0], 1)
    np.testing.assert_allclose(Y[:, 0], Y[0, 0], rtol=1e-10)


# ── iSHT / SHT round-trip ─────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("kind", ["realsn3d", "complex"])
def test_isht_sht_round_trip(kind):
    N = 2
    T = 10
    pos = make_grid(N=N)
    rng = np.random.default_rng(42)
    if kind == "realsn3d":
        data_nm_truth = rng.standard_normal(((N + 1) ** 2, T))
    else:
        data_nm_truth = (rng.standard_normal(((N + 1) ** 2, T))
                         + 1j * rng.standard_normal(((N + 1) ** 2, T)))
    _, data = sph.SHT(data_nm_truth, pos, N, kind=kind)
    _, _, data_nm_rec = sph.iSHT(data, pos, N, kind=kind, beta=None)
    np.testing.assert_allclose(data_nm_rec, data_nm_truth, atol=1e-8)


def test_isht_too_few_dims_raises():
    pos = make_grid(N=1)
    with pytest.raises(ValueError):
        sph.iSHT(np.ones(pos.shape[0]), pos, N=1)


def test_isht_too_many_dims_raises():
    pos = make_grid(N=1)
    with pytest.raises(ValueError):
        sph.iSHT(np.ones((pos.shape[0], 2, 3, 4)), pos, N=1)


# ── iSHTmagls smoke tests ─────────────────────────────────────────────────────

def test_isht_magls_shapes_2d():
    N = 2
    T = 64
    pos = make_grid(N=N)
    Q = pos.shape[0]
    numCoeffs = (N + 1) ** 2
    data = np.random.default_rng(50).standard_normal((Q, T))
    Ynm, YnmInv, data_nm_out = sph.iSHTmagls(data, pos, N)
    assert Ynm.shape == (Q, numCoeffs)
    assert YnmInv.shape == (numCoeffs, Q)
    assert data_nm_out.shape == (numCoeffs, T)


def test_isht_magls_shapes_3d():
    N = 2
    T = 64
    chs = 2
    pos = make_grid(N=N)
    Q = pos.shape[0]
    numCoeffs = (N + 1) ** 2
    data = np.random.default_rng(51).standard_normal((Q, chs, T))
    Ynm, YnmInv, data_nm_out = sph.iSHTmagls(data, pos, N)
    assert Ynm.shape == (Q, numCoeffs)
    assert YnmInv.shape == (numCoeffs, Q)
    assert data_nm_out.shape == (numCoeffs, chs, T)


# ── rotateCoefficients ────────────────────────────────────────────────────────

def test_rotateCoefficients_identity():
    N = 2
    data_nm = np.random.default_rng(1).standard_normal(((N + 1) ** 2, 5))
    data_nm_rot = sph.rotateCoefficients(data_nm, [0, 0, 0], seq="zyx", kind="realsn3d")
    np.testing.assert_allclose(data_nm_rot, data_nm, atol=1e-12)


def test_rotateCoefficients_full_rotation_returns_same():
    N = 2
    data_nm = np.random.default_rng(2).standard_normal(((N + 1) ** 2, 5))
    data_nm_rot = sph.rotateCoefficients(
        data_nm, [2 * np.pi, 0, 0], seq="zyx", kind="realsn3d"
    )
    np.testing.assert_allclose(data_nm_rot, data_nm, atol=1e-10)


def test_rotateCoefficients_invalid_kind_raises():
    with pytest.raises(ValueError):
        sph.rotateCoefficients(np.ones((9, 4)), [0, 0, 0], kind="badkind")


# ── rotateSphPoints ───────────────────────────────────────────────────────────

def test_rotateSphPoints_north_pole_invariant_under_azimuthal_rotation():
    coords = np.array([[0.0, np.pi / 2, 1.0], [np.pi, np.pi / 2, 1.0]])
    coords_rot = sph.rotateSphPoints(coords, [np.pi / 3, 0, 0], seq="zyx")
    np.testing.assert_allclose(coords_rot[:, 1], np.pi / 2, atol=1e-10)


def test_rotateSphPoints_south_pole_invariant_under_azimuthal_rotation():
    coords = np.array([[0.0, -np.pi / 2, 1.0]])
    coords_rot = sph.rotateSphPoints(coords, [np.pi / 4, 0, 0], seq="zyx")
    np.testing.assert_allclose(coords_rot[:, 1], -np.pi / 2, atol=1e-10)


def test_rotateSphPoints_radius_preserved():
    rng = np.random.default_rng(3)
    az = rng.uniform(0, 2 * np.pi, 20)
    el = rng.uniform(-np.pi / 2 * 0.9, np.pi / 2 * 0.9, 20)
    r = rng.uniform(0.5, 2.0, 20)
    coords = np.stack([az, el, r], axis=1)
    coords_rot = sph.rotateSphPoints(coords, [0.5, 0.3, 0.1], seq="zyx")
    np.testing.assert_allclose(coords_rot[:, 2], r, rtol=1e-10)


def test_rotateSphPoints_azimuth_in_range():
    rng = np.random.default_rng(4)
    az = rng.uniform(0, 2 * np.pi, 50)
    el = rng.uniform(-np.pi / 2 * 0.9, np.pi / 2 * 0.9, 50)
    coords = np.stack([az, el], axis=1)
    coords_rot = sph.rotateSphPoints(coords, [1.2, 0.7, 0.3], seq="zyx")
    assert np.all(coords_rot[:, 0] >= 0)
    assert np.all(coords_rot[:, 0] < 2 * np.pi)


def test_rotateSphPoints_front_to_left_under_90deg_yaw():
    coords = np.array([[0.0, 0.0]])
    coords_rot = sph.rotateSphPoints(coords, [np.pi / 2, 0, 0], seq="zyx")
    np.testing.assert_allclose(coords_rot[0, 0], np.pi / 2, atol=1e-10)
    np.testing.assert_allclose(coords_rot[0, 1], 0.0, atol=1e-10)


def test_rotateSphPoints_round_trip():
    rng = np.random.default_rng(5)
    az = rng.uniform(0, 2 * np.pi, 15)
    el = rng.uniform(-np.pi / 2 * 0.8, np.pi / 2 * 0.8, 15)
    coords = np.stack([az, el], axis=1)
    rot = [0.6, 0.3, 0.1]
    coords_fwd = sph.rotateSphPoints(coords, rot, seq="zyx")
    # Inverse of extrinsic "zyx" [a,b,c] is extrinsic "xyz" [-c,-b,-a]
    coords_back = sph.rotateSphPoints(
        coords_fwd, [-rot[2], -rot[1], -rot[0]], seq="xyz"
    )

    def to_cart(c):
        az_, el_ = c[:, 0], c[:, 1]
        return np.stack(
            [np.cos(el_) * np.cos(az_), np.cos(el_) * np.sin(az_), np.sin(el_)],
            axis=1,
        )

    np.testing.assert_allclose(to_cart(coords_back), to_cart(coords), atol=1e-10)


def test_rotateSphPoints_isDegrees_flag_consistent():
    coords = np.array([[0.0, 0.0]])
    rot_rad = sph.rotateSphPoints(coords, [np.pi / 4, 0, 0], seq="zyx", isDegrees=False)
    rot_deg = sph.rotateSphPoints(coords, [45.0, 0.0, 0.0], seq="zyx", isDegrees=True)
    np.testing.assert_allclose(rot_rad, rot_deg, atol=1e-12)


def test_rotateSphPoints_optional_radius_column_consistent():
    coords_2col = np.array([[np.pi / 4, np.pi / 6]])
    coords_3col = np.array([[np.pi / 4, np.pi / 6, 1.0]])
    rot_2 = sph.rotateSphPoints(coords_2col, [0.3, 0.0, 0.0], seq="zyx")
    rot_3 = sph.rotateSphPoints(coords_3col, [0.3, 0.0, 0.0], seq="zyx")
    np.testing.assert_allclose(rot_2[:, :2], rot_3[:, :2], atol=1e-12)


# ── decimateCoefficients ──────────────────────────────────────────────────────

def test_decimateCoefficients_default_retains_m0_only():
    N = 3
    data_nm = np.random.default_rng(6).standard_normal(((N + 1) ** 2, 5))
    data_dec = sph.decimateCoefficients(data_nm)
    assert data_dec.shape[0] == N + 1


def test_decimateCoefficients_custom_retain_acn():
    N = 2
    data_nm = np.eye((N + 1) ** 2)[:, :4]
    data_dec = sph.decimateCoefficients(data_nm, retainACN=[0, 1])
    assert data_dec.shape[0] == 2
    np.testing.assert_array_equal(data_dec[0, :], data_nm[0, :])
    np.testing.assert_array_equal(data_dec[1, :], data_nm[1, :])


def test_decimateCoefficients_out_of_bounds_raises():
    with pytest.raises(ValueError):
        sph.decimateCoefficients(np.ones((9, 4)), retainACN=[999])


# ── orthogonalityMatrix ───────────────────────────────────────────────────────

def test_orthogonality_matrix_shape():
    N = 2
    pos = make_grid(N=N)
    Ynm = sph.sphHarm(pos, N=N, kind="realsn3d")
    orthog = sph.orthogonalityMatrix(Ynm)
    assert orthog.shape == ((N + 1) ** 2, (N + 1) ** 2)


def test_orthogonality_matrix_symmetric():
    N = 2
    pos = make_grid(N=N)
    Ynm = sph.sphHarm(pos, N=N, kind="realsn3d")
    orthog = sph.orthogonalityMatrix(Ynm)
    np.testing.assert_allclose(orthog, orthog.conj().T, atol=1e-12)


@pytest.mark.skip(reason="requires SOFA file on disk")
def test_read_sofa_shape():
    pass
