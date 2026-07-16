import numpy as np
import pytest
from hos.toolboxes import spherical as sph


def makeGrid(N=2):
    naz = max(4 * (N + 1), 12)
    nel = max(2 * (N + 1), 6)
    az = np.linspace(0, 2 * np.pi, naz, endpoint=False)
    el = np.linspace(-np.pi / 2 * 0.85, np.pi / 2 * 0.85, nel)
    az_g, el_g = np.meshgrid(az, el)
    r_g = np.ones_like(az_g)
    return np.stack([az_g.ravel(), el_g.ravel(), r_g.ravel()], axis=1)


class TestNmACN:
    def testACNFormula(self):
        n, m, acn = sph.nmACN(3)
        np.testing.assert_array_equal(acn, n**2 + n + m)

    def testFullOrderSize(self):
        N = 4
        n, m, acn = sph.nmACN(N)
        assert len(n) == (N + 1) ** 2

    def testMaxMZeroReturnsOnlyM0(self):
        n, m, acn = sph.nmACN(4, maxM=0)
        np.testing.assert_array_equal(m, 0)
        assert len(n) == 5

    def testMaxMOneLimitsDegrees(self):
        n, m, acn = sph.nmACN(4, maxM=1)
        assert np.all(np.abs(m) <= 1)

    def testACNIsNonNegativeAndMonotonic(self):
        n, m, acn = sph.nmACN(3)
        assert np.all(acn >= 0)
        assert list(acn) == sorted(acn)


class TestNkr:
    def testNkrNkrfRoundTrip(self):
        N = 4
        r = 0.0875
        c = 343
        f = sph.Nkr(N, r=r, c=c)
        r_rec = sph.Nkr_f(N, f, c=c)
        assert r_rec == pytest.approx(r, rel=1e-6)

    def testNkrScalesWithOrder(self):
        f2 = sph.Nkr(2)
        f4 = sph.Nkr(4)
        assert f4 > f2


class TestSphHarm:
    def testShape(self):
        pos = makeGrid(N=2)
        Y = sph.sphHarm(pos, N=2, kind="realsn3d")
        assert Y.shape == (pos.shape[0], 9)

    def testInvalidKindRaises(self):
        pos = makeGrid(N=1)
        with pytest.raises(ValueError):
            sph.sphHarm(pos, N=1, kind="badkind")

    def testComplexKindReturnsDtype(self):
        pos = makeGrid(N=1)
        Y = sph.sphHarm(pos, N=1, kind="complex")
        assert np.iscomplexobj(Y)

    def testRealn3dDtype(self):
        pos = makeGrid(N=1)
        Y = sph.sphHarm(pos, N=1, kind="realn3d")
        assert not np.iscomplexobj(Y)

    def testOrderZeroIsConstant(self):
        pos = makeGrid(N=0)
        Y = sph.sphHarm(pos, N=0, kind="realsn3d")
        assert Y.shape == (pos.shape[0], 1)
        np.testing.assert_allclose(Y[:, 0], Y[0, 0], rtol=1e-10)


class TestISHTSHTRoundTrip:
    @pytest.mark.slow
    def testRealRoundTrip(self):
        N = 2
        T = 10
        pos = makeGrid(N=N)
        rng = np.random.default_rng(42)
        data_nm_truth = rng.standard_normal(((N + 1) ** 2, T))
        _, data = sph.SHT(data_nm_truth, pos, N, kind="realsn3d")
        _, _, data_nm_rec = sph.iSHT(data, pos, N, kind="realsn3d", beta=None)
        np.testing.assert_allclose(data_nm_rec, data_nm_truth, atol=1e-8)

    @pytest.mark.slow
    def testComplexRoundTrip(self):
        N = 2
        T = 8
        pos = makeGrid(N=N)
        rng = np.random.default_rng(0)
        data_nm_truth = (rng.standard_normal(((N + 1) ** 2, T))
                         + 1j * rng.standard_normal(((N + 1) ** 2, T)))
        _, data = sph.SHT(data_nm_truth, pos, N, kind="complex")
        _, _, data_nm_rec = sph.iSHT(data, pos, N, kind="complex", beta=None)
        np.testing.assert_allclose(data_nm_rec, data_nm_truth, atol=1e-8)

    def testISHTInvalidDimsRaises(self):
        pos = makeGrid(N=1)
        with pytest.raises(ValueError):
            sph.iSHT(np.ones((pos.shape[0], 2, 3, 4)), pos, N=1)

    def testISHTTooFewDimsRaises(self):
        pos = makeGrid(N=1)
        with pytest.raises(ValueError):
            sph.iSHT(np.ones(pos.shape[0]), pos, N=1)


class TestRotateCoefficients:
    def testIdentityRotation(self):
        N = 2
        rng = np.random.default_rng(1)
        data_nm = rng.standard_normal(((N + 1) ** 2, 5))
        data_nm_rot = sph.rotateCoefficients(data_nm, [0, 0, 0], seq="zyx", kind="realsn3d")
        np.testing.assert_allclose(data_nm_rot, data_nm, atol=1e-12)

    def testFullRotationReturnsSame(self):
        N = 2
        rng = np.random.default_rng(2)
        data_nm = rng.standard_normal(((N + 1) ** 2, 5))
        data_nm_rot = sph.rotateCoefficients(
            data_nm, [2 * np.pi, 0, 0], seq="zyx", kind="realsn3d"
        )
        np.testing.assert_allclose(data_nm_rot, data_nm, atol=1e-10)

    def testInvalidKindRaises(self):
        data_nm = np.ones((9, 4))
        with pytest.raises(ValueError):
            sph.rotateCoefficients(data_nm, [0, 0, 0], kind="badkind")


class TestRotateSphPoints:
    def testNorthPoleInvariantUnderAzimuthalRotation(self):
        coords = np.array([[0.0, np.pi / 2, 1.0],
                           [np.pi, np.pi / 2, 1.0]])
        coords_rot = sph.rotateSphPoints(coords, [np.pi / 3, 0, 0], seq="zyx")
        np.testing.assert_allclose(coords_rot[:, 1], np.pi / 2, atol=1e-10)

    def testSouthPoleInvariantUnderAzimuthalRotation(self):
        coords = np.array([[0.0, -np.pi / 2, 1.0]])
        coords_rot = sph.rotateSphPoints(coords, [np.pi / 4, 0, 0], seq="zyx")
        np.testing.assert_allclose(coords_rot[:, 1], -np.pi / 2, atol=1e-10)

    def testRadiusPreserved(self):
        rng = np.random.default_rng(3)
        az = rng.uniform(0, 2 * np.pi, 20)
        el = rng.uniform(-np.pi / 2 * 0.9, np.pi / 2 * 0.9, 20)
        r = rng.uniform(0.5, 2.0, 20)
        coords = np.stack([az, el, r], axis=1)
        coords_rot = sph.rotateSphPoints(coords, [0.5, 0.3, 0.1], seq="zyx")
        np.testing.assert_allclose(coords_rot[:, 2], r, rtol=1e-10)

    def testAzimuthInRange(self):
        rng = np.random.default_rng(4)
        az = rng.uniform(0, 2 * np.pi, 50)
        el = rng.uniform(-np.pi / 2 * 0.9, np.pi / 2 * 0.9, 50)
        coords = np.stack([az, el], axis=1)
        coords_rot = sph.rotateSphPoints(coords, [1.2, 0.7, 0.3], seq="zyx")
        assert np.all(coords_rot[:, 0] >= 0)
        assert np.all(coords_rot[:, 0] < 2 * np.pi)

    def testFrontToLeftUnder90DegYaw(self):
        coords = np.array([[0.0, 0.0]])  # +x axis
        coords_rot = sph.rotateSphPoints(coords, [np.pi / 2, 0, 0], seq="zyx")
        np.testing.assert_allclose(coords_rot[0, 0], np.pi / 2, atol=1e-10)
        np.testing.assert_allclose(coords_rot[0, 1], 0.0, atol=1e-10)

    def testRoundTripRecovery(self):
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

        def toCart(c):
            az_, el_ = c[:, 0], c[:, 1]
            return np.stack(
                [np.cos(el_) * np.cos(az_), np.cos(el_) * np.sin(az_), np.sin(el_)],
                axis=1,
            )

        np.testing.assert_allclose(toCart(coords_back), toCart(coords), atol=1e-10)

    def testIsDegreesFlagConsistency(self):
        coords = np.array([[0.0, 0.0]])
        rot_rad = sph.rotateSphPoints(coords, [np.pi / 4, 0, 0], seq="zyx", isDegrees=False)
        rot_deg = sph.rotateSphPoints(coords, [45.0, 0.0, 0.0], seq="zyx", isDegrees=True)
        np.testing.assert_allclose(rot_rad, rot_deg, atol=1e-12)

    def testOptionalRadiusColumnConsistent(self):
        coords_2col = np.array([[np.pi / 4, np.pi / 6]])
        coords_3col = np.array([[np.pi / 4, np.pi / 6, 1.0]])
        rot_2 = sph.rotateSphPoints(coords_2col, [0.3, 0.0, 0.0], seq="zyx")
        rot_3 = sph.rotateSphPoints(coords_3col, [0.3, 0.0, 0.0], seq="zyx")
        np.testing.assert_allclose(rot_2[:, :2], rot_3[:, :2], atol=1e-12)


class TestDecimateCoefficients:
    def testDefaultRetainsM0Only(self):
        N = 3
        data_nm = np.random.default_rng(6).standard_normal(((N + 1) ** 2, 5))
        data_dec = sph.decimateCoefficients(data_nm)
        assert data_dec.shape[0] == N + 1

    def testCustomRetainACN(self):
        N = 2
        data_nm = np.eye((N + 1) ** 2)[:, :4]
        data_dec = sph.decimateCoefficients(data_nm, retainACN=[0, 1])
        assert data_dec.shape[0] == 2
        np.testing.assert_array_equal(data_dec[0, :], data_nm[0, :])
        np.testing.assert_array_equal(data_dec[1, :], data_nm[1, :])

    def testOutOfBoundsACNRaises(self):
        data_nm = np.ones((9, 4))
        with pytest.raises(ValueError):
            sph.decimateCoefficients(data_nm, retainACN=[999])


class TestOrthogonalityMatrix:
    def testShape(self):
        N = 2
        pos = makeGrid(N=N)
        Ynm = sph.sphHarm(pos, N=N, kind="realsn3d")
        orthog = sph.orthogonalityMatrix(Ynm)
        numCoeffs = (N + 1) ** 2
        assert orthog.shape == (numCoeffs, numCoeffs)

    def testSymmetric(self):
        N = 2
        pos = makeGrid(N=N)
        Ynm = sph.sphHarm(pos, N=N, kind="realsn3d")
        orthog = sph.orthogonalityMatrix(Ynm)
        np.testing.assert_allclose(orthog, orthog.conj().T, atol=1e-12)


class TestReadSofa:
    @pytest.mark.skip(reason="requires SOFA file on disk")
    def testReadSofaShape(self):
        pass
