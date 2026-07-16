import numpy as np
import pytest
from hos.toolboxes import circular


def makeAzGrid(Q=40):
    return np.linspace(0, 2 * np.pi, Q, endpoint=False)


class TestCircHarm:
    def testShape(self):
        az = makeAzGrid(20)
        Y = circular.circHarm(az, N=3, kind="real")
        assert Y.shape == (20, 7)

    def testShapeExp(self):
        az = makeAzGrid(20)
        Y = circular.circHarm(az, N=3, kind="exp")
        assert Y.shape == (20, 7)

    def testInvalidKindRaises(self):
        with pytest.raises(ValueError):
            circular.circHarm(makeAzGrid(), N=2, kind="badkind")

    def testRealDtype(self):
        Y = circular.circHarm(makeAzGrid(), N=2, kind="real")
        assert not np.iscomplexobj(Y)

    def testExpDtype(self):
        Y = circular.circHarm(makeAzGrid(), N=2, kind="exp")
        assert np.iscomplexobj(Y)

    def testNonUniformAzimuthAccepted(self):
        az = np.array([0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2])
        Y = circular.circHarm(az, N=2, kind="real")
        assert Y.shape == (5, 5)


class TestICHTCHTRoundTrip:
    def testRealRoundTrip(self):
        N = 3
        numCoeffs = 2 * N + 1
        T = 8
        az = makeAzGrid(Q=60)
        rng = np.random.default_rng(20)
        data_nm_truth = rng.standard_normal((numCoeffs, T))
        _, data = circular.CHT(data_nm_truth, az, N, kind="real")
        _, _, data_nm_rec = circular.iCHT(data, az, N, kind="real")
        np.testing.assert_allclose(data_nm_rec, data_nm_truth, atol=1e-8)

    def testExpRoundTrip(self):
        N = 2
        numCoeffs = 2 * N + 1
        T = 6
        az = makeAzGrid(Q=40)
        rng = np.random.default_rng(21)
        data_nm_truth = (rng.standard_normal((numCoeffs, T))
                         + 1j * rng.standard_normal((numCoeffs, T)))
        _, data = circular.CHT(data_nm_truth, az, N, kind="exp")
        _, _, data_nm_rec = circular.iCHT(data, az, N, kind="exp")
        np.testing.assert_allclose(data_nm_rec, data_nm_truth, atol=1e-8)

    def testRealRoundTripMultiChannel(self):
        N = 2
        numCoeffs = 2 * N + 1
        Q = 30
        T = 5
        chs = 2
        az = makeAzGrid(Q=Q)
        rng = np.random.default_rng(22)
        data_nm_truth = rng.standard_normal((numCoeffs, chs, T))
        _, data = circular.CHT(data_nm_truth, az, N, kind="real")
        _, _, data_nm_rec = circular.iCHT(data, az, N, kind="real")
        np.testing.assert_allclose(data_nm_rec, data_nm_truth, atol=1e-8)


class TestICHTDimensionValidation:
    def testOneDimRaises(self):
        az = makeAzGrid()
        with pytest.raises(ValueError):
            circular.iCHT(np.ones(len(az)), az, N=2)

    def testFourDimRaises(self):
        az = makeAzGrid()
        with pytest.raises(ValueError):
            circular.iCHT(np.ones((len(az), 2, 3, 4)), az, N=2)


class TestCHTDimensionValidation:
    def testOneDimRaises(self):
        az = makeAzGrid()
        with pytest.raises(ValueError):
            circular.CHT(np.ones(5), az, N=2)

    def testFourDimRaises(self):
        az = makeAzGrid()
        with pytest.raises(ValueError):
            circular.CHT(np.ones((5, 2, 3, 4)), az, N=2)
