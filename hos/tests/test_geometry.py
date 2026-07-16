import numpy as np
import pytest
from hos.toolboxes import geometry


class TestCart2SphSph2Cart:
    def testFrontAxisKnownValue(self):
        s = geometry.cart2sph(np.array([1.0]), np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(s[0, 0], 0.0, atol=1e-12)   # az
        np.testing.assert_allclose(s[0, 1], 0.0, atol=1e-12)   # el
        np.testing.assert_allclose(s[0, 2], 1.0, atol=1e-12)   # r

    def testTopAxisKnownValue(self):
        s = geometry.cart2sph(np.array([0.0]), np.array([0.0]), np.array([1.0]))
        np.testing.assert_allclose(s[0, 1], np.pi / 2, atol=1e-12)
        np.testing.assert_allclose(s[0, 2], 1.0, atol=1e-12)

    def testLeftAxisKnownValue(self):
        s = geometry.cart2sph(np.array([0.0]), np.array([1.0]), np.array([0.0]))
        np.testing.assert_allclose(s[0, 0], np.pi / 2, atol=1e-12)
        np.testing.assert_allclose(s[0, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(s[0, 2], 1.0, atol=1e-12)

    def testRoundTrip(self):
        rng = np.random.default_rng(10)
        xyz = rng.standard_normal((20, 3))
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
        sph_vals = geometry.cart2sph(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        cart_rec = geometry.sph2cart(sph_vals)
        np.testing.assert_allclose(cart_rec, xyz, atol=1e-12)

    def testRadiusScaling(self):
        r = 3.5
        s = geometry.cart2sph(np.array([r]), np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(s[0, 2], r, atol=1e-12)


class TestApplyRotation:
    def testIdentityRotationReturnsSame(self):
        xyz = np.array([1.0, 0.5, 0.3])
        xyz_rot = geometry.applyRotation(xyz, [0, 0, 0])
        np.testing.assert_allclose(xyz_rot.ravel(), xyz, atol=1e-12)

    def testInverseRotationRoundTrip(self):
        xyz = np.array([[1.0, 0.0, 0.0]])
        ypr = [np.pi / 4, np.pi / 6, np.pi / 8]
        xyz_fwd = geometry.applyRotation(xyz, ypr)
        xyz_back = geometry.applyRotation(xyz_fwd, ypr, isInverseRotation=True)
        np.testing.assert_allclose(xyz_back.ravel(), xyz.ravel(), atol=1e-12)

    def testPreservesNorm(self):
        rng = np.random.default_rng(11)
        xyz = rng.standard_normal((10, 3))
        ypr = [0.3, 0.5, 0.1]
        xyz_rot = geometry.applyRotation(xyz, ypr)
        np.testing.assert_allclose(
            np.linalg.norm(xyz_rot, axis=1),
            np.linalg.norm(xyz, axis=1),
            atol=1e-12,
        )

    def testIsDegreesFlagConsistency(self):
        xyz = np.array([1.0, 0.0, 0.0])
        rot_rad = geometry.applyRotation(xyz, [np.pi / 4, 0, 0], isDegrees=False)
        rot_deg = geometry.applyRotation(xyz, [45.0, 0.0, 0.0], isDegrees=True)
        np.testing.assert_allclose(rot_rad.ravel(), rot_deg.ravel(), atol=1e-12)

    def test90DegYawMapsXAxisToY(self):
        xyz = np.array([1.0, 0.0, 0.0])
        xyz_rot = geometry.applyRotation(xyz, [90.0, 0.0, 0.0], isDegrees=True)
        np.testing.assert_allclose(xyz_rot.ravel(), [0.0, 1.0, 0.0], atol=1e-12)


class TestEstimateAndApplyRotation:
    def testHhatFrontIsIdentity(self):
        xyz = np.array([0.5, 0.3, 0.7])
        hhat = np.array([1.0, 0.0, 0.0])
        xyz_rot = geometry.estimateAndApplyRotation(xyz, hhat)
        np.testing.assert_allclose(xyz_rot.ravel(), xyz, atol=1e-10)

    def testHhatMapsToXAxis(self):
        hhat = np.array([0.0, 1.0, 0.0])
        xyz_rot = geometry.estimateAndApplyRotation(hhat[np.newaxis, :], hhat)
        np.testing.assert_allclose(xyz_rot.ravel(), [1.0, 0.0, 0.0], atol=1e-10)

    def testBadXYZRaises(self):
        with pytest.raises(ValueError):
            geometry.estimateAndApplyRotation(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))


class TestCalcRotationMatrix:
    def testIdentityRotation(self):
        R = geometry.calcRotationMatrix(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def testOrthogonal(self):
        R = geometry.calcRotationMatrix(np.array([0.5, 0.3, 0.1]))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def testDeterminantOne(self):
        R = geometry.calcRotationMatrix(np.array([1.2, 0.4, 0.7]))
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def testBadShapeRaises(self):
        with pytest.raises(ValueError):
            geometry.calcRotationMatrix(np.array([0.0, 0.0]))


class TestCalcRotationMatrixYawOnly:
    def testIdentityYaw(self):
        R = geometry.calcRotationMatrixYawOnly(0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def testOrthogonal(self):
        R = geometry.calcRotationMatrixYawOnly(np.pi / 5)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def testDeterminantOne(self):
        R = geometry.calcRotationMatrixYawOnly(np.pi / 3)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def testZAxisUnchanged(self):
        R = geometry.calcRotationMatrixYawOnly(np.pi / 4)
        np.testing.assert_allclose(R @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-12)

    def test90DegYawMapsXToY(self):
        R = geometry.calcRotationMatrixYawOnly(np.pi / 2)
        result = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-12)
