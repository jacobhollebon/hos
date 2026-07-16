import numpy as np
import pytest
from hos.toolboxes import calculator


class TestCalculateHOSAngle:
    def testFrontSourceIsZero(self):
        angle = calculator.calculateHOSAngle(np.array([[1.0, 0.0, 0.0]]))
        np.testing.assert_allclose(angle, [0.0], atol=1e-10)

    def testLeftSourceIsPositiveHalfPi(self):
        angle = calculator.calculateHOSAngle(np.array([[0.0, 1.0, 0.0]]))
        np.testing.assert_allclose(angle, [np.pi / 2], atol=1e-10)

    def testRightSourceIsNegativeHalfPi(self):
        angle = calculator.calculateHOSAngle(np.array([[0.0, -1.0, 0.0]]))
        np.testing.assert_allclose(angle, [-np.pi / 2], atol=1e-10)

    def testHhatNormalisationIgnored(self):
        xyz = np.array([[1.0, 0.0, 0.0]])
        angle_unit = calculator.calculateHOSAngle(xyz, hhat=np.array([1.0, 0.0, 0.0]))
        angle_scaled = calculator.calculateHOSAngle(xyz, hhat=np.array([5.0, 0.0, 0.0]))
        np.testing.assert_allclose(angle_unit, angle_scaled, atol=1e-10)

    def testMultipleSources(self):
        xyz = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        angles = calculator.calculateHOSAngle(xyz)
        assert angles.shape == (3,)
        np.testing.assert_allclose(angles, [0.0, np.pi / 2, -np.pi / 2], atol=1e-10)

    def testBadXYZShapeRaises(self):
        with pytest.raises(ValueError):
            calculator.calculateHOSAngle(np.array([[1.0, 0.0]]))

    def testRotatedHhatChangesAngles(self):
        # Listener looking in +y direction. Source at +x is now to their right → -pi/2.
        hhat = np.array([0.0, 1.0, 0.0])
        angle = calculator.calculateHOSAngle(np.array([[1.0, 0.0, 0.0]]), hhat=hhat)
        np.testing.assert_allclose(angle, [-np.pi / 2], atol=1e-10)

    def testBadHhatShapeRaises(self):
        with pytest.raises(ValueError):
            calculator.calculateHOSAngle(
                np.array([[1.0, 0.0, 0.0]]), hhat=np.array([1.0, 0.0])
            )


class TestCalculateHOSPlant:
    def testShape(self):
        angles = np.deg2rad(np.linspace(-60, 60, 7))
        plant = calculator.calculateHOSPlant(angles, order=4, HOSType="sine")
        assert plant.shape == (5, 7)

    def testOrderZeroAllOnes(self):
        angles = np.deg2rad(np.array([-60, 0, 60]))
        plant = calculator.calculateHOSPlant(angles, order=0, HOSType="sine")
        np.testing.assert_array_equal(plant, np.ones((1, 3)))

    def testSineAngleZeroFirstRowOnly(self):
        plant = calculator.calculateHOSPlant(np.array([0.0]), order=3, HOSType="sine")
        np.testing.assert_allclose(plant[0, 0], 1.0, atol=1e-12)   # sin(0)^0 = 1
        np.testing.assert_allclose(plant[1:, 0], 0.0, atol=1e-12)  # sin(0)^n = 0 for n>0

    def testCosineAngleZeroAllOnes(self):
        plant = calculator.calculateHOSPlant(np.array([0.0]), order=3, HOSType="cosine")
        np.testing.assert_allclose(plant[:, 0], 1.0, atol=1e-12)

    def testInvalidTypeRaises(self):
        with pytest.raises(ValueError):
            calculator.calculateHOSPlant(np.array([0.0]), order=1, HOSType="tangent")

    def testNegativeOrderRaises(self):
        with pytest.raises(ValueError):
            calculator.calculateHOSPlant(np.array([0.0]), order=-1)


class TestCalculateHOSDecoder:
    def testShape(self):
        angles = np.deg2rad(np.linspace(-60, 60, 5))
        order = 4
        plant = calculator.calculateHOSPlant(angles, order=order, HOSType="sine")
        decoder = calculator.calculateHOSDecoder(plant, order=order)
        assert decoder.shape == (5, 5)

    def testPlantTimesDecoderApproxIdentity(self):
        angles = np.deg2rad(np.linspace(-70, 70, 5))
        order = 4
        plant = calculator.calculateHOSPlant(angles, order=order, HOSType="sine")
        decoder = calculator.calculateHOSDecoder(plant, order=order)
        np.testing.assert_allclose(plant @ decoder, np.eye(order + 1), atol=1e-8)

    def testRegularisationDoesNotCrash(self):
        angles = np.deg2rad(np.linspace(-60, 60, 5))
        order = 4
        plant = calculator.calculateHOSPlant(angles, order=order, HOSType="sine")
        decoder = calculator.calculateHOSDecoder(plant, order=order, beta=1e-6)
        assert decoder.shape == (5, 5)

    def testMismatchedOrderRaises(self):
        angles = np.deg2rad(np.linspace(-60, 60, 5))
        plant = calculator.calculateHOSPlant(angles, order=4, HOSType="sine")
        with pytest.raises(ValueError):
            calculator.calculateHOSDecoder(plant, order=3)
