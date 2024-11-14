from nose.tools import *
import automesh
import numpy as np
import numpy.testing as npt


def setup():
    pass


def teardown():
    pass


def test_crop():
    scan = automesh.Scan()
    scan.values = np.arange(125).reshape((5, 5, 5))
    scan.origin = [0, 3, 1.2]
    scan.spacing = [1, 0.9, 2.2]

    cropped = automesh.crop(scan, [[1, -2], [0, 3], [2, 5]])

    npt.assert_equal(cropped.shape, (2, 3, 3))
    npt.assert_almost_equal(cropped.origin, np.array([1., 3., 5.6]))
    npt.assert_array_equal(cropped.values,
                           np.array([
                               [[27, 28, 29], [32, 33, 34], [37, 38, 39]],
                                [[52, 53, 54], [57, 58, 59], [62, 63, 64]]]))