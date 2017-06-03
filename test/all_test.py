# pytest path_to_test.py
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

import numpy as np
import trend
import diff
import utils
import base

# xtab = np.array([
#     [1021, 805, 1476, 835, 48],
#     [203, 186, 397, 333, 37],
#     [55, 58, 129, 132, 32],
#     [16, 20, 43, 54, 27]
# ])

def test_CI():
    """
    "Medical Statistics", 3rd Edition, Example 3-2
    """
    corrected_result = (164.34610086233263, 169.55389913766734)
    assert np.allclose(
        utils.CI_population_mean_base(10, 166.95, 3.64, ci=0.95), 
        corrected_result)

    """
    "Medical Statistics", 3rd Edition, Example 3-3
    """
    corrected_result = (3.4736915410780389, 3.8063084589219613)
    assert np.allclose(
        utils.CI_population_mean_base(200, 3.64, 1.20, ci=0.95), 
        corrected_result)

def test_base_t():
    """
    "Medical Statistics", 3rd Edition, Page 37, Bottom
    """
    corrected_result = 0.0198089207163
    assert np.isclose(
        base.t_1sample_base(36, 130.83, 25.74, 140)[-1], corrected_result)

    """
    "Medical Statistics", 3rd Edition, Example 3-6
    """
    corrected_result = (7.9259757210549608, 9, 1.1919761845463216e-05)
    x1 = np.array([0.84, 0.591, 0.674, 0.632, 0.687, 0.978, 0.75, 0.73, 1.2, 0.87])
    x2 = np.array([0.58, 0.509, 0.5, 0.316, 0.337, 0.517, 0.454, 0.512, 0.997, 0.506])
    assert np.allclose(diff.t_paired(x1, x2), corrected_result)

def test_diff_chi2():
    """
    "Medical Statistics", 3rd Edition, Example 7-1
    """
    xtab = np.array([
        [99, 5],
        [75, 21]
    ])
    corrected_result = (12.857069985717196, 1, 0.00033620659688458589)
    assert np.allclose(diff.chi2(xtab), corrected_result)

def test_trend_chi2():
    """
    "Medical Statistics", 3rd Edition, Example 7-11
    """
    xtab_ms = np.array([
        [70, 22, 4, 2],
        [27, 24, 9, 3],
        [16, 23, 13, 7],
        [9, 20, 15, 14]
    ])
    corrected_result = np.array([[  7.14324940e+01,   9.00000000e+00,   7.96960419e-12],
       [  6.36183196e+01,   1.00000000e+00,   1.51018215e-15],
       [  7.81417435e+00,   8.00000000e+00,   4.51829625e-01]])
    assert np.allclose(trend.chi2_trend_test(xtab_ms, verbose=0), corrected_result)
