# pytest path_to_test.py
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

import numpy as np
import continuous
import discrete
import discriminate
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

def test_t_1sample():
    """
    "Medical Statistics", 3rd Edition, Page 37, Bottom
    """
    corrected_result = 0.0198089207163
    assert np.isclose(
        base.t_1sample_base(36, 130.83, 25.74, 140)[-1], corrected_result)

def test_t_paired():
    """
    "Medical Statistics", 3rd Edition, Example 3-6
    """
    corrected_result = (7.9259757210549608, 9, 1.1919761845463216e-05)
    x1 = np.array([0.84, 0.591, 0.674, 0.632, 0.687, 0.978, 0.75, 0.73, 1.2, 0.87])
    x2 = np.array([0.58, 0.509, 0.5, 0.316, 0.337, 0.517, 0.454, 0.512, 0.997, 0.506])
    assert np.allclose(continuous.t_paired(x1, x2), corrected_result)

def test_chi2():
    """
    "Medical Statistics", 3rd Edition, Example 7-1
    """
    xtab = np.array([
        [99, 5],
        [75, 21]
    ])
    corrected_result = (12.857069985717196, 1, 0.00033620659688458589)
    assert np.allclose(discrete.chi2_test(xtab), corrected_result)

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
    assert np.allclose(discrete.chi2_trend_test(xtab_ms, verbose=0), corrected_result)

def test_var_homo():
    """
    "Medical Statistics", 3rd Edition, Example 3-11
    """
    cr = (1.5983101734517282, 0.157643797395044)
    assert np.allclose(base.var_homo_test_base(20, 3.0601**2, 2.4205**2), cr)

    """
    "Medical Statistics", 3rd Edition, Example 3-12
    """
    cr = (3.7746938775510217, 0.0028660061648581991)
    assert np.allclose(base.var_homo_test_base(20, 1.36**2, 0.7**2), cr)
    
def test_2way_equalvar():
    # "Medical Statistics", 3rd Edition, Example 3-7
    cr = (-0.64187792644640285, 38, 0.73759706676185277)
    ret = base.t_equal_var(20, 2.065, 3.0601**2, 20, 2.625, 2.4205**2)
    assert np.allclose(ret, cr)

def test_cochrancox():
    # "Medical Statistics", 3rd Edition, Example 3-8
    cr = (0.96484629025709379, (19, 19), 2.0930240544082634)
    ret = base.cochrancox(20, 1.46, 1.36**2, 20, 1.13, 0.7**2, alpha=0.05)
    assert np.isclose(ret[0], cr[0])
    assert np.isclose(ret[-1], cr[-1])

def test_satterthwaite():
    # "Medical Statistics", 3rd Edition, Page 41 Bottom
    cr = (0.96484629025709379, 28.406834655762836, 0.17138220384731195)
    ret = base.satterthwaite(20, 1.46, 1.36**2, 20, 1.13, 0.7**2)
    assert np.allclose(ret, cr)

def test_welch():
    # "Medical Statistics", 3rd Edition, Page 42 Top
    cr = (0.96484629025709379, 29.397027777422082, 0.1712462311449513)
    ret = base.welch(20, 1.46, 1.36**2, 20, 1.13, 0.7**2)
    assert np.allclose(ret, cr)

######################################################################   
# python path_to_test.py

xtab_ms = np.array([
    [70, 22, 4, 2],
    [27, 24, 9, 3],
    [16, 23, 13, 7],
    [9, 20, 15, 14]
])
ret = discrete.chi2_trend_test(xtab_ms, verbose=1)

print("---------------")
xtab_ms = np.array([
    [99, 5],
    [75, 21]
])
print(discrete.chi2_test(xtab_ms))

print("---------------")
x1 = np.array([0.84, 0.591, 0.674, 0.632, 0.687, 0.978, 0.75, 0.73, 1.2, 0.87])
x2 = np.array([0.58, 0.509, 0.5, 0.316, 0.337, 0.517, 0.454, 0.512, 0.997, 0.506])
print(continuous.t_paired(x1, x2))

# print("---------------")
# x = np.array([42, 65, 75, 59, 57, 68, 55, 54, 71, 78])
# discriminate.qqplot(x)

print("---------------")
x1 = np.array([-0.7, -5.6, 2., 2.8, 0.7, 3.5, 4., 5.8, 7.1, -0.5, 2.5, -1.6, 1.7, 3., 0.4, 4.5, 4.6, 2.5, 6., -1.4])
x2 = np.array([3.7, 6.5, 5., 5.2, 0.8, 0.2, 0.6, 3.4, 6.6, -1.1,
    6., 3.8, 2., 1.6, 2., 2.2, 1.2, 3.1, 1.7, -2])
print(discriminate.var_homogeneity(x1, x2, alpha=0.1))

print("---------------")
print(continuous.t_approx(x1, x2, alpha=0.05, method="CochranCox"))

print("---------------")
obs = np.array([[10, 10, 20], [20, 20, 20]])
print(discrete.chi2_test(obs))