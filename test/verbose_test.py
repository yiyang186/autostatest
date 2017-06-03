# python path_to_test.py
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

import numpy as np
import trend
import base
import diff
import utils

xtab_ms = np.array([
    [70, 22, 4, 2],
    [27, 24, 9, 3],
    [16, 23, 13, 7],
    [9, 20, 15, 14]
])
ret = trend.chi2_trend_test(xtab_ms, verbose=1)

print("---------------")
xtab_ms = np.array([
    [99, 5],
    [75, 21]
])
print(diff.chi2(xtab_ms))

print("---------------")
x1 = np.array([0.84, 0.591, 0.674, 0.632, 0.687, 0.978, 0.75, 0.73, 1.2, 0.87])
x2 = np.array([0.58, 0.509, 0.5, 0.316, 0.337, 0.517, 0.454, 0.512, 0.997, 0.506])
print(diff.t_paired(x1, x2))

# print("---------------")
# x = np.array([42, 65, 75, 59, 57, 68, 55, 54, 71, 78])
#base.qqplot(x)

print("---------------")
x1 = np.array([-0.7, -5.6, 2., 2.8, 0.7, 3.5, 4., 5.8, 7.1, -0.5,
    2.5, -1.6, 1.7, 3., 0.4, 4.5, 4.6, 2.5, 6., -1.4])
x2 = np.array([3.7, 6.5, 5., 5.2, 0.8, 0.2, 0.6, 3.4, 6.6, -1.1,
    6., 3.8, 2., 1.6, 2., 2.2, 1.2, 3.1, 1.7, -2])
print(diff.homogeneity_of_variance_test(x1, x2, alpha=0.1))
