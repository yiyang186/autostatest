import numpy as np 
import base
import utils
from scipy.stats import norm
import matplotlib.pyplot as plt

def normality_test(x, alpha=0.1, verbose=1):
    x = utils.type_check(x)
    p_skew, p_kurtosis = base.norm_moment(x)

    if verbose == 1:
        print("使用矩法（动差法）检验正态性：")
        print("p_skew = {0}, p_kurtosis = {1}, alpha = {2}".format(
            p_skew, p_kurtosis, alpha))
        if p_skew < alpha and p_kurtosis < alpha:
            print("p_skew < alpha, p_kurtosis < alpha, 这些样本不服从正态分布")
        else:
            print("尚不能内定这些样本不服从正态分布")
    return p_skew, p_kurtosis

def homogeneity_of_variance_test(x1, x2, alpha=0.1, verbose=1):
    x1 = utils.type_check(x1)
    x2 = utils.type_check(x2)
    _F, _p = base.variance_homo_test(x1, x2)

    if verbose == 1:
        print("使用F-test检验方差齐性：")
        print("p = {0}, alpha = {1}".format(_p, alpha))
        if _p < alpha:
            print("p < alpha, 差异有统计学意义，两组总体方差不等")
        else:
            print("p >= alpha, 差异无统计学意义，尚不能认定两组总体方差不等")
    return _F, _p

def qqplot(x):
    x = utils.type_check(x)
    sigma = utils.sample_std(x)
    x.sort()
    qi = (x - x.mean()) / sigma

    i = np.arange(x.size) + 1
    ti = (i - 0.5) / x.size
    pi = -norm.isf(ti)

    line = [-4, 4]
    plt.figure()
    plt.scatter(pi, qi, s=25, marker='o')
    plt.plot(line, line, lw=0.5, c='black')
    plt.xlabel("t Quantiles")
    plt.ylabel("Studentized Residuals")
    plt.title("Q-Q Plot")
    plt.show()