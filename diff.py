import numpy as np 
import base
import utils

def chi2(xtab, alpha=0.05, verbose=1):
    xtab = utils.type_check(xtab)
    _chi2, _df, _p = base.chi2_rxc(xtab)
    
    if verbose == 1:
        utils.compare1(_p, alpha)
            
    return _chi2, _df, _p

def t_1sample(x, mu0, alpha=0.05, verbose=1):
    x = utils.type_check(x)
    t, v, p = base.t_1sample(x, mu0)

    if verbose == 1:
        utils.compare0(p, alpha)
    return p

def t_paired(x1, x2, alpha=0.05, verbose=1):
    x1, x2 = utils.type_check(x1), utils.type_check(x2)
    t, v, p = base.t_paired(x1, x2)

    if verbose == 1:
        utils.compare0(p, alpha)
    return t, v, p

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
