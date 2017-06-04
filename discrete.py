import numpy as np 
import pandas as pd
from scipy.stats import chi2
import scipy.stats as stats
import utils
import base

def chi2_trend_test(xtab, alpha=0.05, verbose=1):
    """
    xtabl: numpy array
    alpha: 
    """

    chi2_total, df_total, p_total = chi2_test(xtab, alpha=alpha, verbose=verbose)
    if p_total >= alpha:
        return None

    n = xtab.sum()
    nc=xtab.sum(axis=0)
    nr=xtab.sum(axis=1)
    x = np.arange(xtab.shape[0]) + 1
    y = np.arange(xtab.shape[1]) + 1
    fx = (x * nr).sum()
    fx2 = (x ** 2 * nr).sum()
    fy = (y * nc).sum()
    fy2 = (y ** 2 * nc).sum()
    fxy = ((xtab * y).T * x).sum()
    lxx = fx2 - fx ** 2 / n
    lyy = fy2 - fy ** 2 / n
    lxy = fxy - fx * fy / n
    b = lxy / lxx
    sb2 = lyy / lxx / n
    chi2_regression = b ** 2 / sb2
    df_regression = 1
    p_regression = chi2.sf(chi2_regression, df_regression)

    chi2_deviation = chi2_total - chi2_regression
    df_deviation = df_total - df_regression
    p_deviation = chi2.sf(chi2_deviation, df_deviation)

    ret = np.array([
        [chi2_total, df_total, p_total],
        [chi2_regression, df_regression, p_regression],
        [chi2_deviation, df_deviation, p_deviation]
    ])

    if verbose == 1:
        index = ['总变异', '线性回归分量', '偏离线性回归分量']
        columns = ['卡方', '自由度', 'P']
        _df = pd.DataFrame(data=ret, index=index, columns=columns)
        print(_df)
        print("alpha=", alpha)
        if ret[1, 2] < alpha:
            print("线性回归分量有统计学意义,可以认为变量间存在相关关系")
            if ret[2, 2] < alpha:
                print("偏离线性回归分量有统计学意义,变量间关系不是线性关系")
            else:
                print("偏离线性回归分量无统计学意义,变量间为线性关系")
        else:
            print("线性回归分量无统计学意义,不可认为变量间存在相关关系")
    return ret

def chi2_test(xtab, alpha=0.05, verbose=1):
    xtab = utils.type_check(xtab)
    _chi2, _df, _p = base.chi2_rxc(xtab)
    
    if verbose == 1:
        utils.compare1(_p, alpha)
            
    return _chi2, _df, _p

def fisher_exact_prob(xtab, alpha=0.05):
    _o, _p = stats.fisher_exact(xtab)

    if verbose == 1:
        utils.compare1(_p, alpha)

    return _o, _p