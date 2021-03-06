import numpy as np 
from scipy.stats import chi2
from scipy.stats import t
from scipy.stats import norm
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import f
import utils

def t_1sample_base(n, m, s, mu0):
    s_SE = s /  np.sqrt(n)
    _t = np.abs(m - mu0) / s_SE
    _v = n - 1
    _p = t.sf(_t, _v)
    return _t, _v, _p

def t_1sample(x, mu0):
    n = x.size
    m = x.mean()
    s = x.std()
    return t_1sample_base(n, m, s)

def chi2_rxc(xtab, verbose=1):
    if xtab.shape[0] < 2 or xtab.shape[1] < 2:
        raise IOError("ERROR: 输入格式错误！！！")

    n = xtab.sum()
    nc = xtab.sum(axis=0)
    nr = xtab.sum(axis=1)
    T = ((np.ones_like(xtab) * nc).T * nr).T / n
    _chi2 = ((xtab - T) ** 2 / T).sum()
    _df = (xtab.shape[0] - 1) * (xtab.shape[1] - 1)

    if n < 40 or (T < 1).sum() > 0:
        raise ValueError("ERROR: 样本过少或理论频数过小，请调整数据或者使用Fisher确切概率法！！！")

    if xtab.shape == (2, 2):
        if (T < 5).sum() > 0:
            _chi2 = ((np.abs(xtab - T) - n / 2) ** 2 / T).sum()
            if verbose == 1:
                print("已自动改用卡方校正公式，您也可自行使用Fisher确切概率法")
    elif (T < 5).sum() > T.size / 5.0:
        raise ValueError("ERROR: 理论频数过小，请调整数据或者使用Fisher确切概率法！！！")

    _p = chi2.sf(_chi2, _df)
    return _chi2, _df, _p

def t_paired(x1, x2):
    if x1.size != x2.size:
        raise IOError("ERROR: 两组样本量不等！！！")

    d = x1 - x2
    return t_paired_base(d)

def t_paired_base(d):
    n = d.size
    d_sum = d.sum() 
    d2_sum = (d ** 2).sum()
    d_bar = d_sum / n
    s_d = np.sqrt((d2_sum - d_sum**2/n) / (n-1))
    s_SE = s_d / np.sqrt(n)
    _t = d_bar / s_SE
    _v = n - 1
    _p = t.sf(_t, _v)
    return _t, _v, _p

def t_equal_var(n1, m1, var1, n2, m2, var2):
    temp = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2) * (1/n1 + 1/n2)
    _t = (m1-m2) / np.sqrt(temp)
    _v = n1 + n2 - 2
    _p = t.sf(_t, _v)
    return _t, _v, _p

def cochrancox(n1, m1, var1, n2, m2, var2, alpha=0.05):
    v1, v2 = n1-1, n2-1
    _t = (m1-m2) / np.sqrt(var1/n1 + var2/n2)
    _t_av1 = t.isf(alpha/2, v1)
    _t_av2 = t.isf(alpha/2, v2)
    var1_SE, var2_SE = var1/n1, var2/n2
    _t_a = (var1_SE*_t_av1 + var2_SE*_t_av2)/(var1_SE+var2_SE)
    return _t, (v1, v2), _t_a

def satterthwaite(n1, m1, var1, n2, m2, var2, alpha=0.05):
    _t = (m1-m2) / np.sqrt(var1/n1 + var2/n2)
    var1_SE, var2_SE = var1/n1, var2/n2
    
    _v = (var1_SE + var2_SE)**2 / (var1_SE**2/(n1-1) + var2_SE**2/(n2-1))
    _p = t.sf(_t, _v)
    return _t, _v, _p

def welch(n1, m1, var1, n2, m2, var2, alpha=0.05):
    _t = (m1-m2) / np.sqrt(var1/n1 + var2/n2)
    var1_SE, var2_SE = var1/n1, var2/n2

    _v = (var1_SE + var2_SE)**2 / (var1_SE**2/(n1+1) + var2_SE**2/(n2+1)) - 2
    _p = t.sf(_t, _v)
    return _t, _v, _p

def norm_moment(x):
    n = x.size
    g1 = skew(x)
    g2 = kurtosis(x)
    sigma_g1 = np.sqrt(6*n/(n-2)*(n-1)/(n+1)/(n+3))
    sigma_g2 = np.sqrt(24*n/(n-3)*(n-1)/(n-2)*(n-1)/(n+3)/(n+5))
    return norm.sf(g1/sigma_g1), norm.sf(g2/sigma_g2)

def var_homo_test(x1, x2):
    return var_homo_test_base(x1.size, x1.var(), x2.var())

def var_homo_test_base(n, var1, var2):
    if var1 < var2:
        var1, var2 = var2, var1
    _F = var1 / var2
    v1, v2 = n-1, n-1
    _p = f.sf(_F, v1, v2)
    return _F, _p



