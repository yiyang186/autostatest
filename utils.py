import numpy as np 
from scipy.stats import t
from scipy.stats import norm

def compare0(_p, alpha):
    print("p = {0}, alpha = {1}".format(_p, alpha))
    if _p >= alpha:
        print("p >= alpha, 差异无统计学意义")
    else:
        print("p < alpha, 差异有统计学意义")

def compare1(_p, alpha):
    print("p = {0}, alpha = {1}".format(_p, alpha))
    if _p >= alpha:
        print("p >= alpha, 变量间差异无统计学意义")
    elif np.abs(_p - alpha) < alpha / 10.0:
        print("p=alpha, 请使用Fisher确切概率法")
    else:
        print("p < alpha, 变量间差异有统计学意义")

def type_check(a):
    if type(a) == np.ndarray:
        return a 
    
    try:
        a = np.array(a)
    except:
        print("ERROR: 请输入numpy数组或列表!!!")
        exit()
    return a

def sample_std(x):
    return np.sqrt(((x - x.mean())**2).sum() / (x.size-1))

def CI_population_mean(x, ci=0.95, sigma=None):
    n = x.size
    m = x.mean()
    s = x.std()
    return CI_population_mean_base(n, m, s, ci=ci, sigma=sigma_SE)

# def CI_population_mean_diff(x1, x2, )

def CI_population_mean_base(n, m, s, ci=0.95, sigma=None):
    alpha = 1 - ci
    s_SE = s /  np.sqrt(n)

    if sigma or n > 60:
        _u = norm.isf(alpha/2)
        if sigma:
            sigma_SE = sigma / np.sqrt(n)
            return (m - _u * sigma_SE, m + _u * sigma_SE)
        else:
            return (m - _u * s_SE, m + _u * s_SE)
    else:
        _t = t.isf(alpha/2, n-1)
        return (m - _t * s_SE, m + _t * s_SE)

