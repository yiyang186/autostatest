import numpy as np 
import base
import utils
import discriminate as dcmn

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

def t_2sample(x1, x2, alpha=0.05, verbose=1):
    x1, x2 = utils.type_check(x1), utils.type_check(x2)
    if not dcmn.isNormal(x1) or not dcmn.isNormal(x2):
        print("样本不服从正态分布，建议做Boxcox变换或使用非参数检验！！！")
        exit()

    if x1.size > 60 and x2.size > 60:
        n1, m1, var1 = x1.size, x1.mean(), x1.var()
        n2, m2, var2 = x2.size, x2.mean(), x2.var()
        t, v, p = base.t_equal_var(x1, x2)
    elif not dcmn.isVarHomo(x1, x2):
        print("样本量小且无方差齐性，建议使用近似t检验或非参数检验！！！")
        exit()
    return

def t_approx(x1, x2, alpha=0.05, method="CochranCox", verbose=1):
    methods = ["CochranCox", "Satterthwaite", "Welch"]
    utils.method_check(method, methods)

    x1, x2 = utils.type_check(x1), utils.type_check(x2)
    if not dcmn.isNormal(x1) or not dcmn.isNormal(x2):
        print("样本不服从正态分布，建议做Boxcox变换或使用非参数检验！！！")
        exit()
    
    func = eval('base.' + method.lower())
    n1, m1, var1 = x1.size, x1.mean(), x1.var()
    n2, m2, var2 = x2.size, x2.mean(), x2.var()
    t, v, p = func(n1, m1, var1, n2, m2, var2, alpha=alpha)

    if verbose == 1:
        utils.compare0(p, alpha)
    return t, v, p
