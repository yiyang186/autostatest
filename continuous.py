import numpy as np 
import base
import utils

def one_way(x, mu0, alpha=0.05, verbose=1):
    x = utils.type_check(x)
    t, v, p = base.t_1sample(x, mu0)

    if verbose == 1:
        utils.compare0(p, alpha)
    return p

def pair(x1, x2, alpha=0.05, verbose=1):
    x1, x2 = utils.type_check(x1), utils.type_check(x2)
    t, v, p = base.t_paired(x1, x2)

    if verbose == 1:
        utils.compare0(p, alpha)
    return t, v, p
