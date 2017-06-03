# pytest path_to_test.py
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

import numpy as np
from base import *

# def test_CI_population_mean():
#     x = np.array()
#     CI_population_mean()
#     assert 1 == 1