'''
traj_analysis.py
'''
# Numerical tools
import numpy as np 

# Dataframes
import pandas as pd 

# I/O
import os
import sys
import tifffile

# pyspaz functions
from pyspaz import spazio
from pyspaz import mask

def compile_displacements(
    trajs,
    n_gaps = 0,
    time_delays = 4,
):
    pass