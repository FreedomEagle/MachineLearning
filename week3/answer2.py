import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.special import expit as sigmoid
from scipy import optimize


# Find Indices of Positive and Negative Examples
pos = y == 1
neg = y == 0

# Plot Examples
pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)