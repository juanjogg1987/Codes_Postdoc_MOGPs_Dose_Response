import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from fitter import Fitter

random_data = stats.gamma.rvs(2, loc=2.5, scale=2, size=5000)

dist_fitter = Fitter(random_data,
                   distributions = ["cauchy",
                                    "rayleigh",
                                    "beta",
                                    "gamma",
                                    "lognorm",
                                    "skewnorm"])
dist_fitter.fit()
dist_fitter.summary()