'''
Evaluation of how the scipy optimize curve fit works
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generate random data

data = np.random.normal(2.0, 1.0, size=1000)
bins = np.linspace(0, 4, 20)

# Use curve fit to set a curve on the hist




# Plot to visualize on histogram

fig, ax = plt.subplots()
ax.hist(data, bins)

plt.show()