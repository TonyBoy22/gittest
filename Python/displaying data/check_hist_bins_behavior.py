'''
Script to evaluate how bins are used to classify information and check how to process it afterward
'''

import numpy as np
import matplotlib.pyplot as plt

data = np.random.randint(1, 3, 15, int)
bins = np.linspace(0, 4, 5)

# Classify data in bins
test_hist, ax = plt.subplots()
ax.hist(data, bins)
plt.tight_layout()
plt.show()

'''
Conclusion: the inclusivity of bins goes by
lb <= value < ub

so if we want to classify 1 on integer bins, it goes in the 
1 to 2 bin instead 0 to 1
'''