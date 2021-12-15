'''
pick data in an array above a lower bound and below an upper bound

mix of numpy.where() and numpy.logical_and()
'''

import numpy as np

data = np.random.randint(low=2, high=10, size=50)
# we want indexes between 3 and 8
lb = 3
up = 8

print('original data: ', data)
arr = np.where(np.logical_and(data >= lb, data <= up))
print('filtered arr: ', arr)


