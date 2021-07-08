'''
Ways to measure and compare techniques for time efficiency

tutorial sources:
https://www.hackerearth.com/blog/developers/faster-python-code/

'''

import timeit
import time

# Comparer le temps de plusieurs commandes produisant le même résultat
# Range
# %timeit.timeit()

# Map
# %timeit.timeit()

# Avoir les timestamp dans une boucle
t = time.time()
print('timestamp from script beginning: ', t)
for i in range(50):
    t = time.time()
    time.sleep(1.0)
    t = time.time() - t
    print(f'timestamp at {i}th iteration: ', t)
'''
if t = time.time() - t at each iteration, the odd will be huge, and the even will be the value we want.
We need to substract a new variable each time in the same loop to avoid the value swings
'''

