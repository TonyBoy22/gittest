'''
use of exec in different cases
'''
import matplotlib.pyplot as plt
import numpy as np

# create figures with different names with exec on fig variables
for i in range(5):
    figure = 'figure'
    exec("figure = 'figure' + str(i)")
    print(figure)