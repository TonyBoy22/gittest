'''
Script to use minimize to solve most common
optimization problems

LP
QCQP


result object:
result.fun: result of the function to optimize at the value of result.x
result.x: minimal value found by optimization
result.success: Boolean to indicate that optimization process went well
'''

from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

'''
First problem is from
stochastic model predictive control for lane change decision of automated Driving Vehicles

J = [l_1(x - s_ttc,k)**2 + l_2*np.absolute(x - s_dist,k) + l_3*(x - s_CP,k)**2]
s_min < s < s_max

for k = 1,...,Np

help from this tutorial
https://www.youtube.com/watch?v=G0yP_TM-oag
'''
def fun(x):
    # constants
    J = 2*(x - 2.0)**2 + 1.5*np.absolute(x - 7.0) + 3*(x - 3.8)**2
    return J

# Starting guess
x_0 = np.array([5.0])
x_0_test = [[5.0]]
print('length initial guess: ', len(x_0), len(x_0_test))

# bounds. Can be tuple for memory size
# As a tip about bounds lenght error:
# https://stackoverflow.com/questions/29150064/python-scipy-indexerror-the-length-of-bounds-is-not-compatible-with-that-of-x0
bnds = ((0.0, 100.0),)

# optimization
'''
To give a variable, we need to define a function and we give the
callable to scipy.optimize.minimize()
It avoids defining symbolic or other ways to declare
arbitrary variables
'''
# First, let's plot the function to get an intuition
x = np.linspace(0, 10.0, 50)
# fig1, ax1 = plt.subplots()
# ax1.plot(x, fun(x))
# plt.show()

result = minimize(fun, x_0, method='SLSQP', bounds=bnds)

if result.success:
    print('result: ', result.x)
else:
    print('optimization failed. result = ', result.fun)

###################################################################
# Case 2: Several arguments in the function to minimize

# Other Arguments. First, outside of a tuple, then inside
a = 5
b = 8
c = -4

tpl = ((a, b, c),)
def fun_with_args(x, tpl):
    '''
    Let's try with same cost function
    J = [l_1(x - s_ttc,k)**2 + l_2*np.absolute(x - s_dist,k) + l_3*(x - s_CP,k)**2]
    '''

    a, b, c = tpl
    d = a+b+c
    J = 2 * (x - 2.0) ** 2 + d * np.absolute(x - 7.0) + 3 * (x - 3.8) ** 2
    return J


def make_constraints(args_in_tuple):
    constraints = {
        'args': args_in_tuple
    }
    return constraints


cstr = make_constraints(tpl)
res = minimize(fun_with_args, x0=np.array([5]), \
               method='SLSQP', args=tpl, bounds=bnds)
if res.success:
    print('result: ', res.x)
else:
    print('opt. failed')