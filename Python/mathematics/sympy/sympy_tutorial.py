'''
Script to evaluate symbolic computation with 2D matrices
'''

import sympy as sp
from sympy.abc import phi, rho
from sympy.matrices import Matrix

pi = sp.pi
print('type and value: ', type(pi), pi)
# Does not return a string or a common data type. all Symbols \
    # from sympy are classes that are specific to sympy
    
# a way to initialize a python session in the console where \
    # sympy imports everything with *
# sp.init_session()

###### Rationals, sqrt, etc
coef_1 = sp.Rational(3,8) # Gives the fraction of argument 1 over argument 2
num_value_coef_1 = sp.N(coef_1) # Gives the numerical value
also_num_val = coef_1.evalf()

# print(num_value_coef_1)
# print(also_num_val)
# Common nomenclature for sympy expressions: expr

################ Now, for larger computations, better to work with symbols ###################
x, n, y = sp.symbols('x n y')
border = x + (n + 1) - x - n - y
# border = sp.Rational(1, 2) * x * n
# print(border)

# Substitution
# math_expression.subs(variable_to_sub, substitute)

########### Presentation of the expressions ##################################################
# sp.gcd(expression_1, expression_2) -> greatest common divisor between 2 expressions
# sp.factor(expression) -> function to factorize the expression. Opposite is sp.expand()
# sp.simplify(expression) -> will set everything on common divisior to simplify reading

# sp.Eq(expression) -> Equality function. Allows to put an equal sign in an expression.
# Otherwise, we get an error because there will be 2 equal signs in the same line or the 
# equality sign will be at the wrong place according to Python

########################### Solving Expressions ##############################################
# sp.solve() -> returns a list. Solves the given expression
# sp.solveset() -> returns a sympy Set object. Handles better several edge cases than sp.solve()
# Q why solveset instead of solve?
# https://docs.sympy.org/latest/modules/solvers/solveset.html

# Q: How to make subscript or superscript with symbols?

#################################### Matrices ################################################
xy = Matrix([x, y])
print('Matrix xy: ', xy)

abcd = Matrix([[1, 2],[7, 8]])
print('Matrix abcd: ', abcd)

mul = xy.transpose()*abcd*xy
mul = mul.expand()

mul 
print('result: ', mul)