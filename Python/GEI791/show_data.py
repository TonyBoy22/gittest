'''
Author: Antoine Marion
Date: 
'''


import numpy as np
from numpy.lib.function_base import average
import sympy as sp
from sympy.matrices import Matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from itertools import combinations
import fractions
import pprint


def confidence_ellipse(data, ax, n_std=3.0, facecolor='none', **kwargs):
    '''
    Inspiration from matplotlib documentation on 'Plot a confidence ellipse'
    
    class data format np.array([[],
                                [],
                                ...
                                []]
    ax: axis object from matplotlib figure
    n_std: Number of standard deviations for confidence level
    facecolor and kwargs: Argument for the plotting of Ellipse
    '''
    cov = np.cov(np.transpose(data))
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, \
        facecolor=facecolor, **kwargs)
    
        # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(data[:,0])

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(data[:,1])

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    
    return ax.add_patch(ellipse)
    

def get_borders(data, averages: tuple):
    '''
    data format: [C1, C2, C3]
    
    averages: (m1, m2, m3)
    
    steps
    get inverse of covariance matrices from each class
    use averages to computes coefficients for border formula
    g(y) = y*A*y + b*y + C where
    A = inv(cov_1) - inv(cov_2)
    b = 2*(inv(cov_2)*m2 - inv(cov_1)*m1)
    c = (transp(m1)*inv(cov_1)*m1 - transp(m2)*inv(cov_2)*m2) + ln(det(cov_2)/det(cov_1))
    '''
    x, y = sp.symbols('x y')
    C1 = data[0]
    C2 = data[1]
    C3 = data[2]
    
    # cov and inverse, then Matrix([inv_cov1[], inv_cov2[]])
    cov1 = (np.round(np.cov(np.transpose(data[0])))).astype(int)
    # print('cov1', cov1)
    det_1 = np.linalg.det(cov1).as_integer_ratio()
    det_1 = fractions.Fraction(*det_1).limit_denominator()
    
    cov1 = Matrix(cov1)
    # print('cov1: ', cov1)
    inv_cov1 = cov1.inv()
    # print('inv_cov1', inv_cov1)
    
    cov2 = (np.round(np.cov(np.transpose(data[1])))).astype(int)
    det_2 = np.linalg.det(cov2).as_integer_ratio()
    det_2 = fractions.Fraction(*det_2).limit_denominator()
    cov2 = Matrix(cov2)
    inv_cov2 = cov2.inv()
    
    cov3 = (np.round(np.cov(np.transpose(data[2])))).astype(int)
    det_3 = np.linalg.det(cov3).as_integer_ratio()
    det_3 = fractions.Fraction(*det_3).limit_denominator()
    cov3 = Matrix(cov3)
    inv_cov3 = cov3.inv()
    
    # Variables
    xy = Matrix([x, y])
    av1 = Matrix(np.round(averages[0]).astype(int))
    av2 = Matrix(np.round(averages[1]).astype(int))
    av3 = Matrix(np.round(averages[2]).astype(int))
    
    A_12 = inv_cov1 - inv_cov2
    # print('A_12', A_12)
    b_12 = (2*(inv_cov2*av2 - inv_cov1*av1))
    # c_12 = (np.dot(np.dot(av1.transpose(),inv_cov1),av1) - \
    #     np.dot(np.dot(av2.transpose(),inv_cov2),av2)) + np.log((det_2)/(det_1))
    c_12 = sp.log(sp.Rational(det_2, det_1))
    borders = []
    # for i in combinations('123', 2):
    #     print(i[0])
    border_12 = xy.transpose()*A_12*xy + b_12.transpose()*xy
    border_12 = border_12.expand()
    
    sp.init_printing(use_latex=True)
    # print(f'{border_12[0]} = {-c_12}')
    sp.pprint(sp.Eq(border_12[0] - c_12))
       
    # for border 2-3
    # A_23 = Matrix(inv_cov2 - inv_cov3)
    # b_23 = Matrix(2*(inv_cov3*av3 - inv_cov2*av2))
    # c_23 = (np.dot(np.dot(av2.transpose(),inv_cov2),av2) - \
    #     np.dot(np.dot(av3.transpose(),inv_cov3),av3))# + np.log((det_3)/(det_2))
        
    # border_23 = xy.transpose()*A_23*xy + b_23.transpose()*xy + c_23
    # border_23 = border_23.expand()
    # print('border expression: ', border_23[0])
    
    # # for border 1-3
    # A_13 = Matrix(inv_cov1 - inv_cov3)
    # b_13 = Matrix(2*(inv_cov3*av3 - inv_cov1*av1))
    # c_13 = (np.dot(np.dot(av1.transpose(),inv_cov1),av1) - \
    #     np.dot(np.dot(av3.transpose(),inv_cov3),av3))+ np.log((det_3)/(det_1))
        
    # border_13 = xy.transpose()*A_13*xy + b_13.transpose()*xy + c_13
    # border_13 = border_13.expand()
    # print('border expression: ', border_13[0])
    
    
    return None
        

def plot_figures(data, averages):
    '''
    Plotting figures
    '''
    C1, C2, C3 = data
    m1, m2, m3 = averages
    fig1, ax1 = plt.subplots(1,1)

    confidence_ellipse(C1, ax1, edgecolor='red')
    ax1.set_title('Données C1.txt')
    ax1.scatter(C1[:,0], C1[:,1])
    ax1.scatter(m1[0], m1[1], c='red')

    fig2, ax2 = plt.subplots(1,1)

    confidence_ellipse(C2, ax2, edgecolor='green')
    ax2.set_title('Données C2.txt')
    ax2.scatter(C2[:,0], C2[:,1])
    ax2.scatter(m2[0], m2[1], c='red')

    fig3, ax3 = plt.subplots(1,1)

    confidence_ellipse(C3, ax3, edgecolor='blue')
    ax3.set_title('Données C3.txt')
    ax3.scatter(C3[:,0], C3[:,1])
    ax3.scatter(m3[0], m3[1], c='red')

    # All data
    fig4, ax4 = plt.subplots(1,1)

    confidence_ellipse(C1, ax4, edgecolor='red')
    confidence_ellipse(C2, ax4, edgecolor='green')
    confidence_ellipse(C3, ax4, edgecolor='blue')
    ax4.set_title('Données 3 classes')
    ax4.scatter(C1[:,0], C1[:,1], c='orange')
    ax4.scatter(C2[:,0], C2[:,1], c='purple')
    ax4.scatter(C3[:,0], C3[:,1], c='black')
    ax4.scatter(m1[0], m1[1], c='red')
    ax4.scatter(m2[0], m2[1], c='green')
    ax4.scatter(m3[0], m3[1], c='blue')
    ax4.legend('')

    plt.show()
    return None 


# Import from text files
C1 = np.loadtxt('C1.txt')
C2 = np.loadtxt('C2.txt')
C3 = np.loadtxt('C3.txt')
data = (C1, C2, C3)

# Averages
m1 = np.average(C1, axis=0)
m2 = np.average(C2, axis=0)
m3 = np.average(C3, axis=0)
averages = (m1, m2, m3)

# plot_figures(data, averages)

borders = get_borders(data, averages)





