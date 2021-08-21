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
    cov1 = np.cov(np.transpose(data[0]))
    det_1 = np.linalg.det(cov1)
    inv_cov1 = np.linalg.inv(cov1)
    
    cov2 = np.cov(np.transpose(data[1]))
    det_2 = np.linalg.det(cov2)
    inv_cov2 = np.linalg.inv(cov2)
    
    cov3 = np.cov(np.transpose(data[2]))
    det_3 = np.linalg.det(cov3)
    inv_cov3 = np.linalg.inv(cov3)
    
    xy = Matrix([x, y])
    
    A_12 = inv_cov1 - inv_cov2
    b = 2*(inv_cov2*averages[1] - inv_cov1*averages[0])
    c = (np.dot(np.dot(averages[0],inv_cov1),averages[0]) - \
        np.dot(np.dot(averages[0],inv_cov1),averages[0])) + np.log((det_2)/(det_1))
    borders = []
    # for i in combinations('123', 2):
    #     print(i[0])
    
    # for border 12
        
    return borders
        

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


# Calcul symbolique




