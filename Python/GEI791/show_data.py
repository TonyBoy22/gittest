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

from sklearn.cluster import KMeans


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
    

def get_borders(data: tuple, averages: tuple):
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

    For symbolic expressions
    Cannot compute determinant, but we can substitute it with its expression if we know matrix dimension
    '''
    # for numerical part
    # initialize lists
    A = []
    B = []
    C = []
    cov_list = []
    det_list = []
    inv_cov_list = []
    combination_items = []
    border_list = []
    border_coefs = []

    for i in range(len(data)):
        cov = (np.round(np.cov(np.transpose(data[i])))).astype(int)
        inv_cov = np.linalg.inv(cov)
        cov_list.append(cov)
        inv_cov_list.append(inv_cov)

        det = np.linalg.det(cov)# .as_integer_ratio()
        # det = int(fractions.Fraction(*det).limit_denominator())
        det_list.append(det)

    for item in combinations(range(len(data)), 2):
        combination_items.append(item)
        # print('item: ', item, item[0], item[1])
        a = np.array(inv_cov_list[item[1]] - inv_cov_list[item[0]])
        b = np.array([2*(np.dot(inv_cov_list[item[1]], averages[item[1]]) - np.dot(inv_cov_list[item[0]], averages[item[0]]))])
        # print('b shape: ', b.shape)
        d = (np.dot(np.dot(averages[item[0]], inv_cov_list[item[0]]), np.transpose(averages[item[0]])) - \
              np.dot(np.dot(averages[item[1]],inv_cov_list[item[1]]),np.transpose(averages[item[1]])))
        c = np.log(det_list[item[1]]/det_list[item[0]])

        A.append(a)
        B.append(b)
        C.append(c)
        # coef order: [x**2, xy, y**2, x, y]
        border_coefs.append([a[0,0], a[0,1] + a[1,0], a[1, 1], b[0,0], b[0,1], c, d])

    # for symbolic stuff
    sp.init_printing(use_latex=True)
    x, y = sp.symbols('x y')
    xy = Matrix([x, y])
    As = []
    Bs = []
    Cs = []
    avs = []
    cov_lists = []
    det_lists = []
    inv_cov_lists = []
    sym_borders = []
    for i in range(len(data)):
        cov = Matrix(cov_list[i])
        inv_cov = cov.inv()
        cov_lists.append(cov)
        inv_cov_lists.append(inv_cov)

        det = cov.det()
        det_lists.append(det)
        avs.append(Matrix(np.round(averages[i]).astype(int)))

    for item in combinations(range(len(data)), 2):
        # print('item: ', item, item[0], item[1])
        a = inv_cov_lists[item[1]] - inv_cov_lists[item[0]]
        b = 2*(inv_cov_lists[item[1]]*avs[item[1]] - inv_cov_lists[item[0]]*avs[item[0]])
        c = sp.log(det_lists[item[1]], det_lists[item[0]])
        As.append(a)
        Bs.append(b)
        Cs.append(c)

        # Display the symbolic expression
        print(f'border between classes {item[1]} and {item[0]}')
        # border_12 = xy.transpose()*A_12*xy + b_12.transpose()*xy
        border = xy.transpose()*a*xy + b.transpose()*xy
        border = border.expand()

        # # print(f'{border_12[0]} = {-c_12}')
        sp.pprint(sp.Eq(border[0] - c))
    return border_list, border_coefs
        

def plot_figures(data, averages, border_coefs):
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

    #confidence_ellipse(C1, ax4, edgecolor='red')
    #confidence_ellipse(C2, ax4, edgecolor='green')
    #confidence_ellipse(C3, ax4, edgecolor='blue')
    ax4.set_title('Données 3 classes')
    ax4.scatter(C1[:,0], C1[:,1], c='orange')
    ax4.scatter(C2[:,0], C2[:,1], c='purple')
    ax4.scatter(C3[:,0], C3[:,1], c='black')
    ax4.scatter(m1[0], m1[1], c='red')
    ax4.scatter(m2[0], m2[1], c='green')
    ax4.scatter(m3[0], m3[1], c='blue')
    ax4.legend('')
    # Add the borders here
    x, y = np.meshgrid(np.linspace(-10, 10, 400),
                       np.linspace(-8, 15, 400))
    for i in range(len(data)):
        ax4.contour(x, y,
                 coefs[i][0] * x ** 2 + coefs[i][2] * y ** 2 - coefs[i][3] * x - coefs[i][6] +\
                 coefs[i][1]*x * y - coefs[i][4] * y, [-coefs[i][5]])



    plt.show()
    return None 


def kmeans_classification(data):
    '''
    Take the borders between classes then
    add it to plotted data points
    :param data: tuple containing data points
    :return:
    '''
    # convert tuple of data to an array
    data = list(data)
    for i in range(len(data)):
        class_label = np.ones((1000,1))*(i+1)
        data[i] = np.concatenate((data[i], class_label), axis=1)
        # print('class shape: ', data[i].shape)

    data = np.array([*data])
    #print('data shape', data.shape)

    # Flattening array
    x, y, z = data.shape
    data_2d = data.reshape(x*y, z)

    # Only euclidean distance in the function
    # TODO: informer que Euclidian only
    kmeans_classifier = KMeans(n_clusters=3)
    kmeans_classifier.fit(data_2d[:,:2])
    # print('clusters: ', kmeans_classifier.cluster_centers_)
    # print('labels: ', kmeans_classifier.labels_)

    # Check if labels are equal to class labels
    figK, (ax1, ax2) = plt.subplots(2,1)
    # ax1 reference and ax2 kmeans result
    ax1.scatter(data_2d[:,0], data_2d[:,1], c=data_2d[:,2], cmap='viridis')
    ax2.scatter(data_2d[:,0], data_2d[:,1], c=kmeans_classifier.labels_, cmap='viridis')

    counter = 0
    for i in range(len(kmeans_classifier.labels_)):
        if (data_2d[i,2]-1) != kmeans_classifier.labels_[i]:
            counter += 1
    good_classification = (data_2d.shape[0]-counter)/data_2d.shape[0]*100
    print('correctement classé par kmeans: ', good_classification)

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

borders, coefs = get_borders(data, averages)
# print(borders)
plot_figures(data, averages, coefs)


kmeans_classification(data)





