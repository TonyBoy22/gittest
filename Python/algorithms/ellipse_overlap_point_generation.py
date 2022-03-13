'''
Script where we first plot 2 overlapping ellipses and we plot 
again the ellipses with dots inside of different colors for
different sections
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky



# Generate 2 ellipses
# format = (a, b, h, k, yaw)

'''
equation of a rotated ellipse
((x - h)cos(yaw) + (y - k)sin(yaw))**2/a**2 \
    + ((x - h)sin(yaw) - (y - k)cos(yaw))**2/b**2 = 1
    
Better use parametric equation

x(alpha) = Rx*cos(alpha)*cos(yaw) - Ry*sin(alpha)sin(yaw) + Cx
y(alpha) = Rx*cos(alpha)*sin(yaw) - Ry*sin(alpha)cos(yaw) + Cy

Cx = h
Cy = k
Rx = a
Ry = b
alpha = range(0 -> 2pi)
yaw = ellipse rotation angle

https://math.stackexchange.com/questions/2645689/what-is-the-parametric-equation-of-a-rotated-ellipse-given-the-angle-of-rotatio
'''


def rotation_mat(yaw):
    _rotation_mat = np.array([[np.cos(yaw), -np.sin(yaw)],[np.sin(yaw), np.cos(yaw)]])
    return _rotation_mat

def random_point_generation(a, b, h, k, yaw):
    
    rho = np.random.rand(500)
    phi = np.random.rand(500)*2*np.pi
    x = np.sqrt(rho)*np.cos(phi)*a
    y = np.sqrt(rho)*np.sin(phi)*b
    
    r = rotation_mat(yaw)
    # rotation matrix
    dots = np.dot(r, np.array([x, y]))
    x = dots[0,:] + h
    y = dots[1,:] + k
    
    # plt.scatter(x, y)
    # plt.show()
    return x,y



ellipse_1 = (5,2,0,0,np.pi/4)
ellipse_2 = (5,2,4,0,np.pi/3)

# Plot ellipses from matplotlib function
# Scatter couple random points with formulas above for x and y

# For first ellipse
alpha = np.linspace(0,2*np.pi,num=360)
x1 = ellipse_1[0]*np.cos(alpha)*np.cos(ellipse_1[4]) - \
    ellipse_1[1]*np.sin(alpha)*np.sin(ellipse_1[4]) + ellipse_1[2]
y1 = ellipse_1[0]*np.cos(alpha)*np.sin(ellipse_1[4]) + \
    ellipse_1[1]*np.sin(alpha)*np.cos(ellipse_1[4]) + ellipse_1[3]
    
plt.scatter(x1,y1, label='pose de l\'égo véhicule')

# for 2nd ellipse
x2 = ellipse_2[0]*np.cos(alpha)*np.cos(ellipse_2[4]) - \
    ellipse_2[1]*np.sin(alpha)*np.sin(ellipse_2[4]) + ellipse_2[2]
y2 = ellipse_2[0]*np.cos(alpha)*np.sin(ellipse_2[4]) + \
    ellipse_2[1]*np.sin(alpha)*np.cos(ellipse_2[4]) + ellipse_2[3]
    
plt.scatter(x2,y2, label='pose de l\'obstacle')

# Generate dots inside ellipse 1
x,y = random_point_generation(*ellipse_1)

 
# Check if they are in ellipse 2
# Apply cartesian equation of ellipse 2 for each dots and filter amount that
fraction = ((x-ellipse_2[2])*np.cos(ellipse_2[4]) + \
    (y - ellipse_2[3])*np.sin(ellipse_2[4]))**2/ellipse_2[0]**2 + \
        ((x - ellipse_2[2])*np.sin(ellipse_2[4]) - \
            (y - ellipse_2[3])*np.cos(ellipse_2[4]))**2/ellipse_2[1]**2
# gives a result above or below 1
fraction = np.array(fraction, dtype='int')
fraction = np.array(fraction, dtype='bool')
fraction = 1-sum(fraction)/len(fraction)
fraction = format(fraction, ".2%")
print('fraction: ', fraction)

plt.scatter(x,y)
# plt.xlabel('x coordinate')
# plt.ylabel('y coordinate')
plt.xlabel('coordonnées en x')
plt.ylabel('coordonnées en y')
plt.title("ellipse d'incertitude des véhicules")
plt.text(-4, 3, f"fraction: {fraction}")
plt.legend()
plt.show()