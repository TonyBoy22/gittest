'''
Script to implement the verification if a point is located inside a random polygon

1st method, the barycentric coordinates

2nd method, parametric equation system

then, the even/odd triangles of the polygon that contains the point
'''

from math import sqrt
import numpy as np
from numpy.lib.polynomial import poly
import matplotlib.pyplot as plt

def display_results(polygon, point):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(polygon[:,0], polygon[:,1])
    ax.scatter(point[0], point[1], color='black')
    plt.show()

def triangle_decomposition(polygon):
    triangles = []
    for i in range(len(polygon)-2):
        triangle = [polygon[0,0], polygon[0,1], polygon[i+1,0], polygon[i+1,1], polygon[i+2,0], polygon[i+2,1]]
        triangles.append(triangle)
    return triangles

def barycentric_coordinates(triangles, point):
    '''
    formula from http://totologic.blogspot.com/2014/01/accurate-point-in-triangle-test.html
    triangle defined by p1, p2, p3 := [x1, y1], [x2, y2], [x3, y3] and a random dot p [x, y]
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3))/((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3))/((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    c = 1 - a - b
    '''
    number_of_occurence = 0
    x, y = point
    x1, y1 = triangles[0][0:2]
    for i in range(len(triangles)):
        x2, y2 = triangles[i][2:4]
        x3, y3 = triangles[i][4:]
        a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3))/((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
        b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3))/((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
        c = 1 - a - b
        print('a, b, c: ', a, b, c)
        print('x1, x2, x3, y1, y2, y3, x, y: ', x1, x2, x3, y1, y2, y3, x, y)
        if (a >= 0 and a <= 1) and (b >= 0 and b <= 1) and (c >= 0 and c <= 1):
        # Source of principle: https://www.youtube.com/watch?v=Nrf6yLuBSTI
            number_of_occurence += 1
    return number_of_occurence
## ============================================================================
# Polygon creation
# x = np.random.randint(0, 10, size=(np.random.randint(4, 9), 1))
# y = np.random.randint(0, 10, size=np.shape(x))

# print('x, y: ', x, y)
# polygon = np.concatenate((x, y), axis = 1) # Sequence of vertices
# '''
# polygon format = [[x1, y1],
#                   [x2, y2],
#                   ...
#                   [xn, yn]]
# '''
# print(polygon)

# point = [np.random.randint(0,len(polygon)), np.random.randint(0,len(polygon))]

# ''' triangles from the polygons in the format
# triangles = [[x1, y1, x2, y2, x3, y3],
#              [x1, y1, x3, y3, x4, y4],
#              ...
#              [x1, y1, xn-1, yn-1, xn, yn]] '''
# triangles = triangle_decomposition(polygon)
# occurence = barycentric_coordinates(triangles, point)
# display_results(polygon, point)
# if occurence % 2 == 1:
#     print('point is in polygon!')
# else:
#     print('point outside polygon')


# ================================================== Ray Tracing ===============================================
# import numba
# Faster approach than


# ================================================== Autres modules ===========================================
# import shapely
from matplotlib.patches import Patch, Polygon
polygon = Polygon(np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]))
point = ([0, 0])

ans = polygon.contains_point(point)
print(ans)