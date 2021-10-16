from matplotlib import pyplot as plt
import numpy as np

print('Game over')
print('blabla')
name = "Alex"


def create_meshgrid():
    '''
    Lets' plot the contours of a non linear function given by
    a*x**2 + b*y**2 + c*xy + d*x + e*y = f
    https://stackoverflow.com/questions/51794943/python-plot-non-linear-equation

    :return:
    '''
    delta = 0.025

    fig = plt.figure()
    ax = fig.add_subplot()
    xy = np.random.randint(-20, 20, size=(2,400))
    ax.scatter(xy[0,:], xy[1,:])
    x, y = np.meshgrid(np.linspace(-20, 20, 400),
                       np.linspace(-20, 20, 400))
    ax.contour(x, y,
                7/12*x**2 + 5/6*y**2 - 7/6*x - 53/12 - np.log(24)\
                + x*y/3 + 5/3*y, [0])
    plt.show()
    return None

create_meshgrid()
# Contour added to a plot on a figure object
