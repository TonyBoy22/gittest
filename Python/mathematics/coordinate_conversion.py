'''
Move a rectagle from an offset and a point

legend:
ur: upper right
ul: upper left
lr: lower right
ll: lower left

applies a for loop to change angles

'''

import numpy as np
import matplotlib.pyplot as plt

angle           = np.pi/4
center_location = (1.0, 1.0) # [x, y]
offset          = 5.0
rectangle_dims  = (5.0, 5.0) # [x, y]
rectangle_center = (center_location[0] + np.cos(angle)*offset, \
    center_location[1] + np.sin(angle)*offset)


x_ur = center_location[0] + np.cos(angle)*offset + \
    np.cos(angle)*rectangle_dims[0] - np.sin(angle)*rectangle_dims[1]
y_ur = center_location[1] + np.sin(angle)*offset + \
    np.sin(angle)*rectangle_dims[0] + np.cos(angle)*rectangle_dims[1]    
edge_ur = np.array([x_ur, y_ur])

x_lr = center_location[0] + np.cos(angle)*offset + \
    np.cos(angle)*rectangle_dims[0] - np.sin(angle)*(-1)*rectangle_dims[1]
y_lr = center_location[1] + np.sin(angle)*offset + \
    np.sin(angle)*rectangle_dims[0] + np.cos(angle)*(-1)*rectangle_dims[1]
edge_lr = np.array([x_lr, y_lr])

x_ll = center_location[0] + np.cos(angle)*offset + \
    np.cos(angle)*(-1)*rectangle_dims[0] - np.sin(angle)*(-1)*rectangle_dims[1]
y_ll = center_location[1] + np.sin(angle)*offset + \
    np.sin(angle)*(-1)*rectangle_dims[0] + np.cos(angle)*(-1)*rectangle_dims[1]
edge_ll = np.array([x_ll, y_ll])

x_ul = center_location[0] + np.cos(angle)*offset + \
    np.cos(angle)*(-1)*rectangle_dims[0] - np.sin(angle)*rectangle_dims[1]
y_ul = center_location[1] + np.sin(angle)*offset + \
    np.sin(angle)*(-1)*rectangle_dims[0] + np.cos(angle)*rectangle_dims[1]
edge_ul = np.array([x_ul, y_ul])
# ===================================================================================================
# x_ur = center_location[0] + \
#     np.cos(angle)*rectangle_dims[0] - np.sin(angle)*rectangle_dims[1]
# y_ur = center_location[1] + \
#     np.sin(angle)*rectangle_dims[0] + np.cos(angle)*rectangle_dims[1]    
# edge_ur = np.array([x_ur, y_ur])

# x_lr = center_location[0] + \
#     np.cos(angle)*rectangle_dims[0] - np.sin(angle)*(-1)*rectangle_dims[1]
# y_lr = center_location[1] + \
#     np.sin(angle)*rectangle_dims[0] + np.cos(angle)*(-1)*rectangle_dims[1]
# edge_lr = np.array([x_lr, y_lr])

# x_ll = center_location[0] + \
#     np.cos(angle)*(-1)*rectangle_dims[0] - np.sin(angle)*(-1)*rectangle_dims[1]
# y_ll = center_location[1] + \
#     np.sin(angle)*(-1)*rectangle_dims[0] + np.cos(angle)*(-1)*rectangle_dims[1]
# edge_ll = np.array([x_ll, y_ll])

# x_ul = center_location[0] + \
#     np.cos(angle)*(-1)*rectangle_dims[0] - np.sin(angle)*rectangle_dims[1]
# y_ul = center_location[1] + \
#     np.sin(angle)*(-1)*rectangle_dims[0] + np.cos(angle)*rectangle_dims[1]
# edge_ul = np.array([x_ul, y_ul])

edges = np.array([edge_ur, edge_lr, edge_ll, edge_ul])

plt.scatter(center_location[0], center_location[1], c='green')
plt.scatter(edges[:,0], edges[:,1], c='purple')
plt.scatter(rectangle_center[0], rectangle_center[1], c='black')
plt.ylim(-15, 15)
plt.xlim(-15, 15)
plt.show()