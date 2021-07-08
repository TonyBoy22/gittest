'''
implementation of Bentley-Ottman algorithm

Doc Sources
linesegmentintersection:
https://pypi.org/project/line-segment-intersections/

bentley_ottmann
'''

from linesegmentintersections import bentley_ottman
import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix(angle):
    # Work in radian
    _rotation_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return _rotation_mat

# linesegmentintersection requires a 3D list for the segments vertices
segment_vertices = [[[0,0],[1, 1]],[[0, -1],[-1, -1]]]
intersections = bentley_ottman(segment_vertices)
# print('intersections: ', intersections)
# print('x coordinates: ', intersections[0].x)

# Returns intersection object with attributes x and y

from ground.base import get_context
from bentley_ottmann.planar import segments_intersect
context = get_context()
# Uses Point and Segment objects instead of a 3D list 
Point, Segment = context.point_cls, context.segment_cls
unit_segments = [Segment(Point(0, 0), Point(1, 0)), 
                  Segment(Point(0, 0), Point(0, 1))]
# print('bool are segment intersecting: ', segments_intersect(unit_segments))


# Ratio differences:
rectangle = np.array([[1,1], [1, -1], [-1, -1], [-1, 1]])
angle = np.pi/4
obstacle_coordinates = np.array([[-5, 1], [5, -1]])
transpose_coordinates = np.transpose(obstacle_coordinates)
print(transpose_coordinates)

rotated_rectangle = np.dot(rotation_matrix(angle), np.transpose(rectangle))
# plt.scatter(rotated_rectangle)

print('rotated_rectangle: ', rotated_rectangle)

backward_rotated_rectangle = np.dot(rotation_matrix(-angle), rotated_rectangle)
print('backward_rot: ', backward_rotated_rectangle)
rectangle_centered_coordinates = np.dot(rotation_matrix(-angle), np.transpose(obstacle_coordinates))
print('rectangle_centered_coordinates: ', rectangle_centered_coordinates)

# 